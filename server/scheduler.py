"""Lanes and the scheduler — the heart of the design (§3, §4). A lane is a capacity; every job belongs to exactly one lane at a time.
Once per event (a job arrives, a job ends, a one-second tick) the scheduler expires what waited too long, moves what may fall back,
and fills every free slot: priority descending, then fair share across keys, then arrival, with the local lane's model affinity
inside a bounded window. A started job never moves."""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator

from .backends.base import Backend, BackendError, ModelMissing
from .store import Store, now_iso, parse_iso

LANES = ("local", "cloud", "maintenance")
PRIORITY_DEFAULT = 5


class Refusal(Exception):
    """A request refused before it became a job (the faces answer the provider's error shape with this status)."""
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


class Lane:
    def __init__(self, name: str, capacity: int):
        self.name, self.capacity = name, capacity      # 0 = unbounded
        self.running: set[str] = set()
        self.paused = False

    def free(self) -> bool:
        return not self.paused and (self.capacity == 0 or len(self.running) < self.capacity)

    def snapshot(self) -> dict:
        return {"name": self.name, "capacity": self.capacity or None, "running": len(self.running), "paused": self.paused}


class Scheduler:
    def __init__(self, store: Store, backend: Backend):
        self.store, self.backend = store, backend
        self.lanes: dict[str, Lane] = {"local": Lane("local", 1), "cloud": Lane("cloud", 0), "maintenance": Lane("maintenance", 1)}
        self.tasks: dict[str, asyncio.Task] = {}
        self.chunks: dict[str, list[dict]] = {}             # a running job's chunks, for subscribers' replay
        self.subscribers: dict[str, list[asyncio.Queue]] = {}
        self.last_served: dict[str, float] = {}             # key_id -> monotonic time of its last dispatch (fair share)
        self.wake = asyncio.Event()
        self.started_at = now_iso()
        self._loop_task: asyncio.Task | None = None
        self._settings: dict = {}
        self._loaded_cache: tuple[float, set[str]] = (0.0, set())
        self.listeners: list[asyncio.Queue] = []            # node events for /events
        self.ended: dict[str, asyncio.Event] = {}           # set when a job leaves queued/running

    # -- lifecycle --------------------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.reload_settings()
        # a restart: whatever was running is lost — mark it failed so nothing hangs
        for r in await self.store.running():
            self.lanes[r["lane_served"]].running.discard(r["id"])
            await self.store.update_run(r["id"], {"status": "failed", "finished_at": now_iso(), "error_code": "node_restarted",
                                                   "error_text": "the node restarted while this job ran"}, ("failed", {"code": "node_restarted"}))
        self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
        for t in list(self.tasks.values()):
            t.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)

    async def reload_settings(self) -> None:
        self._settings = await self.store.settings()
        self.lanes["local"].capacity = int(self._settings["local_parallel"])
        self.lanes["cloud"].capacity = int(self._settings["cloud_parallel"])
        self.wake.set()

    def setting(self, key: str) -> Any:
        return self._settings[key]

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self.wake.clear()
            try:
                await self.tick()
            except Exception as e:          # the loop must never die
                await self._emit("scheduler_error", {"error": repr(e)})

    # -- accepting a job ------------------------------------------------------------------------------------------------------
    async def resolve_model(self, name: str) -> dict:
        if not name:
            name = self.setting("default_model")
        if not name:
            raise Refusal(400, "model_required", "name a model")
        row = await self.store.model(name)
        if not row:
            raise Refusal(404, "model_unknown", f"model {name} is not on this node")
        if not row["allowed"] or row["status"] == "disabled":
            raise Refusal(404, "model_unknown", f"model {row['name']} is not served here")
        return row

    async def submit(self, *, key: dict, face: str, request: dict, node: dict | None, mode: str = "sync", kind: str = "chat",
                     client_ip: str | None = None, user_agent: str | None = None, stream: bool = False) -> dict:
        """Validates against the key's limits and the catalogue, decides the lane and the fallback, inserts the job `queued`."""
        node = dict(node or {})
        unknown = set(node) - {"priority", "mode", "fallback", "max_wait_seconds", "max_queue_seconds", "keep_alive", "webhook_url", "tags", "idempotency_key"}
        if unknown:
            raise Refusal(400, "bad_node_field", f"unknown node field(s): {', '.join(sorted(unknown))}")
        mode = node.get("mode", mode)
        if mode not in ("sync", "async"):
            raise Refusal(400, "bad_mode", "mode must be sync or async")
        idem = node.get("idempotency_key")
        if idem:
            prior = await self.store.find_idempotent(key["id"], str(idem))
            if prior:
                return prior
        model = await self.resolve_model(request.get("model", ""))
        if key.get("allowed_models") and model["name"] not in key["allowed_models"] and request.get("model") not in key["allowed_models"]:
            raise Refusal(403, "model_not_allowed", f"this key may not use {model['name']}")
        lane = model["lane_override"] or model["lane"]
        if lane == "cloud" and not key["cloud_allowed"]:
            raise Refusal(403, "cloud_not_allowed", "this key may not run in the cloud lane")
        if await self.store.in_flight_for_key(key["id"]) >= int(key["max_concurrent"]):
            raise Refusal(429, "too_many_in_flight", f"this key has {key['max_concurrent']} jobs in flight", retry_after=5)
        if key.get("daily_token_cap") and await self.store.tokens_today(key["id"]) >= int(key["daily_token_cap"]):
            raise Refusal(429, "daily_cap", "this key's daily token cap is reached", retry_after=3600)
        try:
            priority = int(node.get("priority", PRIORITY_DEFAULT))
        except (TypeError, ValueError):
            raise Refusal(400, "bad_priority", "priority must be 1–9")
        if not 1 <= priority <= 9:
            raise Refusal(400, "bad_priority", "priority must be 1–9")
        priority = min(priority, int(key["max_priority"]))
        # the fallback (§3): the key AND (the request OR the model) — default none
        fb = node.get("fallback")
        fallback_model, fallback_after = None, None
        if fb is not None and fb != "none" and not isinstance(fb, dict):
            raise Refusal(400, "bad_fallback", 'fallback must be "none" or {"model", "after_seconds"}')
        if fb != "none" and key["fallback_allowed"] and lane == "local":
            if isinstance(fb, dict):
                fallback_model = fb.get("model") or model["fallback_model"]
                fallback_after = fb.get("after_seconds")
            elif fb is None:
                fallback_model = model["fallback_model"]
            else:
                raise Refusal(400, "bad_fallback", 'fallback must be "none" or {"model", "after_seconds"}')
            if fallback_model:
                fbrow = await self.store.model(fallback_model)
                if not fbrow or (fbrow["lane_override"] or fbrow["lane"]) != "cloud" or not fbrow["allowed"]:
                    raise Refusal(400, "bad_fallback", f"fallback model {fallback_model} is not a cloud model on this node")
                if key.get("allowed_models") and fbrow["name"] not in key["allowed_models"]:
                    raise Refusal(403, "model_not_allowed", f"this key may not use the fallback {fbrow['name']}")
                if not key["cloud_allowed"]:
                    raise Refusal(403, "cloud_not_allowed", "a fallback needs a key allowed in the cloud lane")
                fallback_model = fbrow["name"]
                fallback_after = int(fallback_after if fallback_after is not None else (model["fallback_after_seconds"] or self.setting("fallback_after_seconds")))
        if lane == "local" and not await self.backend.health():
            if not fallback_model:
                raise Refusal(503, "backend_down", "the local backend is not answering", retry_after=15)
        tags = node.get("tags") or []
        if not isinstance(tags, list) or any(not isinstance(t, str) for t in tags):
            raise Refusal(400, "bad_tags", "tags must be a list of strings")
        job = {
            "id": str(uuid.uuid4()), "face": face, "mode": mode, "kind": kind, "key_id": key["id"], "key_name": key["name"],
            "tags": json.dumps(tags), "idempotency_key": str(idem) if idem else None, "client_ip": client_ip, "user_agent": user_agent,
            "requested_model": request.get("model") or model["name"], "served_model": model["name"], "lane_requested": lane, "lane_served": lane,
            "fallback_model": fallback_model, "fallback_after_seconds": fallback_after, "priority": priority, "stream": int(bool(stream)),
            "status": "queued", "accepted_at": now_iso(),
            "request_bytes": len(json.dumps(request)), "messages_count": len(request.get("messages", []) or []),
            "images_count": sum(1 for m in request.get("messages", []) or [] for p in (m.get("content") if isinstance(m.get("content"), list) else []) if isinstance(p, dict) and p.get("type") in ("image", "image_url")),
            "tools_count": len(request.get("tools") or []), "context_length": model["context_length"],
            "max_queue_seconds": node.get("max_queue_seconds"), "max_run_seconds": self.setting("max_run_seconds"),
            "keep_alive": str(node.get("keep_alive")) if node.get("keep_alive") is not None else (model["keep_alive"] or self.setting("keep_alive_default")),
            "webhook_url": node.get("webhook_url"),
        }
        self._ended(job["id"])
        await self.store.insert_run(job)
        await self.store.add_event(job["id"], "queued", {"lane": lane, "priority": priority})
        await self.store.store_payload(job["id"], key["payload_retention"], request=request)
        self.wake.set()
        return await self.store.run(job["id"])

    async def submit_maintenance(self, kind: str, model: str, key: dict | None = None, keep_alive: str | None = None) -> dict:
        """A warm (`load`, the local lane) or a pull / delete (the maintenance lane) as a job."""
        lane = "local" if kind == "load" else "maintenance"
        job = {"id": str(uuid.uuid4()), "face": "node", "mode": "async", "kind": kind, "key_id": key["id"] if key else None,
               "key_name": key["name"] if key else "node", "tags": "[]", "requested_model": model, "served_model": model,
               "lane_requested": lane, "lane_served": lane, "priority": 9 if kind == "load" else 5, "stream": 0, "status": "queued",
               "accepted_at": now_iso(), "keep_alive": keep_alive, "max_run_seconds": 86400}
        self._ended(job["id"])
        await self.store.insert_run(job)
        await self.store.add_event(job["id"], "queued", {"lane": lane, "kind": kind})
        self.wake.set()
        return await self.store.run(job["id"])

    # -- the tick ---------------------------------------------------------------------------------------------------------------
    async def tick(self) -> None:
        now = time.time()
        for j in await self.store.fetchall("SELECT id, max_queue_seconds, accepted_at FROM runs WHERE status = 'queued' AND max_queue_seconds IS NOT NULL"):
            mq = j["max_queue_seconds"]
            if now - (parse_iso(j["accepted_at"]) or now) > int(mq):
                await self.store.update_run(j["id"], {"status": "expired", "finished_at": now_iso(), "error_code": "queue_timeout",
                                                      "error_text": f"waited longer than {mq} s"}, ("expired", {}))
                self._finish_subscribers(j["id"])
        await self.fallback_pass()
        pending = {r["lane_served"] for r in await self.store.fetchall("SELECT DISTINCT lane_served FROM runs WHERE status = 'queued'")}
        for lane in self.lanes.values():
            if lane.name not in pending:
                continue
            while lane.free():
                job = await self.pick(lane)
                if not job:
                    break
                await self._start(lane, job)

    async def loaded_models(self) -> set[str]:
        t, names = self._loaded_cache
        if time.monotonic() - t > 2:
            try:
                names = {m["name"] for m in await self.backend.loaded()}
            except Exception:
                names = set()
            self._loaded_cache = (time.monotonic(), names)
        return names

    async def pick(self, lane: Lane) -> dict | None:
        """Priority descending, then the key served longest ago among the first 50 of that priority, then arrival; in the local
        lane, a job for the loaded model inside the affinity window goes first. Bounded queries: a 10,000-job queue costs the same."""
        top = await self.store.fetchone("SELECT priority FROM runs WHERE status = 'queued' AND lane_served = ? ORDER BY priority DESC LIMIT 1", (lane.name,))
        if not top:
            return None
        group = await self.store.fetchall("SELECT * FROM runs WHERE status = 'queued' AND lane_served = ? AND priority = ? ORDER BY accepted_at, seq LIMIT 50",
                                          (lane.name, top["priority"]))
        chosen = min(group, key=lambda r: (self.last_served.get(r["key_id"] or "", 0.0), r["accepted_at"], r["seq"]))
        if lane.name == "local" and chosen["kind"] == "chat":
            loaded = await self.loaded_models()
            if chosen["served_model"] not in loaded and loaded:
                window = await self.store.fetchall("SELECT * FROM runs WHERE status = 'queued' AND lane_served = ? ORDER BY priority DESC, accepted_at, seq LIMIT ?",
                                                   (lane.name, int(self.setting("affinity_window_jobs"))))
                oldest_age = time.time() - (parse_iso(window[0]["accepted_at"]) or time.time())
                if oldest_age < int(self.setting("affinity_window_seconds")):
                    for r in window:
                        if r["kind"] == "chat" and r["served_model"] in loaded:
                            return r
        return chosen

    async def lane_waits(self, lane_name: str) -> dict[str, int]:
        """expected_wait_ms for every queued job of a lane in ONE pass (the running jobs' remainders, then each job ahead)."""
        lane = self.lanes[lane_name]
        queued = await self.store.queued(lane_name)
        if not queued:
            return {}
        if lane.capacity == 0:
            return {r["id"]: 0 for r in queued}
        loaded = await self.loaded_models()
        cache: dict[str, tuple[int, int]] = {}

        async def exp(model: str) -> tuple[int, int]:
            if model not in cache:
                cache[model] = await self.store.expected_ms(model, self.setting("default_expected_ms"), self.setting("default_load_ms"))
            return cache[model]

        now = time.time()
        ahead = 0.0
        for r in await self.store.running(lane_name):
            e, _ = await exp(r["served_model"])
            ahead += max(0.0, e - (now - (parse_iso(r["started_at"]) or now)) * 1000)
        waits = {}
        for r in queued:
            waits[r["id"]] = int(ahead / lane.capacity)
            e, load = await exp(r["served_model"])
            ahead += e + (0 if r["served_model"] in loaded else load)
        return waits

    async def expected_wait_ms(self, job_id: str) -> int:
        job = await self.store.run(job_id)
        if not job or job["status"] != "queued":
            return 0
        return (await self.lane_waits(job["lane_served"])).get(job_id, 0)

    async def fallback_pass(self) -> None:
        queued = await self.store.fetchall("SELECT * FROM runs WHERE status = 'queued' AND lane_served = 'local' AND fallback_model IS NOT NULL ORDER BY priority DESC, accepted_at")
        if not queued:
            return
        healthy = await self.backend.health()
        waits = await self.lane_waits("local") if healthy else {}
        for j in queued:
            reason = None
            if not healthy:
                reason = "local_unavailable"
            elif waits.get(j["id"], 0) > int(j["fallback_after_seconds"] or self.setting("fallback_after_seconds")) * 1000:
                reason = "local_wait"
            if reason:
                await self.move(j["id"], "cloud", j["fallback_model"], reason)

    async def move(self, job_id: str, lane: str, model: str, reason: str = "forced") -> dict:
        job = await self.store.run(job_id)
        if not job or job["status"] != "queued":
            raise Refusal(409, "not_queued", "only a queued job moves")
        if lane not in self.lanes:
            raise Refusal(400, "bad_lane", "no such lane")
        await self.store.update_run(job_id, {"lane_served": lane, "served_model": model, "fallback_reason": reason if lane != job["lane_requested"] else None},
                                    ("moved", {"from": job["lane_served"], "to": lane, "model": model, "reason": reason}))
        self.wake.set()
        return await self.store.run(job_id)

    # -- running ----------------------------------------------------------------------------------------------------------------
    async def _start(self, lane: Lane, job: dict) -> None:
        lane.running.add(job["id"])
        self.last_served[job["key_id"] or ""] = time.monotonic()
        now = time.time()
        loaded = sorted(await self.loaded_models())
        machine = await self._machine()
        await self.store.update_run(job["id"], {"status": "running", "started_at": now_iso(),
                                                "queue_wait_ms": int((now - (parse_iso(job["accepted_at"]) or now)) * 1000),
                                                "loaded_models_at_start": json.dumps(loaded), "memory_free_at_start_bytes": machine.get("memory_free_bytes"),
                                                "cpu_load_1m": machine.get("cpu_load_1m")}, ("started", {"lane": lane.name}))
        self.chunks[job["id"]] = []
        self.tasks[job["id"]] = asyncio.create_task(self._run(lane, job))

    async def _machine(self) -> dict:
        try:
            m = dict(await self.backend.machine())
        except Exception:
            m = {}
        try:
            m["cpu_load_1m"] = os.getloadavg()[0]
        except (OSError, AttributeError):
            pass
        return m

    async def _run(self, lane: Lane, job: dict) -> None:
        jid = job["id"]
        outcome: dict = {}
        event: tuple[str, dict] = ("done", {})
        status = "done"
        t_start = time.monotonic()
        first = None
        first_at = None
        try:
            async with asyncio.timeout(int(job["max_run_seconds"] or self.setting("max_run_seconds"))):
                if job["kind"] == "chat":
                    payload = await self.store.payload(jid)
                    request = dict((payload or {}).get("request") or {})
                    request["model"] = job["served_model"]
                    request["keep_alive"] = job["keep_alive"]
                    final = None
                    async for chunk in self.backend.chat(request):
                        if chunk["type"] == "delta":
                            if first is None:
                                first = time.monotonic()
                                first_at = now_iso()
                            self.chunks[jid].append(chunk)
                            self._publish(jid, chunk)
                        elif chunk["type"] == "final":
                            final = chunk
                    if final is None:
                        raise BackendError("no_final", "the backend ended without a final message")
                    st = final.get("stats") or {}
                    out_tok = int(st.get("output_tokens") or 0)
                    eval_ms = int(st.get("eval_ms") or 0)
                    p_tok = int(st.get("prompt_tokens") or 0)
                    p_ms = int(st.get("prompt_eval_ms") or 0)
                    outcome = {"stop_reason": final.get("stop_reason"), "load_ms": st.get("load_ms"), "prompt_tokens": p_tok, "prompt_eval_ms": p_ms,
                               "output_tokens": out_tok, "eval_ms": eval_ms, "total_backend_ms": st.get("total_ms"),
                               "prompt_tokens_per_second": round(p_tok / (p_ms / 1000), 2) if p_ms else None,
                               "output_tokens_per_second": round(out_tok / (eval_ms / 1000), 2) if eval_ms else None,
                               "model_was_loaded": int((st.get("load_ms") or 0) < 50), "context_used": p_tok + out_tok,
                               "response_bytes": len(json.dumps(final.get("message") or {})), "http_status": 200}
                    await self.store.store_payload(jid, await self._retention(job), response=final.get("message"), chunks=self.chunks[jid])
                    self.chunks[jid].append({"type": "final", "stop_reason": final.get("stop_reason"), "stats": st, "message": final.get("message")})
                    self._publish(jid, self.chunks[jid][-1])
                elif job["kind"] == "embed":
                    payload = await self.store.payload(jid)
                    request = dict((payload or {}).get("request") or {})
                    request["model"] = job["served_model"]
                    res = await self.backend.embed(request)
                    st = res.get("stats") or {}
                    outcome = {"stop_reason": "stop", "prompt_tokens": st.get("prompt_tokens"), "total_backend_ms": st.get("total_ms"),
                               "response_bytes": len(json.dumps(res.get("embeddings") or [])), "http_status": 200}
                    await self.store.store_payload(jid, await self._retention(job), response={"embeddings": res.get("embeddings")})
                    self._publish(jid, {"type": "final", "embeddings": res.get("embeddings"), "stats": st})
                elif job["kind"] == "load":
                    res = await self.backend.load(job["served_model"], job["keep_alive"])
                    outcome = {"load_ms": res.get("load_ms"), "total_backend_ms": res.get("load_ms"), "http_status": 200}
                    await self.store.upsert_model(job["served_model"], loaded=1)
                elif job["kind"] == "pull":
                    await self.store.upsert_model(job["served_model"], status="pulling", pull_error=None)
                    last = {}
                    async for p in self.backend.pull(job["served_model"]):
                        last = p
                        self.chunks[jid].append({"type": "progress", **p})
                        self._publish(jid, self.chunks[jid][-1])
                    outcome = {"http_status": 200, "response_bytes": int(last.get("total") or 0)}
                    await self.refresh_models()
                elif job["kind"] == "delete":
                    await self.backend.delete(job["served_model"])
                    await self.store.delete_model(job["served_model"])
                    outcome = {"http_status": 200}
        except asyncio.CancelledError:
            status, event = "cancelled", ("cancelled", {})
            outcome = {"stop_reason": "cancelled", "error_code": "cancelled"}
        except TimeoutError:
            status, event = "failed", ("failed", {"code": "run_timeout"})
            outcome = {"stop_reason": "error", "error_code": "run_timeout", "error_text": "ran longer than max_run_seconds", "http_status": 504}
        except BackendError as e:
            status, event = "failed", ("failed", {"code": e.code})
            outcome = {"stop_reason": "error", "error_code": e.code, "error_text": e.text[:500], "http_status": e.status}
            if job["kind"] == "pull":
                await self.store.upsert_model(job["served_model"], status="missing", pull_error=e.text[:200])
        except Exception as e:                      # a defect in a backend is a failed run, never a dead lane
            status, event = "failed", ("failed", {"code": "internal"})
            outcome = {"stop_reason": "error", "error_code": "internal", "error_text": repr(e)[:500], "http_status": 500}
        finally:
            t_end = time.monotonic()
            run_ms = int((t_end - t_start) * 1000)
            accepted = parse_iso(job["accepted_at"]) or time.time()
            outcome.update({"status": status, "finished_at": now_iso(), "run_ms": run_ms, "total_ms": int((time.time() - accepted) * 1000)})
            if first is not None:
                outcome.update({"first_token_at": first_at, "time_to_first_token_ms": int((first - t_start) * 1000)})
            try:
                await self.store.update_run(jid, outcome, event)
            finally:
                lane.running.discard(jid)
                self.tasks.pop(jid, None)
                if status != "done":
                    self._publish(jid, {"type": "error", "code": outcome.get("error_code"), "text": outcome.get("error_text"), "status": status})
                self._finish_subscribers(jid)
                self.chunks.pop(jid, None)
                self._loaded_cache = (0.0, set())
                self.wake.set()

    async def _retention(self, job: dict) -> str:
        k = await self.store.key(job["key_id"]) if job["key_id"] else None
        return k["payload_retention"] if k else "metadata"

    # -- the caller's side -------------------------------------------------------------------------------------------------------
    def subscribe(self, job_id: str) -> asyncio.Queue:
        """A queue fed with the job's chunks: the ones stored so far first (replay), then live, then None at the end."""
        q: asyncio.Queue = asyncio.Queue()
        for c in self.chunks.get(job_id, []):
            q.put_nowait(c)
        if job_id in self.ended and self.ended[job_id].is_set():
            q.put_nowait(None)
        else:
            self.subscribers.setdefault(job_id, []).append(q)
        return q

    async def stream(self, job_id: str) -> AsyncIterator[dict]:
        """Every chunk of a job, replayed from the store when it is already done, then live until it ends."""
        job = await self.store.run(job_id)
        if job and job["status"] not in ("queued", "running"):
            p = await self.store.payload(job_id)
            for c in (p or {}).get("chunks") or []:
                yield c
            if job["status"] == "done" and p and p.get("response") is not None:
                yield {"type": "final", "stop_reason": job["stop_reason"], "message": p["response"], "stats": self._stats_of(job)}
            elif job["status"] != "done":
                yield {"type": "error", "code": job["error_code"], "text": job["error_text"], "status": job["status"]}
            return
        q = self.subscribe(job_id)
        while True:
            c = await q.get()
            if c is None:
                return
            yield c

    @staticmethod
    def _stats_of(job: dict) -> dict:
        return {k: job[k] for k in ("load_ms", "prompt_tokens", "prompt_eval_ms", "output_tokens", "eval_ms", "total_backend_ms")}

    def _publish(self, job_id: str, chunk: dict) -> None:
        for q in self.subscribers.get(job_id, []):
            q.put_nowait(chunk)

    def _finish_subscribers(self, job_id: str) -> None:
        for q in self.subscribers.pop(job_id, []):
            q.put_nowait(None)
        self._ended(job_id).set()

    def _ended(self, job_id: str) -> asyncio.Event:
        return self.ended.setdefault(job_id, asyncio.Event())

    async def wait(self, job_id: str, timeout: float | None = None) -> dict:
        """Blocks until the job leaves queued/running (or the timeout), answering the run row."""
        job = await self.store.run(job_id)
        if not job or job["status"] not in ("queued", "running"):
            return job
        ev = self._ended(job_id)
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        return await self.store.run(job_id)

    # -- operations -------------------------------------------------------------------------------------------------------------
    async def cancel(self, job_id: str) -> dict:
        job = await self.store.run(job_id)
        if not job:
            raise Refusal(404, "no_such_job", "no such job")
        if job["status"] == "running":
            t = self.tasks.get(job_id)
            if t:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        elif job["status"] == "queued":
            await self.store.update_run(job_id, {"status": "cancelled", "finished_at": now_iso(), "error_code": "cancelled"}, ("cancelled", {}))
            self._finish_subscribers(job_id)
        return await self.store.run(job_id)

    async def set_priority(self, job_id: str, priority: int) -> dict:
        job = await self.store.run(job_id)
        if not job or job["status"] != "queued":
            raise Refusal(409, "not_queued", "only a queued job is re-prioritised")
        if not 1 <= int(priority) <= 9:
            raise Refusal(400, "bad_priority", "priority must be 1–9")
        await self.store.update_run(job_id, {"priority": int(priority)}, ("priority", {"priority": int(priority)}))
        self.wake.set()
        return await self.store.run(job_id)

    async def pause(self, lane: str, paused: bool) -> dict:
        if lane not in self.lanes:
            raise Refusal(400, "bad_lane", "no such lane")
        self.lanes[lane].paused = paused
        await self._emit("lane_paused" if paused else "lane_resumed", {"lane": lane})
        self.wake.set()
        return self.lanes[lane].snapshot()

    async def refresh_models(self) -> None:
        """The catalogue from the backend (§5.2): on disk, loaded, lane by the `-cloud` suffix unless overridden; what vanished is `missing`."""
        try:
            listed = await self.backend.list_models()
            loaded = {m["name"]: m for m in await self.backend.loaded()}
        except Exception:
            return
        seen = set()
        for m in listed:
            seen.add(m["name"])
            fields = {"on_disk": 1, "lane": m.get("lane") or ("cloud" if m["name"].endswith("-cloud") else "local"),
                      "family": m.get("family"), "parameter_size": m.get("parameter_size"), "quantization": m.get("quantization"),
                      "context_length": m.get("context_length"), "capabilities": m.get("capabilities") or [], "size_bytes": m.get("size_bytes"),
                      "digest": m.get("digest"), "loaded": int(m["name"] in loaded),
                      "loaded_size_bytes": loaded.get(m["name"], {}).get("size_bytes"), "expires_at": loaded.get(m["name"], {}).get("expires_at")}
            existing = await self.store.fetchone("SELECT status, lane, on_disk, loaded, digest, expires_at FROM models WHERE name = ?", (m["name"],))
            if existing and existing["status"] != "disabled":
                fields["status"] = "available"
            changed = not existing or any(existing.get(k) != v for k, v in fields.items() if k in existing)
            if changed or not existing:
                await self.store.upsert_model(m["name"], **fields)
        for r in await self.store.models():
            if r["name"] not in seen and r["on_disk"]:
                await self.store.upsert_model(r["name"], on_disk=0, loaded=0, status="missing" if r["status"] != "disabled" else "disabled")
        self._loaded_cache = (0.0, set())

    async def queue_snapshot(self) -> list[dict]:
        out = []
        pos: dict[str, int] = {}
        waits: dict[str, int] = {}
        for lane in self.lanes:
            for i, r in enumerate(await self.store.queued(lane)):
                pos[r["id"]] = i + 1
            waits.update(await self.lane_waits(lane))
        rows = await self.store.fetchall("SELECT * FROM runs WHERE status IN ('queued', 'running') ORDER BY lane_served, status DESC, priority DESC, accepted_at")
        for r in rows:
            out.append({"job_id": r["id"], "lane": r["lane_served"], "requested_model": r["requested_model"], "served_model": r["served_model"],
                        "key_name": r["key_name"], "priority": r["priority"], "mode": r["mode"], "face": r["face"], "kind": r["kind"], "status": r["status"],
                        "position": pos.get(r["id"]), "queued_at": r["accepted_at"], "started_at": r["started_at"],
                        "expected_wait_ms": waits.get(r["id"], 0),
                        "tokens_so_far": len(self.chunks.get(r["id"], [])), "tags": json.loads(r["tags"]), "fallback_model": r["fallback_model"]})
        return out

    async def status(self) -> dict:
        lanes = []
        for lane in self.lanes.values():
            queued = await self.store.queued(lane.name)
            oldest = (time.time() - (parse_iso(queued[0]["accepted_at"]) or time.time())) if queued else 0
            drain = (await self.lane_waits(lane.name)).get(queued[-1]["id"], 0) if queued else 0
            lanes.append({**lane.snapshot(), "queued": len(queued), "oldest_queued_seconds": int(oldest), "expected_drain_ms": drain})
        return {"started_at": self.started_at, "lanes": lanes, "loaded_models": sorted(await self.loaded_models()),
                "backend": {"kind": self.backend.kind, "reachable": await self.backend.health(), "version": await self.backend.version()},
                "machine": await self._machine(), "counters": await self.store.counters(), "policy_version": await self.store.policy_version(),
                "seq": self.store.seq}

    async def _emit(self, kind: str, detail: dict) -> None:
        await self.store.add_event(None, kind, detail)
        for q in self.listeners:
            q.put_nowait({"kind": kind, "detail": detail, "at": now_iso()})
