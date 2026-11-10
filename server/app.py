"""The node service: one process, two listeners.

  TCP (config `bind`:`port`, 11435)   the Ollama face and /node/v1 for callers with a key. Every route but
                                      /node/v1/health needs `Authorization: Bearer sllm_...`: an inference key opens the
                                      face, a node key opens /node/v1, and neither opens the other.
  the operator's socket               the same routes with no key, for the `sllm` CLI on the machine. The socket sits in
                                      the data directory (0700), so only the service's user and root reach it.

The Ollama face (`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/version`, `/api/show`) is what a
MaluOS gateway's remote backend dials (maluos docs/inference-node-api.md §5-6). Every inference is a job through the
scheduler: it waits for its lane, then streams. Errors are `{"error": {"code", "message", "type"}}`; an error after the
stream began is a line `{"error": "...", "error_class": "..."}` as Ollama sends one.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import __version__, machine
from .backends.base import Backend, BackendError
from .config import NodeConfig
from .keys import Keys, present
from .scheduler import Refusal, Scheduler
from .store import Store, now_iso

API_VERSION = 1
REFRESH_MODELS_S = 60
PURGE_S = 3600
LOOPBACK = ("127.0.0.1", "::1", "localhost")

# The CLI's caller on the operator's socket: no key row, the widest limits.
OPERATOR = {"id": "operator", "name": "operator", "kind": "operator", "max_priority": 9, "max_concurrent": 1000, "cloud_allowed": 1,
            "fallback_allowed": 0, "payload_retention": "metadata", "allowed_models": None, "daily_token_cap": None}


def log(event: str, **fields: Any) -> None:
    """One JSON line on stdout; never a payload, never a key."""
    sys.stdout.write(json.dumps({"at": now_iso(), "event": event, **fields}, default=str) + "\n")
    sys.stdout.flush()


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def error_response(status: int, code: str, message: str, retry_after: int | None = None, headers: dict | None = None) -> JSONResponse:
    kind = {400: "invalid_request", 401: "authentication", 403: "permission", 404: "not_found", 409: "conflict", 429: "rate_limit"}.get(status, "unavailable" if status >= 500 else "error")
    h = dict(headers or {})
    if retry_after:
        h["Retry-After"] = str(retry_after)
    return JSONResponse({"error": {"code": code, "message": message, "type": kind}}, status_code=status, headers=h)


class Node:
    """The service's parts and their lifetime: the store, the backend, the scheduler, the keys."""

    def __init__(self, cfg: NodeConfig, backend: Backend | None = None):
        self.cfg = cfg
        self.store = Store(cfg.data_dir)
        self.backend = backend
        self.sched: Scheduler | None = None
        self.keys: Keys | None = None
        self.name = cfg.name
        self._tasks: list[asyncio.Task] = []
        self._cancels: set[asyncio.Task] = set()

    async def start(self) -> "Node":
        await self.store.open()
        if self.backend is None:
            from .backends.ollama import OllamaBackend
            self.backend = OllamaBackend(self.cfg.backend_url)
        self.sched = Scheduler(self.store, self.backend)
        self.keys = Keys(self.store)
        saved = await self.store.setting("name")
        self.name = saved if saved and saved != "node" else self.cfg.name
        await self.sched.refresh_models()
        await self.sched.start()
        self._tasks = [asyncio.create_task(self._every(REFRESH_MODELS_S, self.sched.refresh_models)), asyncio.create_task(self._every(PURGE_S, self._purge))]
        return self

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.sched:
            await self.sched.stop()
        close = getattr(self.backend, "close", None)
        if close:
            await close()
        await self.store.close()

    async def _every(self, seconds: float, fn) -> None:
        while True:
            await asyncio.sleep(seconds)
            try:
                await fn()
            except Exception as e:          # a failed refresh is tried again; it never ends the service
                log("background_error", task=getattr(fn, "__name__", "?"), error=repr(e)[:200])

    async def _purge(self) -> None:
        await self.store.purge_results(int(self.sched.setting("result_retention_hours")))
        await self.store.purge_payloads(int(self.sched.setting("payload_retention_days")))

    def envelope(self) -> dict:
        return {"node_id": self.store.node_id, "node_name": self.name, "api_version": API_VERSION, "clock": now_iso()}

    def cancel_later(self, job_id: str) -> None:
        """Cancel from a place that may itself be cancelled (a caller hung up mid-stream): the cancel runs as its own task."""
        t = asyncio.get_running_loop().create_task(self.sched.cancel(job_id))
        self._cancels.add(t)
        t.add_done_callback(self._cancels.discard)


# -- the Ollama wire ---------------------------------------------------------------------------------------------------------
def _ns(ms: Any) -> int:
    return int(ms or 0) * 1_000_000


def _final_fields(chunk: dict) -> dict:
    """Ollama's closing numbers: the daemon's own object when the chunk carries it, else rebuilt from the run's stats."""
    raw = chunk.get("raw")
    if raw:
        return {k: v for k, v in raw.items() if k not in ("model", "created_at", "done")}
    st = chunk.get("stats") or {}
    return {"done_reason": chunk.get("stop_reason") or "stop", "total_duration": _ns(st.get("total_ms") or st.get("total_backend_ms")), "load_duration": _ns(st.get("load_ms")),
            "prompt_eval_count": int(st.get("prompt_tokens") or 0), "prompt_eval_duration": _ns(st.get("prompt_eval_ms")),
            "eval_count": int(st.get("output_tokens") or 0), "eval_duration": _ns(st.get("eval_ms"))}


def ollama_line(kind: str, model: str, chunk: dict, whole: bool = False) -> dict:
    """One object of Ollama's stream for a scheduler chunk; `whole` answers the complete message (a non-streaming call)."""
    out: dict = {"model": model, "created_at": now_iso()}
    if chunk["type"] == "delta":
        if kind == "generate":
            out["response"] = chunk.get("text") or ""
            if chunk.get("thinking"):
                out["thinking"] = chunk["thinking"]
        else:
            msg: dict = {"role": "assistant", "content": chunk.get("text") or ""}
            for k in ("thinking", "tool_calls"):
                if chunk.get(k):
                    msg[k] = chunk[k]
            out["message"] = msg
        out["done"] = False
        return out
    message = chunk.get("message") or {}
    if kind == "generate":
        out["response"] = (message.get("content") or "") if whole else ""
        if whole and message.get("thinking"):
            out["thinking"] = message["thinking"]
    else:
        out["message"] = message if whole else {"role": "assistant", "content": ""}
    out["done"] = True
    out.update(_final_fields(chunk))
    return out


def model_entry(m: dict) -> dict:
    """A catalogue row as Ollama's /api/tags lists a model."""
    return {"name": m["name"], "model": m["name"], "modified_at": m["updated_at"], "size": m["size_bytes"] or 0, "digest": m["digest"] or "",
            "details": {"family": m["family"] or "", "parameter_size": m["parameter_size"] or "", "quantization_level": m["quantization"] or ""}}


def make_app(node: Node, operator: bool = False) -> FastAPI:
    """The routes. `operator` is the socket's app: its callers need no key."""
    app = FastAPI(title="maluos-local-llm-server", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, e: ApiError):
        return error_response(e.status, e.code, e.message, e.retry_after)

    @app.exception_handler(Refusal)
    async def _refusal(request: Request, e: Refusal):
        return error_response(e.status, e.code, e.message, e.retry_after)

    @app.exception_handler(BackendError)
    async def _backend_error(request: Request, e: BackendError):
        return error_response(e.status, e.code, e.text, 15 if e.status == 503 else None)

    async def caller(request: Request, kind: str) -> dict:
        if operator:
            return OPERATOR
        auth = request.headers.get("authorization") or ""
        raw = auth[7:].strip() if auth[:7].lower() == "bearer " else None
        if not raw:
            raise ApiError(401, "unauthorized", "this node needs a key: Authorization: Bearer sllm_...")
        key = await node.keys.authenticate(raw, kind)
        if key:
            return key
        other = "node" if kind == "inference" else "inference"
        if await node.keys.authenticate(raw, other):
            raise ApiError(403, "wrong_key_kind", f"{'an inference' if other == 'inference' else 'a node'} key does not open {'the node API' if kind == 'node' else 'the faces'}")
        raise ApiError(401, "unauthorized", "the key is unknown, paused or revoked")

    async def body_of(request: Request) -> dict:
        try:
            body = json.loads(await request.body() or b"{}")
        except ValueError:
            raise ApiError(400, "bad_json", "the body is not JSON")
        if not isinstance(body, dict):
            raise ApiError(400, "bad_json", "the body must be a JSON object")
        return body

    def job_headers(job: dict) -> dict:
        return {"X-Node-Id": node.store.node_id, "X-Node-Job-Id": job["id"], "X-MaluOS-Request": job["id"], "X-Node-Lane": job["lane_served"] or "",
                "X-Node-Served-Model": job["served_model"] or "", "X-Node-Queue-Wait-Ms": str(job["queue_wait_ms"] or 0)}

    def failure(job: dict, chunk: dict | None, headers: dict) -> JSONResponse:
        """A job that ended without an answer, as an HTTP error."""
        status = job["status"] if job else "failed"
        if status == "cancelled":
            return error_response(409, "cancelled", "the job was cancelled", headers=headers)
        if status == "expired":
            return error_response(504, "wait_exceeds", job.get("error_text") or "the job waited too long in the queue", headers=headers)
        code = (chunk or {}).get("code") or job.get("error_code") or "backend_error"
        text = (chunk or {}).get("text") or job.get("error_text") or "the job failed"
        http = int(job.get("http_status") or 502)
        code = {"model_missing": "model_not_found", "backend_down": "local_unavailable"}.get(code, code)
        if http == 401:                # Ollama's own refusal (a cloud model, the host not signed in): never this node's "bad key"
            http = 502
        return error_response(http if http >= 400 else 502, code, text, 15 if http == 503 else None, headers)

    async def next_or_gone(it: AsyncIterator[dict], request: Request) -> dict | None:
        """The stream's next chunk (None at its end); raises ConnectionError when the caller hung up while waiting."""
        task = asyncio.ensure_future(it.__anext__())
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=1.0)
                if done:
                    try:
                        return task.result()
                    except StopAsyncIteration:
                        return None
                if await request.is_disconnected():
                    raise ConnectionError("the caller hung up")
        finally:
            if not task.done():
                task.cancel()

    async def infer(request: Request, kind: str) -> Response:
        key = await caller(request, "inference")
        body = await body_of(request)
        opts = body.pop("node", None) or {}
        if not isinstance(opts, dict):
            raise ApiError(400, "bad_node_field", "node must be an object")
        if body.get("keep_alive") is not None and "keep_alive" not in opts:
            opts["keep_alive"] = body["keep_alive"]
        stream = kind != "embed" and body.get("stream", True) is not False
        job = await node.sched.submit(key=key, face="ollama", request=body, node=opts, kind=kind, stream=stream,
                                      client_ip=request.client.host if request.client else None, user_agent=request.headers.get("user-agent"))
        jid = job["id"]
        chunks = node.sched.stream(jid)
        try:
            first = await next_or_gone(chunks, request)
        except ConnectionError:
            await node.sched.cancel(jid)
            return Response(status_code=499)
        job = await node.store.run(jid)
        headers = job_headers(job)
        if first is None or first["type"] == "error":
            return failure(job, first, headers)
        model = job["served_model"]

        if stream:
            async def lines() -> AsyncIterator[bytes]:
                ended = False
                try:
                    chunk: dict | None = first
                    while chunk is not None:
                        if chunk["type"] == "error":
                            yield (json.dumps({"error": chunk.get("text") or "the job failed", "error_class": chunk.get("code") or "backend_error"}) + "\n").encode()
                            ended = True
                            return
                        if chunk["type"] in ("delta", "final"):
                            yield (json.dumps(ollama_line(kind, model, chunk)) + "\n").encode()
                        if chunk["type"] == "final":
                            ended = True
                            return
                        chunk = await anext(chunks, None)
                    ended = True
                finally:
                    if not ended:              # the caller hung up: stop the model, free the lane
                        node.cancel_later(jid)
            return StreamingResponse(lines(), media_type="application/x-ndjson", headers=headers)

        final, chunk = None, first
        try:
            while chunk is not None:
                if chunk["type"] == "error":
                    return failure(await node.store.run(jid), chunk, headers)
                if chunk["type"] == "final":
                    final = chunk
                    break
                chunk = await next_or_gone(chunks, request)
        except ConnectionError:
            await node.sched.cancel(jid)
            return Response(status_code=499)
        if final is None:
            return failure(await node.store.run(jid), None, headers)
        if kind == "embed":
            raw = final.get("raw") or {"total_duration": _ns((final.get("stats") or {}).get("total_ms")), "prompt_eval_count": int((final.get("stats") or {}).get("prompt_tokens") or 0)}
            return JSONResponse({"model": model, "embeddings": final.get("embeddings") or [], **{k: v for k, v in raw.items() if k != "model"}}, headers=headers)
        return JSONResponse(ollama_line(kind, model, final, whole=True), headers=headers)

    # -- the Ollama face -------------------------------------------------------------------------------------------------------
    @app.post("/api/chat")
    async def api_chat(request: Request):
        return await infer(request, "chat")

    @app.post("/api/generate")
    async def api_generate(request: Request):
        return await infer(request, "generate")

    @app.post("/api/embed")
    async def api_embed(request: Request):
        return await infer(request, "embed")

    @app.get("/api/tags")
    async def api_tags(request: Request):
        key = await caller(request, "inference")
        allowed = key.get("allowed_models")
        rows = [m for m in await node.store.models() if m["on_disk"] and m["allowed"] and m["status"] == "available" and (not allowed or m["name"] in allowed)]
        return {"models": [model_entry(m) for m in rows]}

    @app.get("/api/ps")
    async def api_ps(request: Request):
        await caller(request, "inference")
        loaded = await node.backend.loaded()
        return {"models": [m.get("raw") or {"name": m["name"], "model": m["name"], "size": m.get("size_bytes"), "expires_at": m.get("expires_at")} for m in loaded]}

    @app.get("/api/version")
    async def api_version(request: Request):
        await caller(request, "inference")
        return {"version": await node.backend.version(), "node_version": __version__}

    @app.post("/api/show")
    async def api_show(request: Request):
        await caller(request, "inference")
        body = await body_of(request)
        model = await node.sched.resolve_model(str(body.get("model") or body.get("name") or ""))
        shown = await node.backend.show(model["name"])
        return shown.get("raw") or shown

    # -- the node API: health, status, keys --------------------------------------------------------------------------------------
    @app.get("/node/v1/health")
    async def health():
        ok = await node.backend.health()
        return JSONResponse({"ok": ok, **node.envelope()}, status_code=200 if ok else 503)

    @app.get("/node/v1/status")
    async def status(request: Request):
        await caller(request, "node")
        st = await node.sched.status()
        lanes = {}
        for ln in st["lanes"]:
            lanes[ln["name"]] = {"kind": ln["name"], "capacity": ln["capacity"], "running": ln["running"], "queued": ln["queued"], "paused": ln["paused"],
                                 "oldest_queued_ms": ln["oldest_queued_seconds"] * 1000, "expected_drain_ms": ln["expected_drain_ms"]}
        backend = {"name": "ollama-local", "kind": st["backend"]["kind"], "endpoint": getattr(node.backend, "url", None), "healthy": st["backend"]["reachable"],
                   "version": st["backend"]["version"], "loaded": st["loaded_models"]}
        signed = getattr(node.backend, "signed_in", None)
        if signed:
            backend["cloud_signed_in"] = await signed()
        models = await node.store.models()
        return {**node.envelope(), "name": node.name, "version": __version__, "started_at": st["started_at"], "backends": [backend],
                "machine": machine.facts(), "lanes": lanes, "loaded": st["loaded_models"], "counters": st["counters"],
                "models": {"local": sum(1 for m in models if m["on_disk"] and m["lane"] == "local"), "cloud": sum(1 for m in models if m["on_disk"] and m["lane"] == "cloud")},
                "listen": {"bind": node.cfg.bind, "port": node.cfg.port}, "seq": st["seq"]}

    @app.get("/node/v1/keys")
    async def keys_list(request: Request):
        await caller(request, "node")
        return {**node.envelope(), "keys": [present(k) for k in await node.store.keys()]}

    @app.post("/node/v1/keys")
    async def keys_mint(request: Request):
        await caller(request, "node")
        body = await body_of(request)
        name = str(body.pop("name", "") or "").strip()
        kind = body.pop("kind", "inference")
        if not name:
            raise ApiError(400, "invalid", "a key needs a name")
        try:
            raw, row = await node.keys.mint(kind, name, **body)
        except (ValueError, TypeError) as e:
            raise ApiError(400, "invalid", str(e))
        log("key_minted", name=name, kind=kind, id=row["id"])
        return JSONResponse({**node.envelope(), "key": raw, **present(row)}, status_code=201)

    @app.delete("/node/v1/keys/{key_id}")
    async def keys_revoke(key_id: str, request: Request):
        await caller(request, "node")
        if not await node.store.key(key_id):
            raise ApiError(404, "no_such_key", "no such key")
        await node.keys.revoke(key_id)
        log("key_revoked", id=key_id)
        return {**node.envelope(), **present(await node.store.key(key_id))}

    @app.get("/")
    async def root():
        return {"service": "maluos-local-llm-server", "version": __version__, **node.envelope()}

    return app


# -- serving -------------------------------------------------------------------------------------------------------------------
class _Server(uvicorn.Server):
    """Two of these share one loop; the signals are `serve`'s, not each server's."""

    @contextlib.contextmanager
    def capture_signals(self):
        yield


async def serve(cfg: NodeConfig, backend: Backend | None = None, ready: asyncio.Event | None = None, stop: asyncio.Event | None = None) -> None:
    node = await Node(cfg, backend).start()
    try:
        cfg.data_dir.chmod(0o700)
        if cfg.bind not in LOOPBACK and not [k for k in await node.store.keys() if k["status"] == "active"]:
            raise SystemExit(f"refusing to listen on {cfg.bind} with no key: mint one first (sllm keys mint NAME), or bind 127.0.0.1")
        sock = cfg.socket_path
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()
        common = {"log_level": "warning", "access_log": False, "lifespan": "off", "timeout_graceful_shutdown": 5}
        tcp = _Server(uvicorn.Config(make_app(node), host=cfg.bind, port=cfg.port, **common))
        uds = _Server(uvicorn.Config(make_app(node, operator=True), uds=str(sock), **common))
        stop = stop or asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                loop.add_signal_handler(sig, stop.set)
        servers = [asyncio.create_task(tcp.serve()), asyncio.create_task(uds.serve())]
        t0 = time.monotonic()
        while not (tcp.started and uds.started):
            if any(t.done() for t in servers) or time.monotonic() - t0 > 15:
                for t in servers:
                    t.cancel()
                raise SystemExit(f"could not listen on {cfg.bind}:{cfg.port} or {sock}")
            await asyncio.sleep(0.02)
        with contextlib.suppress(OSError):
            os.chmod(sock, 0o600)
        models = await node.store.models()
        log("started", node_id=node.store.node_id, name=node.name, version=__version__, listen=f"{cfg.bind}:{cfg.port}", socket=str(sock),
            backend=cfg.backend_url, backend_reachable=await node.backend.health(), models=sum(1 for m in models if m["on_disk"]))
        if ready:
            ready.set()
        await stop.wait()
        tcp.should_exit = uds.should_exit = True
        await asyncio.gather(*servers, return_exceptions=True)
        with contextlib.suppress(FileNotFoundError):
            sock.unlink()
        log("stopped")
    finally:
        await node.stop()
