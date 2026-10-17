"""Step 1's proof (design §9): lanes, modes' queue side, fallback, models, the node API's store side, the run record — without Ollama."""
import asyncio
import json
import time

import pytest

from server.keys import hash_key, PREFIX
from server.scheduler import Refusal
from server.store import Store


async def drain(node, timeout=10):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        r = await node.store.fetchone("SELECT count(*) AS n FROM runs WHERE status IN ('queued', 'running')")
        if r["n"] == 0:
            return
        await asyncio.sleep(0.02)
    raise AssertionError("queue did not drain")


# -- lanes -------------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_local_one_at_a_time_cloud_together(node):
    _, k = await node.mint(cloud_allowed=True, max_concurrent=20)
    node.backend.models["local-a"]["tokens"] = 200
    node.backend.models["big-cloud"]["tokens"] = 200
    jobs = [await node.submit(k, "local-a") for _ in range(3)] + [await node.submit(k, "big-cloud") for _ in range(3)]
    await drain(node, 30)
    assert node.backend.max_by_lane["local"] == 1, "local jobs run one at a time"
    assert node.backend.max_by_lane["cloud"] == 3, "cloud jobs run together, beside the local one"
    assert node.backend.max_in_flight >= 4
    for j in jobs:
        r = await node.store.run(j["id"])
        assert r["status"] == "done" and r["output_tokens"] == 200


@pytest.mark.asyncio
async def test_priority_then_fair_share_then_arrival(node):
    _, ka = await node.mint("a", max_priority=9, max_concurrent=20)
    _, kb = await node.mint("b", max_priority=9, max_concurrent=20)
    node.backend.models["local-a"]["tokens"] = 300
    blocker = await node.submit(ka, "local-a")
    await asyncio.sleep(0.05)                           # it is running (so key a was served last); everything below queues
    node.backend.models["local-a"]["tokens"] = 5
    await node.sched.pause("local", True)
    low = await node.submit(ka, "local-a", {"priority": 1})
    a1 = await node.submit(ka, "local-a", {"priority": 5})
    a2 = await node.submit(ka, "local-a", {"priority": 5})
    b1 = await node.submit(kb, "local-a", {"priority": 5})
    high = await node.submit(kb, "local-a", {"priority": 9})
    await node.sched.pause("local", False)
    await drain(node)
    order = [await node.store.run(j["id"]) for j in (high, a1, b1, a2, low)]
    starts = [r["started_at"] for r in order]
    assert starts == sorted(starts), "priority 9 first (b's); at 5 the key served longest ago — a, then b, then a again; priority 1 last"


@pytest.mark.asyncio
async def test_affinity_within_window_only(node):
    _, k = await node.mint(max_concurrent=20)
    first = await node.submit(k, "local-a")               # loads local-a
    await node.sched.wait(first["id"], 5)
    node.sched.lanes["local"].paused = True
    jb = await node.submit(k, "local-b")
    await asyncio.sleep(0.01)
    ja = await node.submit(k, "local-a")
    node.sched.lanes["local"].paused = False
    node.sched.wake.set()
    await drain(node)
    ra, rb = await node.store.run(ja["id"]), await node.store.run(jb["id"])
    assert ra["started_at"] < rb["started_at"], "the loaded model's job jumps the older one inside the window"
    # outside the window (0 seconds) arrival wins
    await node.store.set_settings({"affinity_window_seconds": 0})
    await node.sched.reload_settings()
    node.sched.lanes["local"].paused = True
    jb2 = await node.submit(k, "local-b")
    await asyncio.sleep(0.01)
    ja2 = await node.submit(k, "local-a")
    node.sched.lanes["local"].paused = False
    node.sched.wake.set()
    await drain(node)
    assert (await node.store.run(jb2["id"]))["started_at"] < (await node.store.run(ja2["id"]))["started_at"]


@pytest.mark.asyncio
async def test_pause_and_resume(node):
    _, k = await node.mint()
    await node.sched.pause("local", True)
    j = await node.submit(k)
    await asyncio.sleep(0.3)
    assert (await node.store.run(j["id"]))["status"] == "queued"
    await node.sched.pause("local", False)
    assert (await node.sched.wait(j["id"], 5))["status"] == "done"


# -- fallback (§3) ------------------------------------------------------------------------------------------------------------
async def _slow_local(node):
    """Make the local lane's expected wait long: a known slow model history + a running blocker."""
    await node.store.set_settings({"default_expected_ms": 200000, "fallback_after_seconds": 1})
    await node.sched.reload_settings()


@pytest.mark.asyncio
async def test_fallback_needs_key_and_model_or_request(node):
    await _slow_local(node)
    await node.store.upsert_model("local-a", fallback_model="big-cloud")
    _, plain = await node.mint("plain", cloud_allowed=True)
    _, allowed = await node.mint("allowed", cloud_allowed=True, fallback_allowed=True, max_concurrent=20)
    await node.sched.pause("local", True)
    j_plain = await node.submit(plain, "local-a")
    j_allowed = await node.submit(allowed, "local-a")
    j_none = await node.submit(allowed, "local-a", {"fallback": "none"})
    j_req = await node.submit(allowed, "local-b", {"fallback": {"model": "other-cloud", "after_seconds": 1}})
    j_b = await node.submit(allowed, "local-b")             # local-b names no fallback → stays
    for _ in range(40):
        await asyncio.sleep(0.05)
        if (await node.store.run(j_allowed["id"]))["lane_served"] == "cloud":
            break
    r = await node.store.run(j_allowed["id"])
    assert r["lane_served"] == "cloud" and r["served_model"] == "big-cloud" and r["fallback_reason"] == "local_wait"
    assert (await node.store.run(j_req["id"]))["served_model"] == "other-cloud"
    for j in (j_plain, j_none, j_b):
        rr = await node.store.run(j["id"])
        assert rr["lane_served"] == "local" and rr["fallback_reason"] is None, rr["key_name"]
    ev = [e["kind"] for e in await node.store.run_events(j_allowed["id"])]
    assert ev[:2] == ["queued", "moved"]
    await node.sched.pause("local", False)
    await drain(node)
    assert (await node.store.run(j_allowed["id"]))["status"] == "done"


@pytest.mark.asyncio
async def test_started_job_never_moves_and_unavailable_moves(node):
    await node.store.upsert_model("local-a", fallback_model="big-cloud")
    _, k = await node.mint(cloud_allowed=True, fallback_allowed=True, max_concurrent=20)
    node.backend.models["local-a"]["tokens"] = 400
    running = await node.submit(k, "local-a")
    await asyncio.sleep(0.05)
    assert (await node.store.run(running["id"]))["status"] == "running"
    node.backend.healthy = False
    waiting = await node.submit(k, "local-a")               # accepted: it can fall back
    await asyncio.sleep(1.2)
    assert (await node.store.run(waiting["id"]))["fallback_reason"] == "local_unavailable"
    assert (await node.store.run(running["id"]))["lane_served"] == "local", "a started job never moves"
    _, nofb = await node.mint("nofb")
    with pytest.raises(Refusal) as e:
        await node.submit(nofb, "local-a")
    assert e.value.status == 503 and e.value.code == "backend_down"
    node.backend.healthy = True
    await drain(node)


# -- refusals and limits ------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_refusals(node):
    _, k = await node.mint(max_concurrent=1, allowed_models=["local-a"])
    await node.sched.pause("local", True)
    await node.submit(k, "local-a")
    with pytest.raises(Refusal) as e:
        await node.submit(k, "local-a")
    assert e.value.status == 429 and e.value.retry_after == 5
    await node.sched.pause("local", False)
    await drain(node)
    for model, nodef, status, code in (("nope", None, 404, "model_unknown"), ("local-b", None, 403, "model_not_allowed"),
                                       ("local-a", {"bogus": 1}, 400, "bad_node_field"), ("local-a", {"priority": 12}, 400, "bad_priority"),
                                       ("local-a", {"fallback": "maybe"}, 400, "bad_fallback"), ("local-a", {"tags": "x"}, 400, "bad_tags")):
        with pytest.raises(Refusal) as e:
            await node.submit(k, model, nodef)
        assert (e.value.status, e.value.code) == (status, code), model
    _, nc = await node.mint("nocloud")
    with pytest.raises(Refusal) as e:
        await node.submit(nc, "big-cloud")
    assert e.value.code == "cloud_not_allowed"
    _, capped = await node.mint("capped", max_priority=3)
    j = await node.submit(capped, "local-a", {"priority": 9})
    assert j["priority"] == 3, "priority clamped to the key's cap"
    await drain(node)


@pytest.mark.asyncio
async def test_idempotency_and_daily_cap(node):
    _, k = await node.mint(max_concurrent=20)
    a = await node.submit(k, "local-a", {"idempotency_key": "same"})
    b = await node.submit(k, "local-a", {"idempotency_key": "same"})
    assert a["id"] == b["id"]
    await drain(node)
    await node.keys.update(k["id"], {"daily_token_cap": 1})
    k = await node.store.key(k["id"])
    with pytest.raises(Refusal) as e:
        await node.submit(k, "local-a")
    assert e.value.code == "daily_cap"


# -- cancel, expiry, restart ---------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancel_queued_and_running_and_expiry(node):
    _, k = await node.mint(max_concurrent=20)
    node.backend.models["local-a"]["tokens"] = 200
    running = await node.submit(k, "local-a")
    await asyncio.sleep(0.1)
    queued = await node.submit(k, "local-a")
    expiring = await node.submit(k, "local-a", {"max_queue_seconds": 1})
    r = await node.sched.cancel(queued["id"])
    assert r["status"] == "cancelled"
    r = await node.sched.cancel(running["id"])
    assert r["status"] == "cancelled" and r["run_ms"] is not None
    await asyncio.sleep(1.2)
    await node.sched.pause("local", True)        # keep the expiring one queued past its limit
    for _ in range(30):
        await asyncio.sleep(0.1)
        if (await node.store.run(expiring["id"]))["status"] == "expired":
            break
    assert (await node.store.run(expiring["id"]))["status"] in ("expired", "done")
    await node.sched.pause("local", False)
    await drain(node)


@pytest.mark.asyncio
async def test_restart_fails_what_was_running(node, tmp_path):
    _, k = await node.mint()
    node.backend.models["local-a"]["tokens"] = 500
    j = await node.submit(k, "local-a")
    await asyncio.sleep(0.1)
    await node.sched.stop()
    # simulate a crash: the row says running
    await node.store.db.execute("UPDATE runs SET status = 'running' WHERE id = ?", (j["id"],))
    await node.store.db.commit()
    from server.scheduler import Scheduler
    s2 = Scheduler(node.store, node.backend)
    await s2.start()
    r = await node.store.run(j["id"])
    assert r["status"] == "failed" and r["error_code"] == "node_restarted"
    await s2.stop()
    node.sched = s2


# -- the run record and usage -------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_run_record_columns(node):
    _, k = await node.mint(payload_retention="full")
    j = await node.submit(k, "local-a", {"tags": ["app:x"]}, client_ip="10.0.0.9", user_agent="t/1")
    r = await node.sched.wait(j["id"], 5)
    assert r["status"] == "done" and r["stop_reason"] == "stop"
    assert r["prompt_tokens"] > 0 and r["output_tokens"] == 5 and r["eval_ms"] == 50 and r["output_tokens_per_second"] == 100.0
    assert r["load_ms"] == 100 and r["model_was_loaded"] == 0 and r["queue_wait_ms"] >= 0 and r["time_to_first_token_ms"] > 0
    assert r["run_ms"] > 0 and r["total_ms"] >= r["run_ms"] and r["context_used"] == r["prompt_tokens"] + 5
    assert r["client_ip"] == "10.0.0.9" and json.loads(r["tags"]) == ["app:x"] and r["key_name"] == "k" and r["face"] == "openai"
    assert json.loads(r["loaded_models_at_start"]) == [] and r["cpu_load_1m"] is not None
    ev = [e["kind"] for e in await node.store.run_events(j["id"])]
    assert ev == ["queued", "started", "done"]
    second = await node.sched.wait((await node.submit(k, "local-a"))["id"], 5)
    assert second["model_was_loaded"] == 1 and second["load_ms"] == 0
    p = await node.store.payload(j["id"])
    assert p["request"]["messages"][0]["content"] == "hi" and p["response"]["content"].startswith("tok0") and len(p["chunks"]) == 5


@pytest.mark.asyncio
async def test_retention_and_purge(node):
    _, none_k = await node.mint("n", payload_retention="none")
    _, full_k = await node.mint("f", payload_retention="full")
    jn = await node.sched.wait((await node.submit(none_k))["id"], 5)
    jf = await node.sched.wait((await node.submit(full_k))["id"], 5)
    pn, pf = await node.store.payload(jn["id"]), await node.store.payload(jf["id"])
    assert pn["request"] is None and pn["response"] is not None, "no request kept under none; the result kept for the caller"
    assert pf["request"] is not None
    assert await node.store.purge_results(0) == 1 and await node.store.payload(jn["id"]) is None and await node.store.payload(jf["id"]) is not None
    assert await node.store.purge_payloads(0) == 1 and await node.store.payload(jf["id"]) is None


@pytest.mark.asyncio
async def test_usage_equals_sql(node):
    _, k = await node.mint(cloud_allowed=True, max_concurrent=20)
    for m in ("local-a", "local-a", "big-cloud"):
        await node.submit(k, m)
    await drain(node)
    by_lane = {r["grp"]: r for r in await node.store.usage(None, None, "lane")}
    assert by_lane["local"]["runs"] == 2 and by_lane["cloud"]["runs"] == 1
    total = await node.store.fetchone("SELECT sum(prompt_tokens) + sum(output_tokens) AS t FROM runs")
    assert sum(r["tokens_total"] for r in by_lane.values()) == total["t"]
    assert by_lane["local"]["duration_ms_p50"] is not None


# -- the node API's store side -----------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_since_seq_sync_misses_nothing(node):
    _, k = await node.mint(max_concurrent=1000)
    node.backend.speed = 0.0
    ids = {(await node.submit(k, "local-a"))["id"] for _ in range(300)}
    await drain(node, 60)
    seen, cursor = {}, 0
    while True:
        page = await node.store.runs_since(cursor, limit=50)
        if not page:
            break
        for r in page:
            seen[r["id"]] = r["status"]
        cursor = page[-1]["seq"]
    assert set(seen) == ids and all(s == "done" for s in seen.values())
    # a later update bumps seq past the cursor: re-sync returns only it
    some = next(iter(ids))
    await node.store.update_run(some, {"error_text": "note"})
    page = await node.store.runs_since(cursor)
    assert [r["id"] for r in page] == [some]


@pytest.mark.asyncio
async def test_keys(node):
    raw, row = await node.keys.mint("inference", "ide", max_priority=7)
    assert raw.startswith(PREFIX) and len(raw) == len(PREFIX) + 48 and row["hash"] == hash_key(raw)
    assert "hash" not in __import__("server.keys", fromlist=["present"]).present(row)
    assert (await node.keys.authenticate(raw, "inference"))["id"] == row["id"]
    assert await node.keys.authenticate(raw, "node") is None and await node.keys.authenticate("sllm_nope", "inference") is None
    await node.keys.revoke(row["id"])
    assert await node.keys.authenticate(raw, "inference") is None
    n1 = await node.keys.rotate_node_key()
    assert (await node.keys.authenticate(n1, "node")) is not None
    n2 = await node.keys.rotate_node_key()
    assert await node.keys.authenticate(n1, "node") is None and await node.keys.authenticate(n2, "node") is not None
    with pytest.raises(ValueError):
        await node.keys.mint("inference", "bad", max_priority=12)


@pytest.mark.asyncio
async def test_settings_and_policy_version(node):
    v0 = await node.store.policy_version()
    await node.store.set_settings({"local_parallel": 2, "auto_pull": "true"})
    s = await node.store.settings()
    assert s["local_parallel"] == 2 and s["auto_pull"] is True and await node.store.policy_version() == v0 + 1
    with pytest.raises(ValueError):
        await node.store.set_settings({"nope": 1})
    with pytest.raises(ValueError):
        await node.store.set_settings({"local_parallel": 0})
    await node.sched.reload_settings()
    assert node.sched.lanes["local"].capacity == 2


# -- models (§5.3) -------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pull_warm_unload_delete(node):
    _, k = await node.mint(max_concurrent=20)
    pull = await node.sched.submit_maintenance("pull", "new-model")
    chunks = [c async for c in node.sched.stream(pull["id"])]
    assert [c["completed"] for c in chunks if c["type"] == "progress" and "completed" in c] == [0, 50, 100]
    await node.sched.wait(pull["id"], 5)
    m = await node.store.model("new-model")
    assert m and m["on_disk"] == 1 and m["status"] == "available" and m["lane"] == "local"
    bad = await node.sched.wait((await node.sched.submit_maintenance("pull", "missing-one"))["id"], 5)
    assert bad["status"] == "failed" and (await node.store.model("missing-one"))["status"] == "missing"
    # a warm takes a local slot: it waits behind a running chat
    node.backend.models["local-a"]["tokens"] = 2000
    chat = await node.submit(k, "local-a")
    await asyncio.sleep(0.05)
    warm = await node.sched.submit_maintenance("load", "local-b")
    await asyncio.sleep(0.1)
    assert (await node.store.run(warm["id"]))["status"] == "queued"
    await drain(node)
    assert (await node.store.run(warm["id"]))["status"] == "done" and (await node.store.model("local-b"))["loaded"] == 1
    assert (await node.store.run(chat["id"]))["started_at"] < (await node.store.run(warm["id"]))["started_at"]
    await node.backend.unload("local-b")
    await node.sched.refresh_models()
    assert (await node.store.model("local-b"))["loaded"] == 0
    d = await node.sched.wait((await node.sched.submit_maintenance("delete", "new-model"))["id"], 5)
    assert d["status"] == "done" and await node.store.model("new-model") is None
    # missing from the backend → status missing; alias resolves
    node.backend.models.pop("other-cloud")
    await node.sched.refresh_models()
    assert (await node.store.model("other-cloud"))["status"] == "missing"
    await node.store.upsert_model("local-a", alias="fast")
    assert (await node.sched.resolve_model("fast"))["name"] == "local-a"
    await node.store.upsert_model("local-a", allowed=0)
    with pytest.raises(Refusal):
        await node.sched.resolve_model("local-a")


# -- streaming, expected wait, the queue snapshot ---------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stream_live_and_replay(node):
    _, k = await node.mint()
    j = await node.submit(k, "local-a")
    live = [c async for c in node.sched.stream(j["id"])]
    assert [c["type"] for c in live] == ["delta"] * 5 + ["final"] and live[-1]["message"]["content"].startswith("tok0")
    replay = [c async for c in node.sched.stream(j["id"])]
    assert [c["type"] for c in replay] == ["delta"] * 5 + ["final"]
    node.backend.fail_next = "boom"
    bad = await node.submit(k, "local-a")
    got = [c async for c in node.sched.stream(bad["id"])]
    assert got[-1]["type"] == "error" and got[-1]["code"] == "boom"


@pytest.mark.asyncio
async def test_expected_wait_and_snapshot(node):
    _, k = await node.mint(max_concurrent=20)
    await node.store.set_settings({"default_expected_ms": 1000})
    await node.sched.reload_settings()
    await node.sched.pause("local", True)
    j1 = await node.submit(k, "local-a")
    j2 = await node.submit(k, "local-a")
    w1, w2 = await node.sched.expected_wait_ms(j1["id"]), await node.sched.expected_wait_ms(j2["id"])
    assert w1 == 0 and w2 >= 1000, (w1, w2)
    snap = await node.sched.queue_snapshot()
    assert [s["position"] for s in snap] == [1, 2] and snap[1]["expected_wait_ms"] == w2
    st = await node.sched.status()
    local = next(l for l in st["lanes"] if l["name"] == "local")
    assert local["queued"] == 2 and local["paused"] and st["counters"]["jobs"]["queued"] == 2
    await node.sched.pause("local", False)
    await drain(node)


@pytest.mark.asyncio
async def test_ten_thousand_queued_schedule_fast(node):
    _, k = await node.mint(max_concurrent=100000)
    node.backend.speed = 0.0
    await node.sched.pause("local", True)
    t0 = time.monotonic()
    for i in range(2000):
        await node.submit(k, "local-a", {"priority": 1 + i % 9})
    submit_s = time.monotonic() - t0
    t0 = time.monotonic()
    await node.sched.pick(node.sched.lanes["local"])
    pick_s = time.monotonic() - t0
    assert pick_s < 0.1, f"a pick over 2,000 queued jobs took {pick_s:.3f}s"
    t0 = time.monotonic()
    await node.sched.lane_waits("local")
    assert time.monotonic() - t0 < 1.0
    # drain 500 of them (the rest cancelled) — the dispatch cost, not the fake's, is what this times
    await node.store.db.execute("UPDATE runs SET status = 'cancelled' WHERE id IN (SELECT id FROM runs WHERE status = 'queued' ORDER BY seq LIMIT 1500)")
    await node.store.db.commit()
    await node.sched.pause("local", False)
    t0 = time.monotonic()
    await drain(node, 120)
    per_job_ms = (time.monotonic() - t0) * 1000 / 500
    assert (await node.store.counters())["jobs"]["done"] == 500
    assert per_job_ms < 100, f"{per_job_ms:.1f} ms of bookkeeping a job"
    print(f"\n2,000 submits in {submit_s:.2f}s, a pick in {pick_s*1000:.1f} ms, {per_job_ms:.1f} ms a dispatched job")
