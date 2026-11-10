"""The Ollama face and /node/v1 over the app itself (no socket), against the fake backend: the keys, the wire shapes, the
node headers, the lanes seen from outside, the refusals."""
import asyncio
import json

import httpx
import pytest
import pytest_asyncio

from server.app import Node, make_app
from server.backends.fake import FakeBackend
from server.config import NodeConfig

MODELS = {
    "local-a": {"lane": "local", "tokens": 5, "token_ms": 10, "load_ms": 100},
    "local-b": {"lane": "local", "tokens": 5, "token_ms": 10, "load_ms": 100},
    "big-cloud": {"lane": "cloud", "tokens": 5, "token_ms": 10, "load_ms": 0},
    "embedder": {"lane": "local", "tokens": 1, "caps": ["embedding"]},
}


class Site:
    def __init__(self, node, backend):
        self.node, self.backend = node, backend
        self.tcp = httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(node)), base_url="http://node", timeout=30)
        self.op = httpx.AsyncClient(transport=httpx.ASGITransport(app=make_app(node, operator=True)), base_url="http://node", timeout=30)

    async def key(self, name="caller", kind="inference", **limits):
        raw, _ = await self.node.keys.mint(kind, name, **limits)
        return {"Authorization": f"Bearer {raw}"}


@pytest_asyncio.fixture
async def site(tmp_path):
    backend = FakeBackend({k: dict(v) for k, v in MODELS.items()}, speed=0.02)
    node = await Node(NodeConfig(data_dir=tmp_path / "data", name="test-node"), backend).start()
    s = Site(node, backend)
    yield s
    await s.tcp.aclose()
    await s.op.aclose()
    await node.stop()


def lines(r):
    return [json.loads(l) for l in r.text.splitlines() if l.strip()]


async def test_keys_are_the_wall(site):
    r = await site.tcp.get("/api/tags")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    r = await site.tcp.get("/api/tags", headers={"Authorization": "Bearer sllm_" + "0" * 48})
    assert r.status_code == 401
    inf, nod = await site.key(), await site.key("mgr", "node")
    assert (await site.tcp.get("/api/tags", headers=inf)).status_code == 200
    r = await site.tcp.get("/api/tags", headers=nod)
    assert r.status_code == 403 and r.json()["error"]["code"] == "wrong_key_kind", "a node key does not open the faces"
    r = await site.tcp.get("/node/v1/status", headers=inf)
    assert r.status_code == 403, "an inference key does not open the node API"
    assert (await site.tcp.get("/node/v1/status", headers=nod)).status_code == 200
    assert (await site.tcp.get("/node/v1/health")).status_code == 200, "health needs no key"
    assert (await site.op.get("/api/tags")).status_code == 200 and (await site.op.get("/node/v1/status")).status_code == 200, "the operator's socket needs none"


async def test_tags_ps_version_show(site):
    h = await site.key()
    tags = (await site.tcp.get("/api/tags", headers=h)).json()["models"]
    assert [m["name"] for m in tags] == sorted(MODELS) and tags[0]["model"] == tags[0]["name"]
    assert set(tags[0]) == {"name", "model", "modified_at", "size", "digest", "details"} and tags[0]["details"]["family"] == "fake"
    assert (await site.tcp.get("/api/version", headers=h)).json()["version"] == "fake-1"
    await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [], "stream": False})
    assert [m["name"] for m in (await site.tcp.get("/api/ps", headers=h)).json()["models"]] == ["local-a"]
    assert (await site.tcp.post("/api/show", headers=h, json={"model": "local-a"})).json()["context_length"] == 4096
    assert (await site.tcp.post("/api/show", headers=h, json={"model": "nope"})).status_code == 404
    limited = await site.key("limited", allowed_models=["local-b"])
    assert [m["name"] for m in (await site.tcp.get("/api/tags", headers=limited)).json()["models"]] == ["local-b"], "a key sees the models it may use"


async def test_chat_whole_answer(site):
    h = await site.key()
    r = await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [{"role": "user", "content": "hello there"}], "stream": False})
    assert r.status_code == 200
    d = r.json()
    assert d["model"] == "local-a" and d["done"] is True and d["done_reason"] == "stop"
    assert d["message"] == {"role": "assistant", "content": "tok0 tok1 tok2 tok3 tok4 "}
    assert d["eval_count"] == 5 and d["prompt_eval_count"] == 3 and d["eval_duration"] == 50_000_000 and d["load_duration"] == 100_000_000
    job = await site.node.store.run(r.headers["x-node-job-id"])
    assert job["status"] == "done" and job["face"] == "ollama" and job["kind"] == "chat" and job["key_name"] == "caller" and job["stream"] == 0
    assert (job["prompt_tokens"], job["output_tokens"]) == (3, 5), "the run record keeps what the face answered"
    assert r.headers["x-node-id"] == site.node.store.node_id and r.headers["x-maluos-request"] == job["id"]
    assert r.headers["x-node-lane"] == "local" and r.headers["x-node-served-model"] == "local-a" and int(r.headers["x-node-queue-wait-ms"]) >= 0


async def test_chat_stream_is_ollamas(site):
    h = await site.key()
    r = await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    out = lines(r)
    assert len(out) == 6 and all(o["done"] is False for o in out[:-1]) and out[-1]["done"] is True
    assert "".join(o["message"]["content"] for o in out) == "tok0 tok1 tok2 tok3 tok4 "
    assert out[-1]["message"] == {"role": "assistant", "content": ""} and out[-1]["eval_count"] == 5 and out[-1]["done_reason"] == "stop"
    assert (await site.node.store.run(r.headers["x-node-job-id"]))["stream"] == 1


async def test_generate_and_embed(site):
    h = await site.key()
    r = await site.tcp.post("/api/generate", headers=h, json={"model": "local-a", "prompt": "the sky is", "stream": False})
    assert r.status_code == 200 and r.json()["response"] == "tok0 tok1 tok2 tok3 tok4 " and r.json()["done"] is True and "message" not in r.json()
    assert (await site.node.store.run(r.headers["x-node-job-id"]))["kind"] == "generate"
    out = lines(await site.tcp.post("/api/generate", headers=h, json={"model": "local-a", "prompt": "the sky is"}))
    assert "".join(o["response"] for o in out) == "tok0 tok1 tok2 tok3 tok4 " and out[-1]["done"] is True
    r = await site.tcp.post("/api/embed", headers=h, json={"model": "embedder", "input": ["one", "two"]})
    assert r.status_code == 200 and r.json()["embeddings"] == [[0.1, 0.2, 0.3]] * 2 and r.json()["model"] == "embedder" and r.json()["prompt_eval_count"] == 2
    job = await site.node.store.run(r.headers["x-node-job-id"])
    assert job["kind"] == "embed" and job["status"] == "done"


async def test_the_request_runs_whatever_the_retention(site):
    """The request is kept in memory for the job, not only under `full` retention: the backend gets the caller's messages."""
    for retention in ("none", "metadata", "full"):
        h = await site.key(f"k-{retention}", payload_retention=retention)
        r = await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [{"role": "user", "content": "x" * 40}], "stream": False, "options": {"temperature": 0}})
        assert r.status_code == 200 and r.json()["prompt_eval_count"] == 11, retention
        sent = site.backend.calls[-1]
        assert sent["messages"][0]["content"] == "x" * 40 and sent["options"] == {"temperature": 0} and sent["keep_alive"] == "5m"
    await asyncio.sleep(0.05)                 # the answer leaves a moment before the job's own end
    assert site.node.sched.requests == {}, "nothing is left in memory once the jobs end"


async def test_keep_alive_travels(site):
    h = await site.key()
    await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [], "stream": False, "keep_alive": -1})
    assert site.backend.calls[-1]["keep_alive"] == "-1"


async def test_cloud_is_served_like_local_and_beside_it(site):
    """P3 and P4: no key flag gates a cloud model, and cloud calls run while the one local slot is busy."""
    h = await site.key("plain", max_concurrent=20)                     # cloud_allowed is false on this key
    site.backend.models["local-a"].update(tokens=20)
    site.backend.models["big-cloud"].update(tokens=50, token_ms=50)
    calls = [site.tcp.post("/api/chat", headers=h, json={"model": m, "messages": [], "stream": False})
             for m in ("local-a", "local-b", "big-cloud", "big-cloud", "big-cloud")]
    answers = await asyncio.gather(*calls)
    assert [r.status_code for r in answers] == [200] * 5
    assert site.backend.max_by_lane["local"] == 1, "one local inference at a time"
    assert site.backend.max_by_lane["cloud"] == 3, "the cloud calls overlap each other and the local one"
    assert {r.headers["x-node-lane"] for r in answers[2:]} == {"cloud"}
    waited = [int(r.headers["x-node-queue-wait-ms"]) for r in answers[:2]]
    assert max(waited) > 0, "the second local call waited for the slot, and says so"


async def test_refusals_and_failures(site):
    h = await site.key()
    r = await site.tcp.post("/api/chat", headers=h, json={"model": "nope", "messages": []})
    assert r.status_code == 404 and r.json()["error"] == {"code": "model_unknown", "message": "model nope is not on this node", "type": "not_found"}
    assert (await site.tcp.post("/api/chat", headers=h, content=b"{not json")).status_code == 400
    assert (await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [], "node": {"bogus": 1}})).json()["error"]["code"] == "bad_node_field"
    limited = await site.key("limited", allowed_models=["local-b"])
    assert (await site.tcp.post("/api/chat", headers=limited, json={"model": "local-a", "messages": []})).status_code == 403
    one = await site.key("one", max_concurrent=1)
    await site.node.sched.pause("local", True)
    first = asyncio.create_task(site.tcp.post("/api/chat", headers=one, json={"model": "local-a", "messages": [], "stream": False}))
    await asyncio.sleep(0.1)
    r = await site.tcp.post("/api/chat", headers=one, json={"model": "local-a", "messages": []})
    assert r.status_code == 429 and r.headers["retry-after"] == "5"
    await site.node.sched.pause("local", False)
    assert (await first).status_code == 200
    # the backend fails the job: an HTTP error before the stream began, the run recorded failed
    site.backend.fail_next = "out_of_memory"
    r = await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": []})
    assert r.status_code == 500 and r.json()["error"]["code"] == "out_of_memory" and r.headers["x-node-job-id"]
    assert (await site.node.store.run(r.headers["x-node-job-id"]))["status"] == "failed"
    # the backend is down: refused before it queues
    site.backend.healthy = False
    r = await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": []})
    assert r.status_code == 503 and r.json()["error"]["code"] == "backend_down" and r.headers["retry-after"] == "15"
    assert (await site.tcp.get("/node/v1/health")).status_code == 503
    site.backend.healthy = True


async def test_a_cancelled_job_answers_409(site):
    h = await site.key()
    await site.node.sched.pause("local", True)
    call = asyncio.create_task(site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [], "stream": False}))
    await asyncio.sleep(0.1)
    (queued,) = await site.node.store.queued()
    await site.node.sched.cancel(queued["id"])
    r = await call
    assert r.status_code == 409 and r.json()["error"]["code"] == "cancelled"
    await site.node.sched.pause("local", False)


async def test_status_and_envelope(site):
    h = await site.key("mgr", "node")
    await site.op.post("/api/chat", json={"model": "local-a", "messages": [], "stream": False})
    st = (await site.tcp.get("/node/v1/status", headers=h)).json()
    assert st["node_id"] == site.node.store.node_id and st["node_name"] == "test-node" and st["api_version"] == 1 and st["clock"].endswith("Z")
    assert st["backends"][0]["healthy"] is True and st["backends"][0]["kind"] == "fake" and st["loaded"] == ["local-a"]
    assert st["lanes"]["local"]["capacity"] == 1 and st["lanes"]["cloud"]["capacity"] is None and st["lanes"]["local"]["running"] == 0
    assert st["models"] == {"local": 3, "cloud": 1} and st["counters"]["jobs"] == {"done": 1}
    assert st["machine"]["hostname"] and st["machine"]["cpu_count"] and st["machine"]["gpus"] == []
    health = (await site.tcp.get("/node/v1/health")).json()
    assert health["ok"] is True and health["node_id"] == st["node_id"]


async def test_keys_over_the_node_api(site):
    r = await site.op.post("/node/v1/keys", json={"name": "desk", "max_priority": 7, "allowed_models": ["local-a"]})
    assert r.status_code == 201
    made = r.json()
    assert made["key"].startswith("sllm_") and len(made["key"]) == 53 and made["kind"] == "inference" and made["max_priority"] == 7
    h = {"Authorization": f"Bearer {made['key']}"}
    assert (await site.tcp.post("/api/chat", headers=h, json={"model": "local-a", "messages": [], "stream": False})).status_code == 200
    listed = (await site.op.get("/node/v1/keys")).json()["keys"]
    assert [k["name"] for k in listed] == ["desk"] and "key" not in listed[0] and "hash" not in listed[0], "a key is never read back"
    assert (await site.op.post("/node/v1/keys", json={"name": "x", "bogus": 1})).status_code == 400
    assert (await site.op.post("/node/v1/keys", json={})).status_code == 400
    assert (await site.op.delete(f"/node/v1/keys/{made['id']}")).json()["status"] == "revoked"
    assert (await site.tcp.get("/api/tags", headers=h)).status_code == 401, "a revoked key stops at once"
    assert (await site.op.delete("/node/v1/keys/nope")).status_code == 404
