"""The service on real sockets: the TCP listener needs a key, the operator's socket does not, a caller that hangs up
mid-stream stops the model, and the node refuses to listen beyond loopback with no key. The CLI reads the same socket."""
import asyncio
import json
import socket

import httpx
import pytest
import pytest_asyncio

from server import cli
from server.app import serve
from server.backends.fake import FakeBackend
from server.config import NodeConfig

MODELS = {"slow": {"lane": "local", "tokens": 400, "token_ms": 10, "load_ms": 0}, "quick": {"lane": "local", "tokens": 3, "token_ms": 1, "load_ms": 0},
          "embedder": {"lane": "local", "tokens": 1}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Running:
    def __init__(self, cfg, backend, stop, task):
        self.cfg, self.backend, self.stop, self.task = cfg, backend, stop, task
        self.url = f"http://127.0.0.1:{cfg.port}"

    def operator(self, **kw):
        return httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(self.cfg.socket_path)), base_url="http://sllm", timeout=30, **kw)


@pytest_asyncio.fixture
async def running(tmp_path):
    cfg = NodeConfig(port=free_port(), data_dir=tmp_path / "d", name="socket-node")
    backend = FakeBackend({k: dict(v) for k, v in MODELS.items()}, speed=1.0)
    ready, stop = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(serve(cfg, backend, ready, stop))
    await asyncio.wait_for(ready.wait(), 15)
    yield Running(cfg, backend, stop, task)
    stop.set()
    await asyncio.wait_for(task, 15)


async def test_two_listeners_one_wall(running):
    async with httpx.AsyncClient(base_url=running.url, timeout=10) as tcp, running.operator() as op:
        assert (await tcp.get("/node/v1/health")).status_code == 200
        assert (await tcp.get("/api/tags")).status_code == 401, "the network needs a key, loopback included"
        made = (await op.post("/node/v1/keys", json={"name": "lan"})).json()
        h = {"Authorization": f"Bearer {made['key']}"}
        assert [m["name"] for m in (await tcp.get("/api/tags", headers=h)).json()["models"]] == sorted(MODELS)
        r = await tcp.post("/api/chat", headers=h, json={"model": "quick", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        assert r.status_code == 200 and r.json()["message"]["content"] == "tok0 tok1 tok2 "
    assert oct(running.cfg.socket_path.stat().st_mode & 0o777) == "0o600" and oct(running.cfg.data_dir.stat().st_mode & 0o777) == "0o700"


async def test_hanging_up_mid_stream_stops_the_model(running):
    async with running.operator() as op:
        got = 0
        async with op.stream("POST", "/api/chat", json={"model": "slow", "messages": []}) as r:
            assert r.status_code == 200
            job_id = r.headers["x-node-job-id"]
            async for line in r.aiter_lines():
                if line.strip():
                    got += 1
                if got == 3:
                    break                      # the caller walks away: the context closes the connection
        for _ in range(100):
            st = (await op.get("/node/v1/status")).json()
            if st["lanes"]["local"]["running"] == 0 and st["counters"]["jobs"].get("cancelled") == 1:
                break
            await asyncio.sleep(0.05)
        assert st["counters"]["jobs"] == {"cancelled": 1}, "the job is recorded cancelled, not left running"
        assert running.backend.in_flight == 0, "the backend's stream was closed"
        # the slot is free again at once: the next call is served
        r = await op.post("/api/chat", json={"model": "quick", "messages": [], "stream": False})
        assert r.status_code == 200 and int(r.headers["x-node-queue-wait-ms"]) < 2000
        assert job_id != r.headers["x-node-job-id"]


async def test_hanging_up_while_queued_cancels(running):
    async with running.operator() as op:
        await op.post("/node/v1/keys", json={"name": "x"})
        first = asyncio.create_task(op.post("/api/chat", json={"model": "slow", "messages": [], "stream": False}))
        await asyncio.sleep(0.3)
        async with running.operator() as other:
            with pytest.raises(httpx.ReadTimeout):
                await other.post("/api/chat", json={"model": "quick", "messages": [], "stream": False}, timeout=0.5)     # waits behind `slow`, then gives up
        for _ in range(60):
            st = (await op.get("/node/v1/status")).json()
            if st["counters"]["jobs"].get("cancelled") == 1:
                break
            await asyncio.sleep(0.05)
        assert st["counters"]["jobs"].get("cancelled") == 1 and st["lanes"]["local"]["queued"] == 0, "a caller that gave up leaves nothing in the queue"
        first.cancel()


async def test_refuses_the_lan_without_a_key(tmp_path):
    cfg = NodeConfig(port=free_port(), bind="0.0.0.0", data_dir=tmp_path / "d")
    with pytest.raises(SystemExit) as e:
        await serve(cfg, FakeBackend(dict(MODELS)))
    assert "no key" in str(e.value)


async def test_the_cli_reads_the_socket(running, capsys):
    def run(*argv):
        return cli.main(list(argv))

    cfg = running.cfg
    orig = NodeConfig.load
    NodeConfig.load = classmethod(lambda cls, path=None: cfg)
    try:
        assert await asyncio.to_thread(run, "status") == 0
        out = capsys.readouterr().out
        assert "socket-node" in out and "answering" in out and "lane:      local" in out
        assert await asyncio.to_thread(run, "keys", "mint", "desk", "--models", "quick") == 0
        key = capsys.readouterr().out.strip().splitlines()[-1]      # the service logs to the same stdout in this test
        assert key.startswith("sllm_")
        assert await asyncio.to_thread(run, "keys", "list") == 0
        assert "desk" in capsys.readouterr().out
        assert await asyncio.to_thread(run, "models") == 0
        assert "quick" in capsys.readouterr().out
        assert await asyncio.to_thread(run, "prove", "--model", "quick") == 0
        out = capsys.readouterr().out
        assert "PROVEN: 0 failure(s)" in out and "an embedding is answered" in out, out
        with pytest.raises(SystemExit):
            await asyncio.to_thread(run, "serve")              # a second service on the same socket is refused
    finally:
        NodeConfig.load = orig
