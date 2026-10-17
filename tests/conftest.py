import asyncio
import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from server.backends.fake import FakeBackend
from server.keys import Keys
from server.scheduler import Scheduler
from server.store import Store

MODELS = {
    "local-a": {"lane": "local", "tokens": 5, "token_ms": 10, "load_ms": 100},
    "local-b": {"lane": "local", "tokens": 5, "token_ms": 10, "load_ms": 100},
    "big-cloud": {"lane": "cloud", "tokens": 5, "token_ms": 10, "load_ms": 0},
    "other-cloud": {"lane": "cloud", "tokens": 3, "token_ms": 10, "load_ms": 0},
}


class Node:
    def __init__(self, store, backend, sched, keys):
        self.store, self.backend, self.sched, self.keys = store, backend, sched, keys

    async def mint(self, name="k", **limits):
        raw, row = await self.keys.mint("inference", name, **limits)
        return raw, row

    async def submit(self, key, model="local-a", node=None, **kw):
        req = {"model": model, "messages": [{"role": "user", "content": "hi"}]}
        return await self.sched.submit(key=key, face="openai", request=req, node=node, **kw)


@pytest_asyncio.fixture
async def node(tmp_path):
    store = await Store(tmp_path / "data").open()
    backend = FakeBackend(dict((k, dict(v)) for k, v in MODELS.items()), speed=0.02)
    sched = Scheduler(store, backend)
    await sched.refresh_models()
    await sched.start()
    n = Node(store, backend, sched, Keys(store))
    yield n
    await sched.stop()
    await store.close()
