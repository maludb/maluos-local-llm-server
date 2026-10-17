"""The fake backend the proofs run against (design §9): answers recorded shapes with timings scaled down, counts what it was asked,
and can be made unhealthy, slow or missing a model. Every number it reports is deterministic so the run record can be checked."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .base import Backend, BackendError, ModelMissing


class FakeBackend(Backend):
    kind = "fake"

    def __init__(self, models: dict[str, dict] | None = None, speed: float = 0.01):
        # models: name -> {"lane": local|cloud, "tokens": n, "token_ms": ms per token, "load_ms": ms}
        self.models = models or {}
        self.speed = speed                 # multiplies every sleep (1.0 = the model's timings as declared)
        self.healthy = True
        self._loaded: dict[str, float] = {}
        self.calls: list[dict] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.in_flight_by_lane: dict[str, int] = {}
        self.max_by_lane: dict[str, int] = {}
        self.started: list[tuple[str, float]] = []
        self.fail_next: str | None = None

    async def health(self) -> bool:
        return self.healthy

    async def version(self) -> str:
        return "fake-1"

    async def list_models(self) -> list[dict]:
        return [{"name": n, "lane": m.get("lane", "local"), "size_bytes": m.get("size", 1000), "digest": "sha256:" + n,
                 "family": "fake", "parameter_size": "1B", "quantization": "Q4", "context_length": m.get("ctx", 4096),
                 "capabilities": ["chat"] + m.get("caps", [])} for n, m in self.models.items()]

    async def show(self, model: str) -> dict:
        if model not in self.models:
            raise ModelMissing(model)
        return (await self.list_models())[list(self.models).index(model)]

    async def loaded(self) -> list[dict]:
        return [{"name": n, "size_bytes": 1000, "expires_at": None} for n in self._loaded]

    async def chat(self, request: dict) -> AsyncIterator[dict]:
        model = request["model"]
        if not self.healthy:
            raise BackendError("backend_down", "the backend is not answering", 503)
        if model not in self.models:
            raise ModelMissing(model)
        if self.fail_next:
            code, self.fail_next = self.fail_next, None
            raise BackendError(code, "fake failure", 500)
        spec = self.models[model]
        self.calls.append(request)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        lane = spec.get("lane", "local")
        self.in_flight_by_lane[lane] = self.in_flight_by_lane.get(lane, 0) + 1
        self.max_by_lane[lane] = max(self.max_by_lane.get(lane, 0), self.in_flight_by_lane[lane])
        self.started.append((model, time.monotonic()))
        try:
            t0 = time.monotonic()
            load_ms = 0
            if model not in self._loaded:
                load_ms = spec.get("load_ms", 500)
                await asyncio.sleep(load_ms / 1000 * self.speed)
                if spec.get("lane", "local") == "local":
                    self._loaded = {model: time.monotonic()}      # one local model's worth of memory
            n = spec.get("tokens", 10)
            token_ms = spec.get("token_ms", 20)
            prompt_tokens = sum(len(str(m.get("content", ""))) // 4 + 1 for m in request.get("messages", []))
            prompt_ms = prompt_tokens * 2
            await asyncio.sleep(prompt_ms / 1000 * self.speed)
            text = []
            for i in range(n):
                await asyncio.sleep(token_ms / 1000 * self.speed)
                piece = f"tok{i} "
                text.append(piece)
                yield {"type": "delta", "text": piece}
            total_ms = int((time.monotonic() - t0) * 1000 / max(self.speed, 1e-9)) if self.speed else load_ms + prompt_ms + n * token_ms
            yield {"type": "final", "stop_reason": "stop",
                   "stats": {"load_ms": load_ms, "prompt_tokens": prompt_tokens, "prompt_eval_ms": prompt_ms, "output_tokens": n,
                             "eval_ms": n * token_ms, "total_ms": load_ms + prompt_ms + n * token_ms},
                   "message": {"role": "assistant", "content": "".join(text)}}
        finally:
            self.in_flight -= 1
            self.in_flight_by_lane[lane] -= 1

    async def embed(self, request: dict) -> dict:
        model = request["model"]
        if model not in self.models:
            raise ModelMissing(model)
        inputs = request["input"] if isinstance(request["input"], list) else [request["input"]]
        return {"embeddings": [[0.1, 0.2, 0.3] for _ in inputs], "stats": {"prompt_tokens": sum(len(i) // 4 + 1 for i in inputs), "total_ms": 5}}

    async def pull(self, model: str) -> AsyncIterator[dict]:
        if model.startswith("missing"):
            raise BackendError("pull_failed", f"{model} not in the library", 404)
        for done in (0, 50, 100):
            await asyncio.sleep(0.01 * self.speed)
            yield {"status": "pulling", "completed": done, "total": 100}
        self.models[model] = {"lane": "cloud" if model.endswith("-cloud") else "local", "tokens": 3}
        yield {"status": "success"}

    async def delete(self, model: str) -> None:
        self.models.pop(model, None)
        self._loaded.pop(model, None)

    async def load(self, model: str, keep_alive=None) -> dict:
        if model not in self.models:
            raise ModelMissing(model)
        await asyncio.sleep(self.models[model].get("load_ms", 500) / 1000 * self.speed)
        self._loaded = {model: time.monotonic()}
        return {"load_ms": self.models[model].get("load_ms", 500)}

    async def unload(self, model: str) -> None:
        self._loaded.pop(model, None)

    async def machine(self) -> dict:
        return {"hostname": "fake", "memory_total_bytes": 16 << 30, "memory_free_bytes": 8 << 30}
