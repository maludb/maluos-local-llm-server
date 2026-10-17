"""One class per backend (design §5.1). A chat is an async iterator of deltas ending with a `final` carrying the backend's stats:
    {"type": "delta", "text": "...", "tool_calls": [...]?}            zero or more
    {"type": "final", "stop_reason": "stop|length|tool_calls", "stats": {load_ms, prompt_tokens, prompt_eval_ms, output_tokens,
                                                                         eval_ms, total_ms}, "message": {...the complete message...}}
"""
from __future__ import annotations

from typing import Any, AsyncIterator


class BackendError(Exception):
    def __init__(self, code: str, text: str, status: int = 502):
        super().__init__(text)
        self.code, self.text, self.status = code, text, status


class ModelMissing(BackendError):
    def __init__(self, model: str):
        super().__init__("model_missing", f"model {model} is not on this node", 404)


class Backend:
    kind = "base"

    async def health(self) -> bool: raise NotImplementedError
    async def version(self) -> str: return ""
    async def list_models(self) -> list[dict]: raise NotImplementedError
    async def show(self, model: str) -> dict: raise NotImplementedError
    async def loaded(self) -> list[dict]: raise NotImplementedError
    def chat(self, request: dict) -> AsyncIterator[dict]: raise NotImplementedError
    async def embed(self, request: dict) -> dict: raise NotImplementedError
    def pull(self, model: str) -> AsyncIterator[dict]: raise NotImplementedError
    async def delete(self, model: str) -> None: raise NotImplementedError
    async def load(self, model: str, keep_alive: str | int | None) -> dict: raise NotImplementedError
    async def unload(self, model: str) -> None: raise NotImplementedError
    async def machine(self) -> dict[str, Any]: return {}
