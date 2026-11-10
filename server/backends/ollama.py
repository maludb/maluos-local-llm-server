"""The Ollama backend (design §5.1): the native API over OLLAMA_HOST. The node talks to Ollama natively because only the
native endpoints carry the timings (load, prompt eval, eval) the run record keeps.

A cloud model (one the daemon forwards to ollama.com under the signed-in account) is listed beside the local ones and
served the same way (docs/estate-node-plan.md P3); it is marked lane `cloud` only so it does not wait for the GPU slot (P4).
A host that is not signed in lists no cloud models.

Every stream is the chunk protocol of base.py; the `final` chunk also carries `raw`, Ollama's own final object, so the
Ollama face answers the daemon's numbers untouched.
"""
from __future__ import annotations

import json
import time
from typing import Any, AsyncIterator

import httpx

from .. import machine
from .base import Backend, BackendError, ModelMissing

CAPS = {"completion": "chat", "embedding": "embedding", "vision": "vision", "tools": "tools", "thinking": "thinking"}
SIGNIN_CHECK_S = 300


def is_cloud(m: dict) -> bool:
    """Ollama marks a cloud model with the remote host it forwards to; the `cloud` tag suffix is the older sign."""
    name = m.get("name") or m.get("model") or ""
    return bool(m.get("remote_host") or m.get("remote_model")) or name.rpartition(":")[2].endswith("cloud")


def keep_alive_value(v: Any) -> Any:
    """Ollama takes a number of seconds or a duration string with a unit; the node stores `-1` and `0` as text."""
    if isinstance(v, str):
        try:
            return int(float(v))
        except ValueError:
            return v
    return v


def _ms(ns: Any) -> int:
    try:
        return int(ns) // 1_000_000
    except (TypeError, ValueError):
        return 0


def stats_of(final: dict) -> dict:
    return {"load_ms": _ms(final.get("load_duration")), "prompt_tokens": int(final.get("prompt_eval_count") or 0),
            "prompt_eval_ms": _ms(final.get("prompt_eval_duration")), "output_tokens": int(final.get("eval_count") or 0),
            "eval_ms": _ms(final.get("eval_duration")), "total_ms": _ms(final.get("total_duration"))}


def _error_text(body: bytes | str) -> str:
    text = body.decode(errors="replace") if isinstance(body, bytes) else body
    try:
        err = json.loads(text).get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)
        if err:
            return str(err)
    except (ValueError, AttributeError):
        pass
    return text.strip()[:400] or "the backend answered an error"


class OllamaBackend(Backend):
    kind = "ollama"

    def __init__(self, url: str = "http://127.0.0.1:11434", client: httpx.AsyncClient | None = None):
        self.url = url.rstrip("/")
        # no read timeout on a stream: a model's load can be silent for minutes, and max_run_seconds is the scheduler's
        self.client = client or httpx.AsyncClient(base_url=self.url, timeout=httpx.Timeout(connect=5.0, read=None, write=30.0, pool=10.0),
                                                  limits=httpx.Limits(max_connections=64, max_keepalive_connections=8))
        self._shown: dict[str, dict] = {}                 # digest -> the normalised /api/show
        self._signed_in: tuple[float, bool | None] = (0.0, None)
        self._version = ""

    async def close(self) -> None:
        await self.client.aclose()

    # -- plain calls -------------------------------------------------------------------------------------------------------------
    async def _json(self, method: str, path: str, body: dict | None = None, timeout: float = 15.0, model: str | None = None) -> dict:
        try:
            r = await self.client.request(method, path, json=body, timeout=timeout)
        except httpx.TimeoutException:
            raise BackendError("backend_timeout", f"Ollama did not answer {path} in {timeout:.0f} s", 504)
        except httpx.HTTPError as e:
            raise BackendError("backend_down", f"Ollama is not reachable at {self.url}: {e.__class__.__name__}", 503)
        if r.status_code == 404 and model:
            raise ModelMissing(model)
        if r.status_code >= 400:
            raise BackendError("backend_error", _error_text(r.content), r.status_code if r.status_code in (400, 401, 403, 429) else 502)
        try:
            return r.json() if r.content else {}
        except ValueError:
            raise BackendError("backend_error", f"Ollama answered {path} with something that is not JSON", 502)

    async def health(self) -> bool:
        try:
            r = await self.client.get("/api/version", timeout=3.0)
            if r.status_code == 200:
                self._version = str(r.json().get("version") or "")
                return True
        except (httpx.HTTPError, ValueError):
            pass
        return False

    async def version(self) -> str:
        if not self._version:
            await self.health()
        return self._version

    async def signed_in(self) -> bool | None:
        """True / False when the daemon says whether an ollama.com account is signed in; None when it cannot be told (an
        older daemon), and cloud models are then listed as the daemon lists them."""
        t, known = self._signed_in
        if time.monotonic() - t < SIGNIN_CHECK_S and t:
            return known
        try:
            r = await self.client.post("/api/me", timeout=5.0)
            known = True if r.status_code == 200 else False if r.status_code == 401 else None
        except httpx.HTTPError:
            known = None
        self._signed_in = (time.monotonic(), known)
        return known

    async def list_models(self) -> list[dict]:
        tags = (await self._json("GET", "/api/tags")).get("models") or []
        signed = await self.signed_in() if any(is_cloud(m) for m in tags) else None
        out = []
        for m in tags:
            name = m.get("name") or m.get("model")
            if not name:
                continue
            cloud = is_cloud(m)
            if cloud and signed is False:
                continue
            d = m.get("details") or {}
            row = {"name": name, "lane": "cloud" if cloud else "local", "size_bytes": m.get("size"), "digest": m.get("digest"),
                   "family": d.get("family"), "parameter_size": d.get("parameter_size"), "quantization": d.get("quantization_level"),
                   "context_length": None, "capabilities": ["chat"], "raw": m}
            key = m.get("digest") or name
            if key not in self._shown:
                try:
                    self._shown[key] = await self.show(name)
                except BackendError:
                    self._shown[key] = {}
            shown = self._shown[key]
            row["context_length"] = shown.get("context_length")
            row["capabilities"] = shown.get("capabilities") or row["capabilities"]
            out.append(row)
        return out

    async def show(self, model: str) -> dict:
        raw = await self._json("POST", "/api/show", {"model": model}, model=model)
        d = raw.get("details") or {}
        ctx = None
        for k, v in (raw.get("model_info") or {}).items():
            if k.endswith(".context_length") and isinstance(v, int):
                ctx = v
                break
        caps = [CAPS[c] for c in raw.get("capabilities") or [] if c in CAPS] or ["chat"]
        return {"name": model, "family": d.get("family"), "parameter_size": d.get("parameter_size"), "quantization": d.get("quantization_level"),
                "context_length": ctx, "capabilities": caps, "raw": raw}

    async def loaded(self) -> list[dict]:
        ps = (await self._json("GET", "/api/ps")).get("models") or []
        return [{"name": m.get("name") or m.get("model"), "size_bytes": m.get("size"), "expires_at": m.get("expires_at"), "raw": m}
                for m in ps if m.get("name") or m.get("model")]

    # -- inference ---------------------------------------------------------------------------------------------------------------
    async def _stream(self, path: str, body: dict) -> AsyncIterator[dict]:
        """Ollama's newline-delimited JSON, one object a line. A task cancelled here closes the connection, which is how
        Ollama learns to stop generating."""
        model = body.get("model") or ""
        try:
            async with self.client.stream("POST", path, json=body) as r:
                if r.status_code >= 400:
                    text = _error_text(await r.aread())
                    if r.status_code == 404:
                        raise ModelMissing(model)
                    code = {400: "bad_request", 401: "not_signed_in", 403: "not_allowed", 429: "rate_limited"}.get(r.status_code, "backend_error")
                    raise BackendError(code, text, r.status_code if r.status_code in (400, 401, 403, 429) else 502)
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        raise BackendError("backend_error", "Ollama sent a line that is not JSON", 502)
                    if isinstance(obj, dict) and obj.get("error"):
                        raise BackendError("backend_error", _error_text(json.dumps(obj)), 502)
                    yield obj
        except httpx.ConnectError:
            raise BackendError("backend_down", f"Ollama is not reachable at {self.url}", 503)
        except httpx.HTTPError as e:
            raise BackendError("backend_error", f"the connection to Ollama broke: {e.__class__.__name__}", 502)

    @staticmethod
    def _body(request: dict) -> dict:
        body = {k: v for k, v in request.items() if k != "node"}
        body["stream"] = True
        if "keep_alive" in body:
            body["keep_alive"] = keep_alive_value(body["keep_alive"])
        return body

    async def chat(self, request: dict) -> AsyncIterator[dict]:
        text: list[str] = []
        thinking: list[str] = []
        calls: list = []
        async for obj in self._stream("/api/chat", self._body(request)):
            msg = obj.get("message") or {}
            if msg.get("content"):
                text.append(msg["content"])
            if msg.get("thinking"):
                thinking.append(msg["thinking"])
            if msg.get("tool_calls"):
                calls.extend(msg["tool_calls"])
            if obj.get("done"):
                message: dict = {"role": "assistant", "content": "".join(text)}
                if thinking:
                    message["thinking"] = "".join(thinking)
                if calls:
                    message["tool_calls"] = calls
                reason = obj.get("done_reason") or "stop"
                yield {"type": "final", "stop_reason": "tool_calls" if calls else reason, "stats": stats_of(obj), "message": message,
                       "raw": {k: v for k, v in obj.items() if k != "message"}}
                return
            if msg.get("content") or msg.get("thinking") or msg.get("tool_calls"):
                delta: dict = {"type": "delta", "text": msg.get("content") or ""}
                if msg.get("thinking"):
                    delta["thinking"] = msg["thinking"]
                if msg.get("tool_calls"):
                    delta["tool_calls"] = msg["tool_calls"]
                yield delta

    async def generate(self, request: dict) -> AsyncIterator[dict]:
        text: list[str] = []
        thinking: list[str] = []
        async for obj in self._stream("/api/generate", self._body(request)):
            if obj.get("response"):
                text.append(obj["response"])
            if obj.get("thinking"):
                thinking.append(obj["thinking"])
            if obj.get("done"):
                message: dict = {"role": "assistant", "content": "".join(text)}
                if thinking:
                    message["thinking"] = "".join(thinking)
                # `context` (the token ids of the whole exchange) is deprecated and large: not kept, not relayed
                yield {"type": "final", "stop_reason": obj.get("done_reason") or "stop", "stats": stats_of(obj), "message": message,
                       "raw": {k: v for k, v in obj.items() if k not in ("response", "thinking", "context")}}
                return
            if obj.get("response") or obj.get("thinking"):
                delta = {"type": "delta", "text": obj.get("response") or ""}
                if obj.get("thinking"):
                    delta["thinking"] = obj["thinking"]
                yield delta

    async def embed(self, request: dict) -> dict:
        body = {k: v for k, v in request.items() if k not in ("node", "stream")}
        if "keep_alive" in body:
            body["keep_alive"] = keep_alive_value(body["keep_alive"])
        raw = await self._json("POST", "/api/embed", body, timeout=600.0, model=body.get("model"))
        return {"embeddings": raw.get("embeddings") or [],
                "stats": {"prompt_tokens": int(raw.get("prompt_eval_count") or 0), "load_ms": _ms(raw.get("load_duration")), "total_ms": _ms(raw.get("total_duration"))},
                "raw": {k: v for k, v in raw.items() if k != "embeddings"}}

    # -- models ------------------------------------------------------------------------------------------------------------------
    async def pull(self, model: str) -> AsyncIterator[dict]:
        async for obj in self._stream("/api/pull", {"model": model, "stream": True}):
            yield {"status": obj.get("status"), "completed": obj.get("completed"), "total": obj.get("total"), "digest": obj.get("digest")}

    async def delete(self, model: str) -> None:
        await self._json("DELETE", "/api/delete", {"model": model}, model=model)

    async def load(self, model: str, keep_alive: str | int | None = None) -> dict:
        body: dict = {"model": model, "stream": False}
        if keep_alive is not None:
            body["keep_alive"] = keep_alive_value(keep_alive)
        raw = await self._json("POST", "/api/generate", body, timeout=1800.0, model=model)
        return {"load_ms": _ms(raw.get("load_duration"))}

    async def unload(self, model: str) -> None:
        await self._json("POST", "/api/generate", {"model": model, "keep_alive": 0, "stream": False}, timeout=60.0, model=model)

    async def machine(self) -> dict:
        return machine.memory()
