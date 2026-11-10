"""The Ollama backend against a recorded shape of the daemon's native API (no Ollama needed): the catalogue with a cloud model,
the chunk protocol with the daemon's own numbers, embeddings, the failures."""
import json

import httpx
import pytest

from server.backends.base import BackendError, ModelMissing
from server.backends.ollama import OllamaBackend, is_cloud, keep_alive_value

FINAL = {"done": True, "done_reason": "stop", "total_duration": 1_500_000_000, "load_duration": 400_000_000, "prompt_eval_count": 12,
         "prompt_eval_duration": 100_000_000, "eval_count": 3, "eval_duration": 600_000_000}
TAGS = {"models": [
    {"name": "qwen3:0.6b", "model": "qwen3:0.6b", "size": 522653526, "digest": "aaa", "details": {"family": "qwen3", "parameter_size": "751.63M", "quantization_level": "Q4_K_M"}},
    {"name": "nomic-embed-text:latest", "model": "nomic-embed-text:latest", "size": 274302450, "digest": "bbb", "details": {"family": "nomic-bert", "parameter_size": "137M", "quantization_level": "F16"}},
    {"name": "gpt-oss:120b-cloud", "model": "gpt-oss:120b-cloud", "size": 384, "digest": "ccc", "remote_model": "gpt-oss:120b", "remote_host": "https://ollama.com:443",
     "details": {"family": "gptoss", "parameter_size": "116.8B", "quantization_level": "MXFP4"}}]}
SHOW = {"qwen3:0.6b": {"details": {"family": "qwen3"}, "model_info": {"general.architecture": "qwen3", "qwen3.context_length": 40960}, "capabilities": ["completion", "tools", "thinking"]},
        "nomic-embed-text:latest": {"details": {"family": "nomic-bert"}, "model_info": {"nomic-bert.context_length": 2048}, "capabilities": ["embedding"]},
        "gpt-oss:120b-cloud": {"details": {"family": "gptoss"}, "model_info": {"gptoss.context_length": 131072}, "capabilities": ["completion", "tools", "thinking"]}}


def ndjson(*objs):
    return ("\n".join(json.dumps(o) for o in objs) + "\n").encode()


class Daemon:
    """What the node asked, and what the daemon answers."""

    def __init__(self, signed_in=True):
        self.signed_in, self.seen, self.fail = signed_in, [], None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.seen.append((request.method, path, body))
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.13.5"})
        if path == "/api/tags":
            return httpx.Response(200, json=TAGS)
        if path == "/api/me":
            return httpx.Response(200, json={"name": "owner"}) if self.signed_in else httpx.Response(401, json={"error": "unauthorized"})
        if path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:0.6b", "model": "qwen3:0.6b", "size": 900000000, "size_vram": 900000000, "expires_at": "2026-11-10T12:05:00Z"}]})
        if path == "/api/show":
            return httpx.Response(200, json=SHOW[body["model"]]) if body["model"] in SHOW else httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
        if path in ("/api/chat", "/api/generate") and body.get("model") not in SHOW:
            return httpx.Response(404, json={"error": f"model '{body.get('model')}' not found"})
        if path == "/api/chat":
            if self.fail == "midstream":
                return httpx.Response(200, content=ndjson({"model": "m", "message": {"role": "assistant", "content": "He"}, "done": False}, {"error": "an error was encountered while running the model"}))
            if self.fail == "signin":
                return httpx.Response(401, json={"error": "unauthorized"})
            if body.get("tools"):
                return httpx.Response(200, content=ndjson(
                    {"model": "m", "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Perth"}}}]}, "done": False},
                    {"model": "m", "created_at": "t", "message": {"role": "assistant", "content": ""}, **FINAL}))
            return httpx.Response(200, content=ndjson(
                {"model": "m", "message": {"role": "assistant", "content": "", "thinking": "hmm "}, "done": False},
                {"model": "m", "message": {"role": "assistant", "content": "Hello"}, "done": False},
                {"model": "m", "message": {"role": "assistant", "content": " there"}, "done": False},
                {"model": "m", "created_at": "t", "message": {"role": "assistant", "content": ""}, **FINAL}))
        if path == "/api/generate":
            if not body.get("prompt"):
                return httpx.Response(200, json={"model": body["model"], "done": True, "done_reason": "load" if body.get("keep_alive") != 0 else "unload", "load_duration": 2_000_000_000})
            return httpx.Response(200, content=ndjson({"model": "m", "response": "blue", "done": False}, {"model": "m", "response": "", "context": [1, 2, 3], **FINAL}))
        if path == "/api/embed":
            n = len(body["input"]) if isinstance(body["input"], list) else 1
            return httpx.Response(200, json={"model": body["model"], "embeddings": [[0.5, 0.25]] * n, "total_duration": 20_000_000, "load_duration": 1_000_000, "prompt_eval_count": 7})
        return httpx.Response(404, json={"error": "not found"})


def backend(daemon):
    return OllamaBackend("http://ollama.test", client=httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(daemon)))


async def collect(it):
    return [c async for c in it]


async def test_catalogue_lists_cloud_beside_local():
    b = backend(Daemon())
    assert await b.health() and await b.version() == "0.13.5"
    models = {m["name"]: m for m in await b.list_models()}
    assert set(models) == {"qwen3:0.6b", "nomic-embed-text:latest", "gpt-oss:120b-cloud"}
    assert models["qwen3:0.6b"]["lane"] == "local" and models["gpt-oss:120b-cloud"]["lane"] == "cloud"
    assert models["qwen3:0.6b"]["context_length"] == 40960 and models["qwen3:0.6b"]["capabilities"] == ["chat", "tools", "thinking"]
    assert models["nomic-embed-text:latest"]["capabilities"] == ["embedding"]
    assert models["qwen3:0.6b"]["quantization"] == "Q4_K_M" and models["qwen3:0.6b"]["size_bytes"] == 522653526
    assert [m["name"] for m in await b.loaded()] == ["qwen3:0.6b"]


async def test_show_is_asked_once_per_digest():
    d = Daemon()
    b = backend(d)
    await b.list_models()
    await b.list_models()
    assert sum(1 for _, p, _ in d.seen if p == "/api/show") == 3


async def test_a_host_not_signed_in_lists_no_cloud_models():
    b = backend(Daemon(signed_in=False))
    assert await b.signed_in() is False
    assert {m["name"] for m in await b.list_models()} == {"qwen3:0.6b", "nomic-embed-text:latest"}


def test_cloud_is_told_by_the_remote_host_or_the_tag():
    assert is_cloud({"name": "x:latest", "remote_host": "https://ollama.com"})
    assert is_cloud({"name": "qwen3-coder:480b-cloud"}) and is_cloud({"name": "glm-4.6:cloud"})
    assert not is_cloud({"name": "qwen3:8b"}) and not is_cloud({"name": "cloudy:7b"})


async def test_chat_stream_keeps_the_daemons_numbers():
    d = Daemon()
    chunks = await collect(backend(d).chat({"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "hi"}], "keep_alive": "-1", "node": {"x": 1}, "stream": False}))
    assert [c["type"] for c in chunks] == ["delta", "delta", "delta", "final"]
    assert chunks[0]["thinking"] == "hmm " and chunks[1]["text"] == "Hello"
    final = chunks[-1]
    assert final["message"] == {"role": "assistant", "content": "Hello there", "thinking": "hmm "}
    assert final["stats"] == {"load_ms": 400, "prompt_tokens": 12, "prompt_eval_ms": 100, "output_tokens": 3, "eval_ms": 600, "total_ms": 1500}
    assert final["raw"]["eval_duration"] == 600_000_000 and "message" not in final["raw"]
    sent = d.seen[-1][2]
    assert sent["stream"] is True and sent["keep_alive"] == -1 and "node" not in sent, "the node always streams from the daemon; -1 goes as a number"


async def test_tool_calls_reach_the_final_message():
    chunks = await collect(backend(Daemon()).chat({"model": "qwen3:0.6b", "messages": [], "tools": [{"type": "function", "function": {"name": "get_weather"}}]}))
    assert chunks[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert chunks[-1]["stop_reason"] == "tool_calls" and chunks[-1]["message"]["tool_calls"][0]["function"]["arguments"] == {"city": "Perth"}


async def test_generate_drops_the_context_ids():
    chunks = await collect(backend(Daemon()).generate({"model": "qwen3:0.6b", "prompt": "sky?"}))
    assert chunks[0] == {"type": "delta", "text": "blue"}
    assert chunks[-1]["message"]["content"] == "blue" and "context" not in chunks[-1]["raw"] and chunks[-1]["stats"]["output_tokens"] == 3


async def test_embed():
    res = await backend(Daemon()).embed({"model": "nomic-embed-text:latest", "input": ["a", "b"]})
    assert res["embeddings"] == [[0.5, 0.25], [0.5, 0.25]] and res["stats"] == {"prompt_tokens": 7, "load_ms": 1, "total_ms": 20}
    assert "embeddings" not in res["raw"]


async def test_failures_are_named():
    d = Daemon()
    b = backend(d)
    with pytest.raises(ModelMissing):
        await collect(b.chat({"model": "nope", "messages": []}))
    d.fail = "midstream"
    with pytest.raises(BackendError) as e:
        await collect(b.chat({"model": "qwen3:0.6b", "messages": []}))
    assert e.value.code == "backend_error" and "error was encountered" in e.value.text
    d.fail = "signin"
    with pytest.raises(BackendError) as e:
        await collect(b.chat({"model": "gpt-oss:120b-cloud", "messages": []}))
    assert e.value.code == "not_signed_in" and e.value.status == 401

    def refuse(request):
        raise httpx.ConnectError("refused")
    down = OllamaBackend("http://ollama.test", client=httpx.AsyncClient(base_url="http://ollama.test", transport=httpx.MockTransport(refuse)))
    assert await down.health() is False
    with pytest.raises(BackendError) as e:
        await collect(down.chat({"model": "qwen3:0.6b", "messages": []}))
    assert e.value.code == "backend_down" and e.value.status == 503


async def test_load_and_unload():
    d = Daemon()
    b = backend(d)
    assert await b.load("qwen3:0.6b", "30m") == {"load_ms": 2000}
    assert d.seen[-1][2] == {"model": "qwen3:0.6b", "stream": False, "keep_alive": "30m"}
    await b.unload("qwen3:0.6b")
    assert d.seen[-1][2]["keep_alive"] == 0


def test_keep_alive_values():
    assert keep_alive_value("-1") == -1 and keep_alive_value("0") == 0 and keep_alive_value("300.0") == 300
    assert keep_alive_value("5m") == "5m" and keep_alive_value(60) == 60
