"""`sllm`: the operator's command on the machine. `init` and `serve` work on the files; everything else asks the running
service over its socket, where the operator needs no key (run it as the service's user, or with sudo).

  sllm init [--name NAME]            the config, the data directory, the node id; the first node key, shown once
  sllm serve                         run the service (launchd / systemd run this)
  sllm status [--json]               the node, its backend, lanes and models
  sllm models                        the catalogue
  sllm keys mint NAME [--kind inference|node] [--max-priority N] [--max-concurrent N] [--models a,b]
  sllm keys list | revoke ID
  sllm prove [--model M] [--embed-model M]   a chat, a stream and an embedding through the queue; non-zero on any failure
"""
from __future__ import annotations

import argparse
import asyncio
import json
import socket
import sys
import time

import httpx

from . import __version__
from .config import NodeConfig


def die(msg: str, code: int = 1):
    print(f"sllm: {msg}", file=sys.stderr)
    raise SystemExit(code)


def running(cfg: NodeConfig) -> bool:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1)
    try:
        s.connect(str(cfg.socket_path))
        return True
    except OSError:
        return False
    finally:
        s.close()


def client(cfg: NodeConfig, timeout: float = 30.0) -> httpx.Client:
    if not cfg.socket_path.exists():
        die(f"the service is not running (no socket at {cfg.socket_path}): start it with `sllm serve`")
    return httpx.Client(transport=httpx.HTTPTransport(uds=str(cfg.socket_path)), base_url="http://sllm", timeout=timeout)


def ask(c: httpx.Client, method: str, path: str, body: dict | None = None) -> dict:
    try:
        r = c.request(method, path, json=body)
    except httpx.ConnectError:
        die("the service is not answering on its socket (is it running, and are you its user or root?)")
    except httpx.HTTPError as e:
        die(f"{method} {path}: {e.__class__.__name__}: {e}")
    try:
        data = r.json()
    except ValueError:
        data = {}
    if r.status_code >= 400:
        err = data.get("error") if isinstance(data, dict) else None
        die(f"{method} {path}: {r.status_code} {err.get('code') if isinstance(err, dict) else ''}: {err.get('message') if isinstance(err, dict) else r.text[:200]}")
    return data


# -- init, serve -----------------------------------------------------------------------------------------------------------------
async def _init(cfg: NodeConfig, name: str | None) -> None:
    from .keys import Keys
    from .store import Store
    store = await Store(cfg.data_dir).open()
    try:
        if name:
            await store.set_settings({"name": name})
        print(f"node id:   {store.node_id}")
        print(f"name:      {await store.setting('name')}")
        print(f"data:      {cfg.data_dir}")
        if not [k for k in await store.keys("node") if k["status"] == "active"]:
            raw, _ = await Keys(store).mint("node", "manager")
            print(f"node key:  {raw}")
            print("           shown once: it opens /node/v1 for a manager. An inference key for a caller: sllm keys mint NAME")
        else:
            print("node key:  already minted (sllm keys list)")
    finally:
        await store.close()


def cmd_init(cfg: NodeConfig, a) -> int:
    if running(cfg):
        die("the service is running: init works on the files, stop it first")
    path = cfg.path
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# maluos-local-llm-server: the node's own settings. No secret here: keys are hashes in the database.\n"
                        f'port = {cfg.port}\nbind = "{cfg.bind}"\nbackend_url = "{cfg.backend_url}"\n'
                        + (f'name = "{a.name}"\n' if a.name else ""))
        print(f"config:    {path} (written)")
    else:
        print(f"config:    {path}")
    asyncio.run(_init(cfg, a.name))
    return 0


def cmd_serve(cfg: NodeConfig, a) -> int:
    from .app import serve
    if running(cfg):
        die(f"a service already answers on {cfg.socket_path}")
    asyncio.run(serve(cfg))
    return 0


# -- reading the running node ------------------------------------------------------------------------------------------------
def _gb(n) -> str:
    return f"{n / 2 ** 30:.1f} GB" if isinstance(n, (int, float)) else "?"


def cmd_status(cfg: NodeConfig, a) -> int:
    with client(cfg) as c:
        st = ask(c, "GET", "/node/v1/status")
    if a.json:
        print(json.dumps(st, indent=1))
        return 0
    b = (st.get("backends") or [{}])[0]
    m = st.get("machine") or {}
    print(f"node:      {st['name']} ({st['node_id']}), version {st['version']}, started {st['started_at']}")
    print(f"listen:    {st['listen']['bind']}:{st['listen']['port']}")
    signed = b.get("cloud_signed_in")
    print(f"backend:   {b.get('kind')} at {b.get('endpoint')}: {'answering' if b.get('healthy') else 'NOT answering'}"
          + (f", version {b['version']}" if b.get("version") else "")
          + ("" if signed is None else f", ollama.com account {'signed in' if signed else 'not signed in'}"))
    print(f"machine:   {m.get('hostname')}, {m.get('os')} {m.get('arch')}, {m.get('cpu_count')} cores, memory {_gb(m.get('memory_free_bytes'))} free of {_gb(m.get('memory_total_bytes'))}")
    print(f"models:    {st['models']['local']} local, {st['models']['cloud']} cloud; loaded: {', '.join(st.get('loaded') or []) or 'none'}")
    for name, ln in (st.get("lanes") or {}).items():
        print(f"lane:      {name:<12} capacity {ln['capacity'] or 'unbounded'}, running {ln['running']}, queued {ln['queued']}" + (", PAUSED" if ln["paused"] else ""))
    jobs = (st.get("counters") or {}).get("jobs") or {}
    print("jobs:      " + (", ".join(f"{k} {v}" for k, v in sorted(jobs.items())) or "none yet"))
    return 0 if b.get("healthy") else 2


def cmd_models(cfg: NodeConfig, a) -> int:
    with client(cfg) as c:
        models = ask(c, "GET", "/api/tags").get("models") or []
    if not models:
        print("no models: pull one with Ollama (ollama pull qwen3:0.6b); the node lists it within a minute")
        return 0
    for m in models:
        d = m.get("details") or {}
        print(f"{m['name']:<40} {_gb(m.get('size')):>9}  {d.get('family') or '':<12} {d.get('parameter_size') or '':<8} {d.get('quantization_level') or ''}")
    return 0


def cmd_keys(cfg: NodeConfig, a) -> int:
    with client(cfg) as c:
        if a.keys_cmd == "mint":
            body: dict = {"name": a.name, "kind": a.kind}
            if a.max_priority is not None:
                body["max_priority"] = a.max_priority
            if a.max_concurrent is not None:
                body["max_concurrent"] = a.max_concurrent
            if a.models:
                body["allowed_models"] = [m.strip() for m in a.models.split(",") if m.strip()]
            r = ask(c, "POST", "/node/v1/keys", body)
            print(r["key"])
            print(f"shown once: the {r['kind']} key '{r['name']}' ({r['id']}). Only its hash is kept.", file=sys.stderr)
        elif a.keys_cmd == "list":
            for k in ask(c, "GET", "/node/v1/keys")["keys"]:
                print(f"{k['id']}  {k['kind']:<9} {k['status']:<8} {k['prefix']}…  {k['name']}" + (f"  last used {k['last_used_at']}" if k.get("last_used_at") else ""))
        elif a.keys_cmd == "revoke":
            r = ask(c, "DELETE", f"/node/v1/keys/{a.id}")
            print(f"revoked: {r['name']} ({r['id']})")
    return 0


# -- prove -----------------------------------------------------------------------------------------------------------------------
def cmd_prove(cfg: NodeConfig, a) -> int:
    """The live proof: what the queue serves is what the backend ran. Each check prints ok or FAIL."""
    failed = 0

    def check(name: str, ok: bool, detail: str = "") -> bool:
        nonlocal failed
        failed += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
        return ok

    with client(cfg, timeout=600.0) as c:
        r = c.get("/node/v1/health")
        if not check("the backend answers", r.status_code == 200, "" if r.status_code == 200 else "Ollama is not reachable from the node"):
            return 1
        models = [m["name"] for m in ask(c, "GET", "/api/tags").get("models") or []]
        if not check("the catalogue lists models", bool(models), ", ".join(models[:6]) or "none: ollama pull qwen3:0.6b"):
            return 1
        model = a.model or next((m for m in models if "embed" not in m and not m.rpartition(":")[2].endswith("cloud")), models[0])
        check("the model is in the catalogue", model in models, model)
        # a chat that waits for its whole answer
        t0 = time.monotonic()
        r = c.post("/api/chat", json={"model": model, "messages": [{"role": "user", "content": "Say hello in five words."}], "stream": False, "think": False})
        ok = r.status_code == 200 and r.json().get("done") is True
        d = r.json() if ok else {}
        tps = d.get("eval_count", 0) / (d.get("eval_duration", 0) / 1e9) if d.get("eval_duration") else 0
        check("a chat is answered", ok, f"{d.get('prompt_eval_count')} tokens in, {d.get('eval_count')} out, {tps:.1f} tokens/s, {time.monotonic() - t0:.1f} s; job {r.headers.get('x-node-job-id')}"
              if ok else f"{r.status_code} {r.text[:200]}")
        if ok:
            check("the answer has text", bool((d.get("message") or {}).get("content", "").strip()), repr((d.get("message") or {}).get("content", "")[:60]))
            check("the node headers are on the answer", all(r.headers.get(h) for h in ("x-node-id", "x-node-job-id", "x-node-served-model", "x-maluos-request")))
        # the same as a stream
        n, last = 0, {}
        with c.stream("POST", "/api/chat", json={"model": model, "messages": [{"role": "user", "content": "Count from one to five."}], "think": False}) as r:
            status = r.status_code
            for line in r.iter_lines():
                if line.strip():
                    last = json.loads(line)
                    n += 1
        check("a chat streams", status == 200 and n >= 2 and last.get("done") is True and not last.get("error"), f"{n} lines, {last.get('eval_count')} tokens out" if status == 200 else f"{status}")
        # a completion
        r = c.post("/api/generate", json={"model": model, "prompt": "One word for the colour of the sky:", "stream": False, "think": False})
        check("a completion is answered", r.status_code == 200 and r.json().get("done") is True, repr(r.json().get("response", "")[:40]) if r.status_code == 200 else f"{r.status_code} {r.text[:200]}")
        # an embedding, when the node has a model for one
        emb = a.embed_model or next((m for m in models if "embed" in m), None)
        if emb:
            r = c.post("/api/embed", json={"model": emb, "input": ["the queue in front of a machine that runs models"]})
            vec = (r.json().get("embeddings") or [[]])[0] if r.status_code == 200 else []
            check("an embedding is answered", r.status_code == 200 and len(vec) > 0, f"{emb}: {len(vec)} dimensions" if vec else f"{r.status_code} {r.text[:200]}")
        else:
            print("skip an embedding: no embedding model on the node (ollama pull nomic-embed-text)")
        # an unknown model is refused before it queues
        r = c.post("/api/chat", json={"model": "no-such-model:0", "messages": []})
        check("an unknown model is refused 404", r.status_code == 404 and (r.json().get("error") or {}).get("code") == "model_unknown")
        st = ask(c, "GET", "/node/v1/status")
        check("nothing is left in the queue", all(ln["running"] == 0 and ln["queued"] == 0 for ln in st["lanes"].values()))
    print(f"{'PROVEN' if not failed else 'NOT PROVEN'}: {failed} failure(s)")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sllm", description="maluos-local-llm-server: the queue in front of a machine that runs models")
    ap.add_argument("--version", action="version", version=f"sllm {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init", help="the config, the data directory, the first node key")
    p.add_argument("--name")
    sub.add_parser("serve", help="run the service")
    p = sub.add_parser("status", help="the node, its backend, lanes and models")
    p.add_argument("--json", action="store_true")
    sub.add_parser("models", help="the catalogue")
    p = sub.add_parser("keys", help="mint, list, revoke")
    ks = p.add_subparsers(dest="keys_cmd", required=True)
    m = ks.add_parser("mint")
    m.add_argument("name")
    m.add_argument("--kind", choices=("inference", "node"), default="inference")
    m.add_argument("--max-priority", type=int)
    m.add_argument("--max-concurrent", type=int)
    m.add_argument("--models", help="the models this key may use, comma-separated (default: any)")
    ks.add_parser("list")
    ks.add_parser("revoke").add_argument("id")
    p = sub.add_parser("prove", help="a chat, a stream and an embedding through the queue")
    p.add_argument("--model")
    p.add_argument("--embed-model")
    a = ap.parse_args(argv)
    cfg = NodeConfig.load()
    return {"init": cmd_init, "serve": cmd_serve, "status": cmd_status, "models": cmd_models, "keys": cmd_keys, "prove": cmd_prove}[a.cmd](cfg, a)


if __name__ == "__main__":
    raise SystemExit(main())
