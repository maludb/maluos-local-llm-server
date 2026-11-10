# maluos-local-llm-server

The queue in front of a machine that runs models. It sits on a computer with [Ollama](https://ollama.com) (an Apple
Silicon Mac, a DGX Spark, any Linux box), takes model calls from the network with a key, runs one local inference at a
time, keeps a record of every run, and serves a MaluOS estate as an inference node.

- Design: [docs/server-design.md](docs/server-design.md)
- Build plan for macOS and DGX Spark as estate nodes: [docs/estate-node-plan.md](docs/estate-node-plan.md)

## State

Slice N1 of the build plan: a runnable node on one machine.

| Part | State |
|---|---|
| Store, scheduler, keys | built |
| Ollama backend (native API, streaming, cancel) | built |
| Ollama face: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, `/api/version`, `/api/show` | built |
| `/node/v1/health`, `/node/v1/status`, `/node/v1/keys` | built |
| `sllm` CLI: `init`, `serve`, `status`, `models`, `keys`, `prove` | built |
| Estate contract on the serving side (classes, peers, share switch) | N2, not built |
| Estate client, WireGuard overlay, service packaging | N3 to N5, not built |
| OpenAI and Anthropic faces, the rest of `/node/v1` | later |

## Try it

Needs Python 3.12 or newer and a running Ollama with at least one model (`ollama pull qwen3:0.6b`).

```bash
./install.sh                         # a virtual environment in .venv
.venv/bin/sllm init --name my-node   # the config, the data directory, the node key (shown once)
.venv/bin/sllm serve                 # listens on 127.0.0.1:11435 and on data/sllm.sock
```

In another terminal:

```bash
.venv/bin/sllm status                # the node, Ollama, the lanes
.venv/bin/sllm prove                 # a chat, a stream, a completion and an embedding through the queue
.venv/bin/sllm keys mint my-laptop   # an inference key for a caller, shown once
curl -s http://127.0.0.1:11435/api/chat -H "Authorization: Bearer sllm_..." \
  -d '{"model": "qwen3:0.6b", "messages": [{"role": "user", "content": "Say hello in five words."}], "stream": false}'
```

The network listener always needs a key. The `sllm` command talks to the service over a socket in the data directory
and needs none; run it as the user that runs the service.

A cloud model of a signed-in Ollama (`ollama signin`) is listed and served like a local one; it runs beside the local
inference rather than waiting for the GPU.

## Proofs

```bash
tests/run.sh          # without Ollama: the scheduler, the Ollama backend against a recorded daemon, the face, real sockets
tests/live.sh         # with Ollama on this machine, in a throwaway data directory
```

## Settings

`config/node.toml` (written by `sllm init`): `port`, `bind`, `backend_url`, `name`, `data_dir`, `socket`. The environment
overrides it: `SLLM_HOME` (where `config/` and `data/` live), `SLLM_PORT`, `SLLM_BIND`, `OLLAMA_HOST`, `SLLM_DATA_DIR`,
`SLLM_SOCKET`. The node refuses to listen beyond loopback until a key exists.
