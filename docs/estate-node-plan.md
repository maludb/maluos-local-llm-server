# Build plan: the node service as an estate inference node on macOS and DGX Spark

2026-11-10. This plan takes `maluos-local-llm-server` from its current state (store, scheduler, keys, a fake
backend; commit `3328e84`) to the point where an Apple Silicon Mac or a DGX Spark, keeping its vendor operating
system, serves models to a MaluOS estate. It amends `docs/server-design.md` where the two disagree; section 2 lists
the amendments.

The operator's process this plan must make true:

1. Set up the host with its own operating system and Ollama (the guides `11-gpu-macos.md`, `11-gpu-dgx-spark.md`).
2. Install the node service: `install.sh`, then `sudo sllm init`.
3. Join the estate: `sudo sllm estate join https://app.DOMAIN --code CODE --name NAME`, with an `inference` code
   minted at the office. The node then appears under Settings → Estate → Shared inference, and
   `maluctl gateway chat --prefer estate` on any member is served by it.

## 1. Decisions (the owner, 2026-11-10)

| # | Decision |
|---|---|
| P1 | The node joins the estate **directly** with a one-time `inference` node code, as a MaluOS host does. The Sovereign LLM Manager is optional and not on this path. |
| P2 | **WireGuard overlay on both platforms.** The kernel publishes a backend only when it has an overlay address, and that rule stays. |
| P3 | **Ollama cloud is allowed when the host is signed in, and a cloud model is treated like a local model**: one catalogue, the same keys, no separate cloud policy and no fallback machinery. |
| P4 | A cloud call runs **beside** the local inference; it does not wait for the GPU slot. |
| P5 | A node setting `hide_cloud_from_estate` (default **off**) removes cloud models from what estate peers see and may call. It exists because the estate treats this node's models as private. |
| P6 | **The Ollama-native face and `/node/v1` health and status come first**; that is all the estate dials. The OpenAI and Anthropic faces and the rest of `/node/v1` follow. |

## 2. Amendments to the server design

| Design | Becomes |
|---|---|
| §1: faces at `/openai/v1` and `/anthropic/v1` | `/v1/...` as `maluos/docs/inference-node-api.md` §5 has them, plus the Ollama face `/api/...` |
| §3, R6, S4, S9: the cloud lane's policy, `cloud_allowed`, `fallback_allowed`, fallback models and thresholds | removed for Ollama cloud (P3). The `cloud` lane stays only as a capacity (P4) |
| §2: priority 1 to 9 per key | kept inside the scheduler; the wire carries the estate's classes: `interactive` = 7, `agent` = 5, `batch` = 3 |
| S10: the node never dials out; registration later | the node dials the office: join, heartbeat, peers (P1) |
| §8: one process | the service as a dedicated user, plus a small root helper that owns the tunnel |
| S12: Apache-2.0 | the repository is MIT; the design's line is corrected |

The scheduler's fallback code and its tests stay in the tree, unused, until the owner says to delete them.

## 3. What the estate requires of a node

Read from the MaluOS code, not from the documents. Each line is a test in the slices below.

**The caller's gateway** (`maluos_gateway/remote.py`, `estate.py`) sends, with `Authorization: Bearer <key>`:

- `GET /api/tags`, `GET /api/ps`, `GET /api/version` every refresh; a 401 marks the rung "key refused".
- `POST /api/chat`, `/api/generate`, `/api/embed`; streaming answers are newline-delimited JSON as Ollama's.
- Request headers `X-MaluOS-Origin-Request`, `X-MaluOS-Origin-Node`, `X-MaluOS-Origin-Node-Name`,
  `X-MaluOS-Class`, `X-MaluOS-Max-Wait`, `X-MaluOS-Sensitivity`, `X-MaluOS-Envelope`.
- It reads the answer header `X-MaluOS-Request` as the serving node's call id, and errors as
  `{"error": {"code", "message"}}` with the codes `not_shared`, `model_not_found`, `unauthorized`, `policy`,
  `queue_full`, `wait_exceeds`, `classification`.

**The office** (`/api/v1/estate/...`, bearer = the node token):

- `POST join` with `{surface: 1, code, kind: "inference", name, principal, public_key, overlay_public_key,
  hostname, os_platform, os_release, cpu_cores, memory_mb, storage_gb, machine_id}` answers `token`,
  `inference_key`, `node`, `endpoints`, `overlay`, `heartbeat_s`, `report_s`.
- `POST heartbeat` every `heartbeat_s` with `inference: {models, lanes, queue, expected_wait_s, share,
  contribution, port, api_version}` (at most 8 KB). The answer's `overlay_changed`, `inference_changed` and
  `peers_changed` each ask for a refresh. A 401 or 410 means the node was retired.
- `GET inference/peers` lists the members allowed to call this node: `key_hash` (the SHA-256 of the member's
  inference key), `node`, `class_max`, `classification_max`, `priority`; refreshed every 300 s and on `peers_changed`.
- `GET overlay` gives `self`, `hub` and `peers` for the tunnel.
- The node is published only while it is online, `share` is true and it has an overlay address.

## 4. Shape of the result

```
/opt/maluos-llm (Linux)  /usr/local/maluos-llm (macOS)
  venv/               a pinned Python 3.12 environment
  config/node.toml    port, bind, backend URL, name
  data/node.sqlite3   runs, keys, settings          (the service user's, 0600)
  data/estate.json    the enrolment: office, node, endpoints (no secret)
  data/secrets.json   node token, inference key, WireGuard key   (0600)

sllm.service / org.maluos.sllm            the node service, user `sllm`, port 11435
sllm-overlay.service / org.maluos.sllm-overlay    root: the `estate` tunnel and nothing else
```

The service binds loopback until it has joined; after the join it also binds the overlay address, never the LAN,
unless the operator pairs by hand with `sllm expose`.

## 5. Slices

Each slice ends with a proof that runs without the hardware, then a check on the hardware where one is needed.

### N1. A runnable node on one machine (L)

- `pyproject.toml` with a lock file; `install.sh` for a development checkout; `tests/run.sh` builds its own venv.
- `server/backends/ollama.py`: the native API over `OLLAMA_HOST`: tags, ps, show, version, chat, generate, embed,
  load, unload; streaming; a cancelled job aborts the backend request.
- `server/app.py`: the Ollama face (`/api/tags`, `/api/ps`, `/api/version`, `/api/show`, `/api/chat`,
  `/api/generate`, `/api/embed`), every inference a job through the existing scheduler.
- `/node/v1/health` and `/node/v1/status` with the envelope (`node_id`, `node_name`, `api_version`, `clock`).
- The catalogue: local and cloud models in one list; a cloud model is marked `remote` for display and for P4's
  capacity only. A host that is not signed in lists no cloud models.
- `sllm init | serve | status | keys mint|list|revoke | prove`.
- Proof: the fake backend suite extended to the face (stream and not, embed, cancel, the local slot holding while
  cloud jobs overlap); `tests/live.sh` against a real Ollama with `qwen3:0.6b`.

### N2. The serving side speaks the estate's contract (M)

- Classes on the wire mapped to priorities; `expected_wait_s` per class from the queue ahead.
- The peers table: a presented key is looked up by SHA-256 among the node's own keys, then among the estate's
  peers; a peer is the principal `node:<ref>`, capped at `class_max`, refused above `classification_max`.
- The share switch (`sllm share on|off`); off answers 503 `not_shared` to network callers.
- The run record gains `principal`, `class`, `origin_request`, `origin_node`, `origin_node_name`, `envelope`,
  `sensitivity`; answers carry `X-MaluOS-Request`, `X-Node-Id`, `X-Node-Job-Id`, `X-Node-Served-Model`,
  `X-Node-Queue-Wait-Ms`.
- `hide_cloud_from_estate` (P5).
- The facts document for the heartbeat, in the kernel's shape.
- Proof: on the MaluOS dev host, the real gateway pairs with the node by hand (`maluctl gateway backend add NAME
  http://127.0.0.1:11435 --key sllm_...`) and `maluctl gateway chat --prefer NAME` is served by it; both ledgers
  carry the same request id. This proves the wire before any estate code exists.

### N3. The estate client (M)

- `server/estate/client.py`: join, heartbeat, me, leave, peers, overlay, a new inference key. A port of
  `maluos_estate/client.py` without the vault and without `/proc`.
- The heartbeat loop inside the service: presence, the machine facts, the inference facts; the three "changed"
  flags; retirement stops sharing and says so in `sllm estate status`.
- A minimal report (disk, memory, load, the service's state) so the office's health rules have something to read.
  To settle in this slice: what the kernel's rules do with a node that sends no units, agents or updates.
- `sllm estate join | status | leave | peers | heartbeat`.
- Machine facts per platform: memory, cores, disk, a stable machine id, the operating system's name, and the GPU
  (`nvidia-smi` on the Spark, `system_profiler` on the Mac) in the shape `maluctl gpu status --json` reports.
- Proof: `python3 -m maluos_estate.mock_office --node-code CODE` from the maluos repository as the office: join,
  facts stored, the peers answer honoured, a peer's key accepted and a retired peer's refused, leave. Then the
  kernel's own proof (`bin/test_estate_inference.php`) with the node as a member.

### N4. The overlay and the package on Linux, for the Spark (M)

- `server/estate/overlay.py`: a port of `maluos_estate/overlay.py` (the same `ip` and `wg` commands, the same
  reconciliation, hub or spoke although a node is normally a spoke), reading the key from `secrets.json`.
- `deploy/linux/`: `sllm.service`, `sllm-overlay.service`, the `sllm` user, `install.sh`, `uninstall.sh`, upgrade
  in place keeping `data/`.
- After the join, the listener binds the overlay address; `sllm doctor` checks the tunnel, the hub's handshake,
  a firewall on port 11435 and Ollama's reachability.
- Proof: an Ubuntu VM joined to the mock office with the overlay in its dry-run mode; then an arm64 run.
- On the Spark: the tunnel answers `ping office.internal`; the office lists the node with its models; a chat from
  the office is served by the GB10; the same after a reboot.

### N5. macOS (L)

- The overlay on macOS: `wireguard-go` and `wg` (the tunnel is a `utun` device, so the interface cannot be named
  `estate`), `ifconfig` and `route` in place of `ip`. A separate implementation behind the same interface as N4.
- `deploy/macos/`: two LaunchDaemons (the service as a dedicated user, the overlay as root), `install.sh`,
  `uninstall.sh`.
- Ollama without a login session: a LaunchDaemon that runs `ollama serve` with `OLLAMA_HOST=127.0.0.1:11434` and
  the capacity settings in its plist, replacing the guide's `launchctl setenv` and menu-bar app. The guide keeps
  the app as the route for a first trial only.
- Sleep: `pmset` settings in the guide; the node reports itself offline cleanly when the Mac sleeps.
- Proof on the owner's Mac: join, tunnel, a chat from the office served on Metal, the same after a reboot with
  nobody logged in.

### N6. Acceptance and the documents (M)

- The three-machine proof's steps 2 and 3 with the Mac and the Spark as the inference nodes: the call in both
  ledgers, a deny rule at the office removes the rung, `sllm share off` refuses, a restricted call is refused.
- A signed-in host: a `-cloud` model listed at the office and served to a peer; with `hide_cloud_from_estate` on,
  it is neither listed nor served.
- Guides: `11-arm-inference.md`, `11-gpu-macos.md`, `11-gpu-dgx-spark.md` and `11-compute-roadmap.md` rewritten
  around the three steps (cloud no longer disabled, `ollama signin` optional, the join commands); `13-estate.md`
  loses its "do not apply to these hosts" paragraph; the dangling verification-note link removed.
- maluos: `docs/inference-node-api.md` (the node service column), `docs/roadmap.md` G3, the deferred note in
  `docs/phase13-slice3-inference.md`.

### N7. Afterwards (not in this plan's scope)

The OpenAI and Anthropic faces, async jobs, the rest of `/node/v1` (queue, runs, models, keys) for the manager,
model pulls from the office, a keychain for the secrets on macOS.

## 6. Changes outside this repository

None is required in the kernel or in the MaluOS packages: the join accepts any `os_platform`, and the gateway's
estate rung already speaks to anything that answers the Ollama face. Two things to look at during N3 and decide
then: how the office's Estate page shows a node with no MaluOS components, and whether its remote actions
(restart an agent, apply updates) should be hidden for such a node.

One consequence of P3 to keep in view: a member's gateway counts this node as private, so a caller with a
private-only policy can be served by an Ollama cloud model here. P5 is the switch for an estate where that matters.

## 7. To verify on the hardware, early

| Question | Where it bites | Fallback |
|---|---|---|
| Does DGX OS carry the WireGuard kernel module and `wireguard-tools`? | N4 | `wireguard-go` as on macOS |
| Does `nvidia-smi` report memory for the GB10's unified memory? | N3 facts | report system memory as the GPU's |
| Python 3.12 on DGX OS and on macOS without Homebrew | N1 install | the installer fetches a pinned Python |
| How does Ollama say "not signed in" for a cloud model, and can sign-in be read without a call? | N1 catalogue | probe one cloud model at start and on a failure |
| Can the Mac's and the Spark's networks reach the hub on UDP 51820? | N4, N5 | the proxy host's `stream` block, as the office uses |
| `ollama serve` as a LaunchDaemon with Metal, nobody logged in | N5 | a LaunchAgent with automatic login, recorded as a limit |

## 8. Order and size

N1 → N2 → N3 → N4 (the Spark is usable here) → N5 (the Mac) → N6. N1 to N3 need no hardware and can be built
and proven on the MaluOS dev host. N4 and N5 each end on the owner's machine. Three large or medium slices before
the first hardware run; the Spark comes first because the Linux overlay code carries over unchanged.

## 9. Status

**N1 built 2026-11-10** on the branch `n1-runnable-node` (not committed at the time of writing).

- Built: `pyproject.toml`, `install.sh`, `requirements.lock`; `server/backends/ollama.py`; `server/app.py` (the Ollama
  face, `/node/v1/health`, `/node/v1/status`, `/node/v1/keys`, the TCP listener and the operator's socket);
  `server/cli.py` (`sllm init | serve | status | models | keys | prove`); `server/machine.py`; `tests/live.sh`.
- Changed in the scheduler: a job's request is held in memory while it waits and runs (it was read back from the
  payload store, which keeps a request only under `full` retention, so a real backend would have received no
  messages); the `generate` kind; `cloud_allowed` no longer gates a cloud model (P3).
- Proof: `tests/run.sh`, 48 checks: the scheduler (20, one changed for P3), the Ollama backend against a recorded
  daemon (11), the face over the app (12), the service on real sockets with the CLI (5). `tests/live.sh` passed
  against the mock Ollama of the maluos repository (`maluos_gateway.mock_ollama`) over real HTTP.
- Not done in N1: `tests/live.sh` against a real Ollama (the dev host has none), and so nothing about a real model, a
  GPU, or a signed-in account. The sign-in check (`POST /api/me`) is written from Ollama's documentation and unproven.
- Left for N2: a caller waiting in the queue receives no bytes until its job starts (Ollama's stream has no comment
  line; the MaluOS gateway allows 600 s); the scheduler's `ended` table keeps one entry per job for the life of the
  process; the fallback code is still in the tree, unused.
