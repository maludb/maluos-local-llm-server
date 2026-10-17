# maluos-local-llm-server — design

2026-10-16 · the **queue in front of a machine that runs models**: a service that sits on a computer with Ollama (the
owner's Mac Mini M4 is the first), takes LLM calls in the **OpenAI** and the **Anthropic** wire formats from anything on the
network, queues them, runs **one local inference at a time** while **cloud models through the Ollama account run beside
it without limit**, falls back to a cloud model when a local one cannot keep up and the caller allows it, pulls, warms and
unloads models, keeps the run history with its performance numbers in a database of its own, and exposes all of that to
the **Sovereign LLM Manager** (`maludb-os-sovereign-llm-manager`, an application of the MaluDB Business OS) through a
management API. Many such machines — **nodes** — serve one business; the manager watches them all.

The owner's brief (2026-10-16): *a queueing system on a computer that runs local models and cloud models through Ollama;
one local prompt at a time, cloud beside it; the machine an API server in the OpenAI or Anthropic format for anything on
the network; token usage reported; the caller chooses batch (synchronous) or asynchronous and streaming is supported;
cloud is not limited and is also the fallback when local does not keep up; the server pulls, unloads and warms models;
the node keeps its own history and the manager pulls it, so the node stands alone when the manager is down.*

> **What this document is.** The design of the node service and, in §6, the **node API** — the contract the manager is
> built against. The manager's own design is `/srv/apps/sovereign_llm_manager/docs/sovereign-llm-manager-design.md`,
> which reads §6 by reference and never restates it. The decisions still open are §12 (S1–S12); the owner's answers go in
> §13. No code is written until the owner approves this document and the manager's together.

## 0. What is decided, and what Ollama and the estate say

### 0.1 Decided on 2026-10-16 — the owner's answers to the seven questions

| # | Decision | In one line |
|---|---|---|
| R1 | Two parts | the **node service** (this repository, on the machine with Ollama) and the **manager** (an OS application, in the estate) |
| R2 | Many nodes | the manager watches a fleet; every node API answer names the node |
| R3 | The node stands alone | Python; the node keeps its history in **SQLite** on the machine; the manager **pulls** through the node API; the node needs no manager to serve |
| R4 | Callers | anything on the network, each with a key; **token usage is reported** on every answer and kept per run |
| R5 | Modes | the caller chooses **sync** (the connection waits; streaming when asked) or **async** (a job id, polled or streamed later); both faces stream |
| R6 | Cloud | the cloud lane is **not limited**; a local job that cannot keep up **falls back to a cloud model** when the policy allows |
| R7 | Models | the node **pulls, unloads and warms** models, and keeps a model warm by policy |

### 0.2 What Ollama gives, in the shape we use

- **One daemon, two kinds of model.** Ollama runs local models from its own store and, once the machine is signed in
  (`ollama signin`), **cloud models** whose names end in `-cloud` (for example `gpt-oss:120b-cloud`): the daemon forwards
  those to ollama.com under the account, so a cloud call costs the machine nothing but a socket and runs while a local model
  is busy. The node routes cloud calls **through the local daemon** so the account's credential lives in one place
  (ollama.com directly with `OLLAMA_API_KEY` is a switch, §12 S9).
- **Local concurrency is the daemon's.** `OLLAMA_NUM_PARALLEL` and `OLLAMA_MAX_LOADED_MODELS` decide how many requests and
  models the daemon runs at once; a Mac Mini with one model's worth of memory runs **one**. The node never sends the local
  lane more than its capacity (default **1**), so the daemon never queues on its own and every wait is visible in ours.
- **Native endpoints carry the numbers.** `POST /api/chat` and `/api/generate` answer, in the final message, `load_duration`,
  `prompt_eval_count`, `prompt_eval_duration`, `eval_count`, `eval_duration` and `total_duration` — the load time, the prompt
  tokens, the output tokens and the time each took. The node talks to Ollama **natively** and translates both faces itself;
  Ollama's own `/v1/chat/completions` would lose the timings.
- **Warm, unload, keep.** A request's `keep_alive` keeps the model loaded for that long after it answers (`-1` for ever,
  `0` unloads at once); an empty `/api/chat` with a model name loads it (a warm); `/api/ps` lists what is loaded and its
  memory; `/api/tags` lists what is on disk; `/api/pull` streams a pull's progress; `/api/delete` removes; `/api/show` tells
  the family, parameters, context length and capabilities (vision, tools, embedding).
- **Embeddings** are `/api/embed`. Vision is images in a message. Tools are `tools` on the request and `tool_calls` on the
  answer, in a shape close to OpenAI's; the node maps Anthropic's `tool_use` / `tool_result` blocks onto it.
- **What Ollama cannot do:** tell a caller where it stands in a queue, keep a history, meter per caller, or hold a key per
  caller — that is this service.

### 0.3 What the estate already has, and what this is not

- **The kernel's ledger proxy** (`mcp/agent_runner/ledger_proxy.py`, port 8816) already forwards an agent's call to a
  registered model's `endpoint_url` with no key when the provider is local, prices it from `model_registry` and writes the
  prompt ledger. A node is **one more endpoint** there: a `model_registry` row of provider `local`, harness
  `openai_compatible`, `endpoint_url = http://<node>:11435/openai` and a per-node key in its `api_secret_id`. Nothing in the
  proxy changes; the node's `X-Node-*` answer headers are extra information the proxy may keep later (K58, the manager's
  design §12).
- **This is not an OS application.** It runs where the models run — the Mac, a GPU box, a desk — with no PostgreSQL, no
  MaluDB, no PHP and no kernel. It carries no `maludb-os.json`. Its one tie to the estate is the node API the manager
  pulls, and the keys the manager mints through it.
- **The estate's rules it keeps anyway:** nothing leaves the machine unless a policy says so (the cloud lane is that
  policy, per key and per model); every run is a row; secrets are never printed; a proof suite runs without a manager.

## 1. What it is, and is not

**Is.** One process, `maluos-local-llm-server`, listening on one port (default **11435**, Ollama's plus one) with three faces:

| Face | Prefix | Who | What |
|---|---|---|---|
| OpenAI | `/openai/v1/…` | any client speaking OpenAI's Chat Completions, Embeddings and Models | `chat/completions` (stream or not), `embeddings`, `models`, `completions` (Extended), plus the job routes of §4 |
| Anthropic | `/anthropic/v1/…` | any client speaking Anthropic's Messages | `messages` (stream or not), `models`, `messages/count_tokens` (Extended), plus the job routes |
| Node | `/node/v1/…` | the manager (and the local CLI) | status, lanes, queue, runs and their metrics, usage, models (pull / warm / unload / keep), keys, policies, settings, events |

Behind the faces: **lanes** (§3) with a scheduler, **jobs** (§4) in SQLite, a **backend** (§5, Ollama first), and **metrics**
on every run (§7).

**Is not.** Not a model router across providers (one backend per node; the kernel's proxy routes across providers), not a
gateway with billing (prices are the manager's), not a chat UI, not a vector store, not a fine-tuning service, and not a
replacement for Ollama (it never loads weights itself).

## 2. Who calls it

| Caller | Face | Key kind | Notes |
|---|---|---|---|
| The kernel's ledger proxy, for an agent on a local model | OpenAI or Anthropic | `inference` key held in the kernel's `model_registry` secret | the agent never sees the node |
| An application's engine (Knowledge's MaluDB answer model, Data Lake's steward) | OpenAI | `inference` key | through the kernel's proxy when it must be ledgered there; direct when the manager's usage is enough |
| A person's tool on the LAN (an IDE, a script, Open WebUI) | either | `inference` key, named for the person | the key's limits are the only wall |
| The manager | Node | `node` key, one per node, sealed in the manager | pulls status, runs and usage; mints inference keys; pushes policies |
| The machine's own operator | Node, through the CLI `sllm` | the local socket, no key | first setup, the node key shown once, a break-glass |

Every key is `sllm_` + 48 hex, shown once, only its SHA-256 kept (the estate's rule). An inference key carries: a name,
`max_priority` (1–9; 9 is highest, default cap 5), `max_concurrent` (jobs in flight at once; default 4), `cloud_allowed`
(may its jobs run in the cloud lane — false by default, §3), `fallback_allowed` (may a local job of its fall back — false by
default), `payload_retention` (`none` / `metadata` / `full`; default `metadata`), optional `allowed_models`, optional daily
token cap, `status`. **Keys are the one wall**: there is no user, no role, no session on the node.

## 3. Lanes — the heart of the design

A **lane** is a capacity. Every job belongs to exactly one lane at a time.

| Lane | Capacity | Holds | Why |
|---|---|---|---|
| `local` | `local_parallel` (default **1**) | inference on a local model; a **warm** (a load) | one model's worth of memory; a load beside a running inference thrashes |
| `cloud` | unbounded (an optional soft cap `cloud_parallel`, default none) | inference on a `-cloud` model | costs the machine nothing; R6 |
| `maintenance` | 1 | a pull, a delete | disk and network, never inference memory |

**Which lane a model is in** is the model's `lane` in the node's catalogue (§5.2): learned from Ollama (`-cloud` suffix,
`/api/show`'s remote flag) and overridable by the manager. A caller never names a lane; it names a model.

**The scheduler**, once per event (a job arrives, a job ends, a timer):

1. Take the lane with a free slot.
2. Among its `queued` jobs, pick by **priority descending**, then **fair share across keys** (round robin among keys at the
   same priority, so one busy key never starves another), then **arrival**.
3. In the `local` lane only, **model affinity**: if the model Ollama has loaded has a queued job within `affinity_window`
   (default 10 jobs or 30 s of the oldest waiting job's age, whichever bounds first), run it ahead of a different model's
   job — a reload costs seconds and the owner's Mac has room for one. Starvation is bounded by the window.
4. Hand the job to the backend, mark `running`, stream its chunks to the caller and into the job's store.

**Expected wait.** Every queued job carries `expected_wait_ms` = the sum of the expected durations of the jobs ahead of it
in its lane (a model's expected duration = the median `total_duration` of its last 20 runs, with a load added when the model
is not loaded; unknown model → the node's default). It is answered to the caller in a header on every wait and to the
manager on every poll.

**Fallback** (R6): a queued `local` job whose `expected_wait_ms` passes its **fallback threshold**, or whose model is
unavailable (the backend down, the model missing and not pullable, a load failure), is **moved** to the `cloud` lane on its
**fallback model**, provided every one of these is true: the key has `fallback_allowed`; the request did not say
`fallback: "none"`; the model's catalogue row names a `fallback_model` (or the request named one in `fallback.model`); and
the fallback model is in the cloud lane and allowed to the key. The threshold is the request's `fallback.after_seconds`, else
the model's `fallback_after_seconds`, else the node setting (default **60 s**). A job that has **started** never moves. The
answer carries `X-Node-Served-Model` and `X-Node-Fallback: local_wait|local_unavailable`, the run row keeps both models and
the reason, and the OpenAI/Anthropic `model` field names the model actually served (a client that trusts it is told the
truth). **Default is no fallback** — a prompt sent to a local model leaves the machine only when a person said it may, twice
(the key and the model or the request). The manager's screens make the one-click version of that.

**Timeouts.** `max_queue_seconds` per request (default none for async, the connection's own patience for sync); a job
running longer than `max_run_seconds` (default 1 h) is cancelled and the backend's request aborted. A cancel from the caller
or the manager aborts the backend stream and records `cancelled`.

## 4. Jobs, modes and streaming

A request on either face becomes one **job** row the moment it is accepted (status `queued`), with the request body stored
according to the key's `payload_retention`, and is answered according to the **mode**:

| Mode | How chosen | What the caller gets |
|---|---|---|
| `sync` (the default; the owner's "batch") | nothing to add, or `X-Node-Mode: sync` | **The connection waits.** Non-streaming: the complete answer in the face's shape, with the node headers. Streaming (`stream: true`): the face's own SSE, and while the job is still queued the node sends **SSE comments** every 10 s (`: queued position=3 expected_wait_ms=42000`) so proxies and clients keep the connection; the first real event is the face's first event. An `X-Node-Max-Wait: <seconds>` cap, when passed and reached, turns the answer into the async `202` below instead of a failure — nothing queued is lost |
| `async` | `X-Node-Mode: async`, or the body field `node: {"mode": "async"}` | **`202 Accepted`** at once with `{ "job_id", "status": "queued", "position", "expected_wait_ms", "status_url", "result_url", "events_url" }`; optionally `node.webhook_url` is POSTed the final result (signed `X-Node-Signature`, HMAC of the body with the key's webhook secret, three retries) |

Job routes on **both faces**, so an async caller never changes format:

- `GET …/jobs/{id}` — status, position, expected wait, timings; when `done`, the result in that face's shape (an OpenAI job
  answers a chat completion object, an Anthropic job a message).
- `GET …/jobs/{id}/events` — SSE: replays the chunks stored so far in the face's own event shape, then follows live; a
  client that lost its stream resumes here (`Last-Event-ID` honoured).
- `DELETE …/jobs/{id}` — cancel.
- `GET …/jobs?status=` — the key's own jobs.

A job is one of `queued → running → done | failed | cancelled | expired`, with `moved` recorded as an event, not a status.
Done jobs keep their result for `result_retention_hours` (default 72) and their metadata for ever (the manager has pulled
them long before). The same `node: {…}` body object (or `X-Node-*` headers, equivalent, for clients that cannot add body
fields) carries `priority`, `mode`, `fallback` (`"none"` or `{ "model", "after_seconds" }`), `max_wait_seconds`,
`max_queue_seconds`, `keep_alive`, `webhook_url`, `tags` (free strings the manager can group by, e.g. an application's
name), `idempotency_key` (a repeat within 24 h answers the first job). Unknown `node.*` fields are refused 400, never ignored.

**The faces are faithful.** The OpenAI face accepts the Chat Completions request (messages with text and image parts,
`tools`, `tool_choice`, `response_format: json_object` and `json_schema` → Ollama's `format`, `temperature`, `top_p`,
`max_tokens` / `max_completion_tokens` → `num_predict`, `stop`, `seed`, `stream_options.include_usage`) and answers the
object, the SSE chunks (`[DONE]` last) and the `usage` block with `prompt_tokens`, `completion_tokens`, `total_tokens`. The
Anthropic face accepts Messages (`system`, content blocks text / image / tool_use / tool_result, `tools`, `max_tokens`
required, `stop_sequences`, `metadata`) and answers the message and the event stream (`message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta` with `usage`, `message_stop`, with `ping`s) and
`usage.input_tokens` / `output_tokens`. Both faces answer the provider's own error shape with the provider's status codes
(400 bad request, 401 key, 403 model not allowed, 404 model unknown, 409 idempotency clash, 429 the key's cap, 503 backend
down with `Retry-After`). What a backend cannot do (an Anthropic `thinking` block, OpenAI `n > 1`, `logprobs`) is refused
400 by name, never silently dropped. The **node headers** on every answer: `X-Node-Id`, `X-Node-Job-Id`,
`X-Node-Lane`, `X-Node-Served-Model`, `X-Node-Fallback` (when it happened), `X-Node-Queue-Wait-Ms`,
`X-Node-Tokens-Per-Second`, and `X-Node-Usage: input=…;output=…` for clients that cannot read the body.

## 5. The backend and the models

### 5.1 The backend interface

One class per backend: `list_models()`, `show(model)`, `loaded()`, `chat(request) → stream of deltas + final stats`,
`embed(request)`, `pull(model) → progress stream`, `delete(model)`, `load(model, keep_alive)`, `unload(model)`, `health()`.
**Ollama is the first and only v1 backend** (`backends/ollama.py`, native API over `OLLAMA_HOST`, default
`http://127.0.0.1:11434`). llama.cpp's server, LM Studio and MLX are Extended on the same interface; a node runs one backend.

### 5.2 The catalogue

The node keeps a `models` table it refreshes from the backend every minute and on every change it makes: `name`, `lane`
(`local` / `cloud`), `family`, `parameter_size`, `quantization`, `context_length`, `capabilities` (chat / tools / vision /
embedding), `size_bytes`, `digest`, `on_disk`, `loaded` (+ `loaded_size_bytes`, `expires_at` from `/api/ps`), `status`
(`available` / `pulling` / `missing` / `disabled`), and the policies the manager sets: `keep_alive` (`5m` default; `-1` pinned
warm; `0` unload after every answer), `warm_schedule` (cron-ish, e.g. warm at 06:00 on weekdays), `fallback_model`,
`fallback_after_seconds`, `allowed` (a node-wide switch), `alias` (a caller may say `default` or `fast`; §12 S6).

### 5.3 Model operations

| Operation | Lane | How |
|---|---|---|
| **Pull** | `maintenance` | `POST /node/v1/models/pull {name}` → a job whose progress (`completed` / `total` bytes, the layer) streams on `/node/v1/jobs/{id}/events` and sits in `models.status = pulling`; a failed pull keeps the error |
| **Warm** | `local` (a job of kind `load`) | `POST /node/v1/models/{name}/warm {keep_alive?}` → an empty chat with `keep_alive`; takes a local slot for the load's duration, so it waits behind a running inference like any job; also by `warm_schedule` |
| **Unload** | none (immediate) | `POST /node/v1/models/{name}/unload` → `keep_alive: 0`; a running job on that model is not interrupted — the unload is applied when the slot frees |
| **Keep warm** | — | `keep_alive` on every request of that model, `-1` pins it |
| **Delete** | `maintenance` | `DELETE /node/v1/models/{name}`; refused while a job of that model is queued or running |
| **Disable** | — | `allowed = false` → 404 to callers, the queue drained to fallback or failed |

The node never pulls a model a caller names and it does not have ("auto-pull" is a manager policy per node, off by default,
S7): an unknown model is 404 and the manager sees the miss in the run row (`status = failed`, `error = model_missing`) and can
offer the pull.

## 6. The node API — the contract with the manager

Prefix `/node/v1`, bearer = the **node key**. Every answer carries `node_id` (a UUID made at first start, also in
`X-Node-Id`), `node_name`, `api_version` (`1`) and `clock` (the node's UTC time, so the manager can see skew). Lists are
cursor-paged (`?after=<cursor>&limit=`), newest-last, and **every mutable row carries `seq`**, a node-wide monotonic
sequence stamped on every insert and update, so the manager syncs with `?since_seq=` and misses nothing. All times are UTC
ISO-8601 with milliseconds.

### 6.1 Status and health

| Route | Answer |
|---|---|
| `GET /status` | the node: name, version, started_at, backend (kind, version, reachable, `OLLAMA_HOST`), machine (hostname, os, cpu, memory total/free, gpu or Apple silicon unified memory, disk free at the model store), lanes (each: capacity, running, queued, oldest queued age, expected drain ms), loaded models, counters since start (jobs by status, tokens in/out), the policy versions it holds (§6.6), `last_manager_seen_at` |
| `GET /health` | `200 {"ok": true}` when the backend answers, `503` otherwise; no key needed (for load balancers and the manager's cheap probe) |
| `GET /events?since_seq=` | SSE of node events (job state changes, model changes, lane changes, warnings) for a manager that prefers a stream to polling; the same rows as `/runs` and `/models`, so a lost stream is recovered by `since_seq` |

### 6.2 Queue

| Route | Answer |
|---|---|
| `GET /queue` | every `queued` and `running` job across lanes: job id, lane, model (requested, served), key name, priority, mode, face, position, queued_at, started_at, expected_wait_ms, tokens so far, tags |
| `POST /queue/{job_id}/priority {priority}` | re-prioritise (the manager's Operator) |
| `POST /queue/{job_id}/move {lane, model}` | force a fallback now, or move a job back (only while queued) |
| `DELETE /queue/{job_id}` | cancel; a running job's backend call is aborted |
| `POST /queue/pause {lane}` / `/queue/resume {lane}` | stop dispatching a lane (running jobs finish); the node answers sync callers with SSE comments meanwhile and async ones with `queued` |

### 6.3 Runs and metrics

`GET /runs?since_seq=&status=&model=&lane=&key=&from=&to=` — one row per job, the **run record** of §7, metadata only.
`GET /runs/{id}` adds the stored request and response when the key's retention kept them (`full`), the backend's raw final
stats, the event timeline (queued, moved, started, first token, finished), and the error text. `GET /runs/{id}/payload`
answers the request and response bodies alone (the manager stores them in its own retention). The manager's sync is one
loop: `since_seq` from the last row it holds, until the page is short.

### 6.4 Usage

`GET /usage?from=&to=&group_by=key|model|lane|tag|day|hour` — tokens in / out / total, runs, failures, fallbacks, queue
wait p50/p95, duration p50/p95, tokens per second p50 — computed from `runs`. The manager keeps its own usage tables and
uses this only for a check; a client with nothing but a node key can still answer "what ran here last month".

### 6.5 Models

`GET /models` (the catalogue, §5.2, with `since_seq`), `GET /models/{name}`, `POST /models/pull`, `DELETE /models/{name}`,
`POST /models/{name}/warm`, `POST /models/{name}/unload`, `PATCH /models/{name}` (the policies: `keep_alive`,
`warm_schedule`, `fallback_model`, `fallback_after_seconds`, `allowed`, `alias`, `lane` override), `GET /models/pullable?q=`
(Ollama's library search, when reachable; Extended).

### 6.6 Keys, policies and settings

- `GET /keys`, `POST /keys` (name + limits → the key shown **once** in the answer and never again; the manager seals it or
  hands it to the person), `PATCH /keys/{id}` (limits, status), `DELETE /keys/{id}` (revoke; in-flight jobs finish).
- `GET /settings`, `PATCH /settings`: `local_parallel`, `cloud_parallel`, `affinity_window`, `fallback_after_seconds`,
  `max_run_seconds`, `result_retention_hours`, `payload_retention_days`, `auto_pull`, `default_model`, `keep_alive_default`,
  `backend.url`, `cloud_route` (`daemon` / `direct`), `name`, `timezone` (for schedules only; storage is UTC).
- **Policy versions.** Every `PATCH` bumps a `policy_version` the status reports; the manager compares and re-pushes what
  drifted (a node restored from a backup, an operator's CLI change). The manager is the source of a policy once it has set
  one; until then the node's own value stands.
- `POST /node-key/rotate` — a new node key shown once; the old one dies at the next call (the manager does this from its
  node page, holding the new key before confirming).

### 6.7 What the API never does

Serve an inference (that is the two faces, with inference keys), read a key back, return a payload a key's retention did not
keep, or accept a request from a manager it has not been given a node key by. There is no "push" from the node to the
manager in v1 (the node knows no manager address); the Extended **registration mode** (the node dials a manager URL with a
one-time enrolment code, for nodes behind NAT) is S10.

## 7. The run record — what is measured

One row per job in `runs` (SQLite, WAL mode, one file `data/node.sqlite3`; `payloads` in a second file so the metadata
stays small and the payload file can be purged by retention):

| Group | Columns |
|---|---|
| Identity | `id` (UUID), `seq`, `node_id`, `face` (openai / anthropic), `mode`, `kind` (chat / embed / load / pull / delete), `key_id`, `key_name`, `tags`, `idempotency_key`, `client_ip`, `user_agent` |
| Model | `requested_model`, `served_model`, `lane_requested`, `lane_served`, `fallback_reason` (null / `local_wait` / `local_unavailable` / `forced`), `priority`, `stream` |
| Times | `accepted_at`, `queued_until` (= `started_at`), `started_at`, `first_token_at`, `finished_at`; derived `queue_wait_ms`, `time_to_first_token_ms`, `run_ms`, `total_ms` |
| Backend stats | `load_ms`, `prompt_tokens`, `prompt_eval_ms`, `output_tokens`, `eval_ms`, `total_backend_ms` — straight from Ollama's final message; derived `prompt_tokens_per_second`, `output_tokens_per_second`, `model_was_loaded` (load_ms below 50 ms) |
| Size | `request_bytes`, `response_bytes`, `messages_count`, `images_count`, `tools_count`, `context_length` (the model's), `context_used` (prompt + output) |
| Outcome | `status`, `stop_reason` (`stop` / `length` / `tool_calls` / `cancelled` / `error`), `error_code`, `error_text`, `http_status` |
| Machine | `memory_free_at_start_bytes`, `loaded_models_at_start` (names), `cpu_load_1m` — so a slow run can be explained by what else was on the box |

Embeddings record `prompt_tokens` and `vectors`; loads and pulls record their durations and bytes. The `events` table
(`run_id`, `at`, `kind`, `detail`) keeps the timeline; the `payloads` table (`run_id`, `request_json`, `response_json`,
`chunks` for the replay) is written only when retention says so and purged after `payload_retention_days` (default 30).

**Token counts are the backend's**, never estimated: Ollama's `prompt_eval_count` and `eval_count` are the usage every face
answers. When Ollama serves a prompt from its cache it reports fewer prompt tokens; the node records what it reports and
`X-Node-Cache: hit` when the count is below the request's own estimate by more than half (a hint, not an accounting).

## 8. Process, files and operation on the machine

- **Python 3.12, FastAPI + uvicorn, httpx (streaming), aiosqlite, apscheduler** for the warm schedules; one process, one
  asyncio loop; the scheduler is a function run on events; no Redis, no Celery, no Docker required.
- Layout: `server/` (`app.py`, `faces/openai.py`, `faces/anthropic.py`, `node_api.py`, `scheduler.py`, `lanes.py`,
  `jobs.py`, `store.py`, `backends/ollama.py`, `translate/` (the two wire translations, each proven against recorded
  fixtures), `metrics.py`, `keys.py`, `cli.py`), `data/` (the SQLite files, mode 600), `config/node.toml` (the port, the
  backend URL, the name; secrets never — keys are hashes in the database), `deploy/` (a **launchd** plist for macOS, a
  systemd unit for Linux, both running as a dedicated user, bound to all interfaces or to Tailscale's only — S8), `tests/`.
- **The CLI `sllm`** (same code, local socket): `init` (the node id, the first node key shown once, the config), `status`,
  `queue`, `keys mint|revoke|list`, `models pull|warm|unload|list`, `serve`, `prove`. It is how the owner sets up the Mac
  before the manager knows the node, and the break-glass when the manager is away.
- **TLS** is not the node's: on a LAN or over Tailscale the node speaks HTTP; in front of the public it sits behind the
  owner's proxy. The node refuses to bind a non-loopback address with no key configured.
- **Logs** to stdout (launchd/systemd capture), structured JSON lines, never a payload or a key.
- **Backups**: the SQLite files, copied by `sllm backup` with the WAL checkpointed; a node restored from a backup replays its
  policies from the manager at the next sync (§6.6).

## 9. Proofs

`tests/run.sh` — **without Ollama**, against a fake backend that answers recorded Ollama streams with timings:

- the OpenAI face: a non-streaming chat, a streaming chat with `[DONE]`, tools, images, json mode, embeddings, `models`, every
  refusal code, the node headers, the usage block equals the fake's counts;
- the Anthropic face: a message, the event stream in order with `ping`s, tool_use → tool_result round trip, system, the
  usage, `max_tokens` required;
- lanes: the local lane runs one at a time while three cloud jobs run together; priority, fair share across keys,
  affinity within the window and not beyond it; pause/resume;
- modes: sync waits with comments every 10 s; `X-Node-Max-Wait` turns into 202; async 202 → poll → events replay →
  `Last-Event-ID` resume; webhook signed and retried; idempotency;
- fallback: moves only when the key, the request and the model all allow; never after start; the served model named; the
  reason recorded; `local_unavailable` when the backend is down;
- models: pull progress, warm takes a local slot, unload waits for the slot, delete refused while queued, disable drains;
- the node API: `since_seq` sync misses nothing across 1,000 jobs under concurrent writes; a rotated node key kills the old;
  keys shown once; retention `none` stores no payload; payload purge;
- the run record: every column filled from the fake's timings; usage grouping equals a SQL sum;
- 10,000 queued jobs scheduled in under a second a dispatch; the SQLite file after 100,000 runs under 200 MB.

`tests/live.sh` — **with Ollama** on the owner's Mac: one small local model (`qwen3:0.6b` or whatever is present) and one
cloud model: a local job and a cloud job overlap; a warm shortens the next load; the fallback fires with a 5-second threshold
and two queued local jobs; the numbers in `runs` match Ollama's own final message.

## 10. Build order

1. **Step 0 — this document approved** with the manager's; the two repositories created in the org by the owner
   (`maluos-local-llm-server` public; §12 S12).
2. **Step 1 — the store and the scheduler** (`store.py`, `lanes.py`, `scheduler.py`, `jobs.py`, keys): lanes, priority,
   fair share, affinity, fallback, the run record, `since_seq`; proven against the fake backend.
3. **Step 2 — the Ollama backend and the OpenAI face** (`backends/ollama.py`, `translate/openai.py`, sync and async, streaming,
   the headers, usage); the live proof on the Mac.
4. **Step 3 — the Anthropic face** (`translate/anthropic.py`, the event stream, tools and blocks).
5. **Step 4 — models** (pull, warm, unload, keep, schedules, the catalogue refresh).
6. **Step 5 — the node API** (§6 complete), the CLI, `deploy/`, the backups; `tests/run.sh` whole.
7. **Step 6 — the manager's Phase 3 slice 1 reads it** (the manager's design §10) — the end-to-end proof: the Mac's queue on
   a screen in the estate.

Steps 1–3 are the planning model's (the scheduler and the translations are the exemplars); steps 4–5 are a Sonnet 5.5
worker's from this document, one at a time, each verified.

## 11. Extended and later

Registration mode for nodes behind NAT (S10); more backends (llama.cpp, LM Studio, MLX, vLLM); `/v1/completions` and
`count_tokens`; the OpenAI Responses API; Anthropic's batch API shape (`/v1/messages/batches`) as a view of async jobs;
speculative "shadow" runs (the same prompt to a cloud model for a quality comparison the manager grades); a model's
expected-duration curve by prompt length instead of a median; multiple local lanes on a machine with two GPUs; prompt
caching across jobs on the node (Ollama's own, surfaced); a per-key monthly budget in the manager's prices.

## 12. The decisions (recommendations; the owner's answers go in §13)

R1–R7 (§0.1) are decided and not asked again.

| # | Decision | Recommendation | Alternative |
|---|---|---|---|
| S1 | Name and port | repository **`maluos-local-llm-server`** as the owner named it, the binary and service `sllm`, port **11435** (Ollama's plus one), the SQLite store `data/node.sqlite3` | `maludb-os-llm-node` to match the estate's naming; port 8080 |
| S2 | Stack | **Python 3.12, FastAPI + uvicorn, httpx, aiosqlite**, no queue broker; launchd on macOS, systemd on Linux | Go for a single binary (no Python on the Mac) — slower to build, harder for workers |
| S3 | Default mode | **sync** with SSE keep-alive comments while queued, and `X-Node-Max-Wait` turning a long wait into async | async by default (every OpenAI client would then break) |
| S4 | Fallback default | **off**, requiring the key's `fallback_allowed` AND a fallback model named on the model or the request; threshold 60 s of expected wait or local unavailable | on by default for any key with `cloud_allowed` |
| S5 | Local capacity | **1** per node by default (`local_parallel`), the scheduler never over-subscribing Ollama; the manager raises it on a machine that can | read `OLLAMA_NUM_PARALLEL` and match it |
| S6 | Aliases | **`default`** and per-model `alias`es the manager sets (a caller says `fast` or `smart`; the node resolves it, the run keeps both) | callers must name the exact model |
| S7 | Auto-pull | **off** — an unknown model is 404 and the manager offers the pull | on, so a caller's new model name pulls itself (surprise disk use) |
| S8 | Bind address | **all interfaces, keys required**, with `bind` settable to Tailscale's address alone; refuse to start non-loopback with no key | loopback only, with the proxy's doing |
| S9 | Cloud route | **through the local Ollama daemon** (one signed-in account) with `cloud_route = direct` (`OLLAMA_API_KEY` to ollama.com) as a switch | direct only |
| S10 | Reachability | **v1 pull only** — the manager dials the node; registration mode (the node dials in) Extended | registration in v1 |
| S11 | Payload retention | key default **`metadata`**; `full` for the kernel's keys (the proxy keeps its own copy anyway) and for anyone debugging; purge at 30 days | `full` by default (the box is sovereign) |
| S12 | Repository | **public** under `github.com/maludb`, Apache-2.0 like the kernel; nothing in it is a business's | private until it has run a month |

## 13. The owner's answers

2026-10-17 — **S1 the repository `maluos-local-llm-server`** (the owner's name kept), **S2–S11 the recommendations**, **S12 public**.
Every decision in §12 stands as recommended. The design is approved.

## 14. State

2026-10-16 — designed. 2026-10-17 — **APPROVED by the owner** (§13). Nothing built. Local repository `~/maluos-local-llm-server`; the
GitHub repository `github.com/maludb/maluos-local-llm-server` (public) is the owner's to create before step 1 (§10).
