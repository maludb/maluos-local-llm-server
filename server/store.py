"""The node's own memory: SQLite (WAL) — `node.sqlite3` holds keys, runs, events, models and settings; `payloads.sqlite3` holds request
and response bodies (written only when a key's retention says so, purged by age) so the metadata file stays small (design §7).
Every mutable row carries `seq`, a node-wide monotonic sequence stamped on every insert and update, so the manager syncs with
`since_seq` and misses nothing (§6). One process writes; a single asyncio lock serialises writers."""
from __future__ import annotations

import asyncio
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from .config import SETTING_DEFAULTS, coerce_setting

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS keys (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('inference', 'node')),
    name TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    max_priority INTEGER NOT NULL DEFAULT 5,
    max_concurrent INTEGER NOT NULL DEFAULT 4,
    cloud_allowed INTEGER NOT NULL DEFAULT 0,
    fallback_allowed INTEGER NOT NULL DEFAULT 0,
    payload_retention TEXT NOT NULL DEFAULT 'metadata' CHECK (payload_retention IN ('none', 'metadata', 'full')),
    allowed_models TEXT,                      -- JSON list or NULL = any
    daily_token_cap INTEGER,
    webhook_secret TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'revoked')),
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    face TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'sync',
    kind TEXT NOT NULL DEFAULT 'chat' CHECK (kind IN ('chat', 'embed', 'load', 'pull', 'delete')),
    key_id TEXT,
    key_name TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    idempotency_key TEXT,
    client_ip TEXT,
    user_agent TEXT,
    requested_model TEXT,
    served_model TEXT,
    lane_requested TEXT,
    lane_served TEXT,
    fallback_reason TEXT,
    fallback_model TEXT,
    fallback_after_seconds INTEGER,
    priority INTEGER NOT NULL DEFAULT 5,
    stream INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled', 'expired')),
    accepted_at TEXT NOT NULL,
    started_at TEXT,
    first_token_at TEXT,
    finished_at TEXT,
    queue_wait_ms INTEGER,
    time_to_first_token_ms INTEGER,
    run_ms INTEGER,
    total_ms INTEGER,
    load_ms INTEGER,
    prompt_tokens INTEGER,
    prompt_eval_ms INTEGER,
    output_tokens INTEGER,
    eval_ms INTEGER,
    total_backend_ms INTEGER,
    prompt_tokens_per_second REAL,
    output_tokens_per_second REAL,
    model_was_loaded INTEGER,
    request_bytes INTEGER,
    response_bytes INTEGER,
    messages_count INTEGER,
    images_count INTEGER,
    tools_count INTEGER,
    context_length INTEGER,
    context_used INTEGER,
    stop_reason TEXT,
    error_code TEXT,
    error_text TEXT,
    http_status INTEGER,
    memory_free_at_start_bytes INTEGER,
    loaded_models_at_start TEXT,
    cpu_load_1m REAL,
    max_queue_seconds INTEGER,
    max_run_seconds INTEGER,
    keep_alive TEXT,
    webhook_url TEXT,
    webhook_status TEXT,
    result_expires_at TEXT
);
CREATE INDEX IF NOT EXISTS runs_seq ON runs (seq);
CREATE INDEX IF NOT EXISTS runs_status ON runs (status, lane_served, priority, accepted_at);
CREATE INDEX IF NOT EXISTS runs_key ON runs (key_id, accepted_at);
CREATE INDEX IF NOT EXISTS runs_key_status ON runs (key_id, status);
CREATE INDEX IF NOT EXISTS runs_fallback ON runs (lane_served, status) WHERE fallback_model IS NOT NULL;
CREATE INDEX IF NOT EXISTS runs_expiry ON runs (status) WHERE max_queue_seconds IS NOT NULL;
CREATE INDEX IF NOT EXISTS runs_model ON runs (served_model, status, finished_at);
CREATE INDEX IF NOT EXISTS runs_idem ON runs (key_id, idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq INTEGER NOT NULL,
    run_id TEXT,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS events_run ON events (run_id, id);
CREATE INDEX IF NOT EXISTS events_seq ON events (seq);

CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    lane TEXT NOT NULL DEFAULT 'local' CHECK (lane IN ('local', 'cloud')),
    lane_override TEXT,
    family TEXT, parameter_size TEXT, quantization TEXT, context_length INTEGER,
    capabilities TEXT NOT NULL DEFAULT '[]',
    size_bytes INTEGER, digest TEXT,
    on_disk INTEGER NOT NULL DEFAULT 0,
    loaded INTEGER NOT NULL DEFAULT 0, loaded_size_bytes INTEGER, expires_at TEXT,
    status TEXT NOT NULL DEFAULT 'available' CHECK (status IN ('available', 'pulling', 'missing', 'disabled')),
    keep_alive TEXT, warm_schedule TEXT, fallback_model TEXT, fallback_after_seconds INTEGER,
    allowed INTEGER NOT NULL DEFAULT 1,
    alias TEXT,
    pull_error TEXT,
    first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS models_alias ON models (alias) WHERE alias IS NOT NULL;

CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL, seq INTEGER NOT NULL);
"""

PAYLOAD_SCHEMA = """
CREATE TABLE IF NOT EXISTS payloads (
    run_id TEXT PRIMARY KEY,
    stored_at TEXT NOT NULL,
    retention TEXT NOT NULL DEFAULT 'metadata',
    request_json TEXT,
    response_json TEXT,
    chunks_json TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def _row(r: aiosqlite.Row | None) -> dict | None:
    return dict(r) if r is not None else None


class Store:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.db: aiosqlite.Connection | None = None
        self.pdb: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._seq = 0
        self.node_id = ""

    async def open(self) -> "Store":
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = await aiosqlite.connect(self.data_dir / "node.sqlite3")
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;" + SCHEMA)
        self.pdb = await aiosqlite.connect(self.data_dir / "payloads.sqlite3")
        self.pdb.row_factory = aiosqlite.Row
        await self.pdb.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;" + PAYLOAD_SCHEMA)
        cur = await self.db.execute("SELECT value FROM meta WHERE key = 'seq'")
        r = await cur.fetchone()
        self._seq = int(r["value"]) if r else 0
        cur = await self.db.execute("SELECT value FROM meta WHERE key = 'node_id'")
        r = await cur.fetchone()
        if r:
            self.node_id = r["value"]
        else:
            self.node_id = str(uuid.uuid4())
            await self.db.execute("INSERT INTO meta (key, value) VALUES ('node_id', ?), ('created_at', ?)", (self.node_id, now_iso()))
            await self.db.commit()
        for p in (self.data_dir / "node.sqlite3", self.data_dir / "payloads.sqlite3"):
            try:
                p.chmod(0o600)
            except OSError:
                pass
        return self

    async def close(self) -> None:
        if self.db:
            await self.db.close()
        if self.pdb:
            await self.pdb.close()

    # -- the sequence ---------------------------------------------------------------------------------------------------------
    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @property
    def seq(self) -> int:
        return self._seq

    async def _commit_seq(self) -> None:
        await self.db.execute("INSERT INTO meta (key, value) VALUES ('seq', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                              (str(self._seq),))

    async def write(self, sql: str, params: Iterable = ()) -> int:
        """One stamped write: assigns the next seq (the SQL names ? for it last), commits, returns the seq."""
        async with self._lock:
            seq = self._next_seq()
            await self.db.execute(sql, (*params, seq))
            await self._commit_seq()
            await self.db.commit()
            return seq

    async def write_many(self, statements: list[tuple[str, tuple]]) -> int:
        async with self._lock:
            seq = self._next_seq()
            for sql, params in statements:
                await self.db.execute(sql, (*params, seq))
            await self._commit_seq()
            await self.db.commit()
            return seq

    async def fetchone(self, sql: str, params: Iterable = ()) -> dict | None:
        cur = await self.db.execute(sql, tuple(params))
        return _row(await cur.fetchone())

    async def fetchall(self, sql: str, params: Iterable = ()) -> list[dict]:
        cur = await self.db.execute(sql, tuple(params))
        return [dict(r) for r in await cur.fetchall()]

    # -- settings ---------------------------------------------------------------------------------------------------------------
    async def settings(self) -> dict[str, Any]:
        out = dict(SETTING_DEFAULTS)
        for r in await self.fetchall("SELECT key, value FROM settings"):
            if r["key"] in SETTING_DEFAULTS:
                out[r["key"]] = coerce_setting(r["key"], json.loads(r["value"]))
        return out

    async def setting(self, key: str) -> Any:
        return (await self.settings())[key]

    async def set_settings(self, changes: dict[str, Any]) -> int:
        stmts = []
        for k, v in changes.items():
            v = coerce_setting(k, v)
            stmts.append(("INSERT INTO settings (key, value, seq) VALUES (?, ?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value, seq = excluded.seq",
                          (k, json.dumps(v))))
        if not stmts:
            return self._seq
        seq = await self.write_many(stmts)
        await self.bump_policy_version()
        return seq

    async def policy_version(self) -> int:
        r = await self.fetchone("SELECT value FROM meta WHERE key = 'policy_version'")
        return int(r["value"]) if r else 0

    async def bump_policy_version(self) -> int:
        v = await self.policy_version() + 1
        await self.db.execute("INSERT INTO meta (key, value) VALUES ('policy_version', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", (str(v),))
        await self.db.commit()
        return v

    async def meta(self, key: str) -> str | None:
        r = await self.fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return r["value"] if r else None

    async def set_meta(self, key: str, value: str) -> None:
        await self.db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))
        await self.db.commit()

    # -- runs ---------------------------------------------------------------------------------------------------------------
    async def insert_run(self, run: dict) -> int:
        cols = [c for c in run.keys() if c != "seq"]
        sql = f"INSERT INTO runs ({', '.join(cols)}, seq) VALUES ({', '.join('?' * len(cols))}, ?)"
        return await self.write(sql, [run[c] for c in cols])

    async def update_run(self, run_id: str, changes: dict, event: tuple[str, dict] | None = None) -> int:
        """One stamped update, with its timeline event in the same transaction when given."""
        sets = ", ".join(f"{k} = ?" for k in changes)
        async with self._lock:
            seq = self._next_seq()
            await self.db.execute(f"UPDATE runs SET {sets}, seq = ? WHERE id = ?", (*changes.values(), seq, run_id))
            if event:
                await self.db.execute("INSERT INTO events (run_id, at, kind, detail, seq) VALUES (?, ?, ?, ?, ?)",
                                      (run_id, now_iso(), event[0], json.dumps(event[1]), seq))
            await self._commit_seq()
            await self.db.commit()
            return seq

    async def add_event(self, run_id: str | None, kind: str, detail: dict | None = None) -> int:
        return await self.write("INSERT INTO events (run_id, at, kind, detail, seq) VALUES (?, ?, ?, ?, ?)",
                                (run_id, now_iso(), kind, json.dumps(detail or {})))

    async def run(self, run_id: str) -> dict | None:
        return await self.fetchone("SELECT * FROM runs WHERE id = ?", (run_id,))

    async def run_events(self, run_id: str) -> list[dict]:
        rows = await self.fetchall("SELECT id, at, kind, detail FROM events WHERE run_id = ? ORDER BY id", (run_id,))
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows

    async def queued(self, lane: str | None = None) -> list[dict]:
        if lane:
            return await self.fetchall("SELECT * FROM runs WHERE status = 'queued' AND lane_served = ? ORDER BY priority DESC, accepted_at, seq", (lane,))
        return await self.fetchall("SELECT * FROM runs WHERE status = 'queued' ORDER BY priority DESC, accepted_at, seq")

    async def running(self, lane: str | None = None) -> list[dict]:
        if lane:
            return await self.fetchall("SELECT * FROM runs WHERE status = 'running' AND lane_served = ? ORDER BY started_at", (lane,))
        return await self.fetchall("SELECT * FROM runs WHERE status = 'running' ORDER BY started_at")

    async def in_flight_for_key(self, key_id: str) -> int:
        r = await self.fetchone("SELECT count(*) AS n FROM runs WHERE key_id = ? AND status IN ('queued', 'running')", (key_id,))
        return int(r["n"])

    async def find_idempotent(self, key_id: str, idem: str) -> dict | None:
        return await self.fetchone("SELECT * FROM runs WHERE key_id = ? AND idempotency_key = ? AND accepted_at > ? ORDER BY accepted_at DESC LIMIT 1",
                                   (key_id, idem, datetime.fromtimestamp(time.time() - 86400, timezone.utc).isoformat()))

    async def expected_ms(self, model: str, default_ms: int, default_load_ms: int) -> tuple[int, int]:
        """(median total_ms of the model's last 20 done runs, median load_ms of those that loaded) — the expected duration (§3)."""
        rows = await self.fetchall("SELECT total_backend_ms AS t, load_ms AS l, model_was_loaded AS w FROM runs WHERE served_model = ? AND status = 'done' AND kind = 'chat' "
                                   "AND total_backend_ms IS NOT NULL ORDER BY finished_at DESC LIMIT 20", (model,))
        if not rows:
            return default_ms, default_load_ms
        ts = sorted(int(r["t"]) - int(r["l"] or 0) for r in rows)
        ls = sorted(int(r["l"]) for r in rows if r["l"] and not r["w"]) or [default_load_ms]
        return ts[len(ts) // 2], ls[len(ls) // 2]

    async def runs_since(self, since_seq: int, limit: int = 500, **filters) -> list[dict]:
        where = ["seq > ?"]
        params: list = [since_seq]
        for k in ("status", "served_model", "lane_served", "key_name", "kind"):
            if filters.get(k):
                where.append(f"{k} = ?")
                params.append(filters[k])
        if filters.get("from_"):
            where.append("accepted_at >= ?"); params.append(filters["from_"])
        if filters.get("to"):
            where.append("accepted_at < ?"); params.append(filters["to"])
        params.append(limit)
        return await self.fetchall(f"SELECT * FROM runs WHERE {' AND '.join(where)} ORDER BY seq LIMIT ?", params)

    async def events_since(self, since_seq: int, limit: int = 500) -> list[dict]:
        rows = await self.fetchall("SELECT * FROM events WHERE seq > ? ORDER BY seq, id LIMIT ?", (since_seq, limit))
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows

    # -- payloads -------------------------------------------------------------------------------------------------------------
    async def store_payload(self, run_id: str, retention: str = "metadata", request: Any = None, response: Any = None, chunks: Any = None) -> None:
        """The RESULT (response + chunks) is kept for every job until result_retention_hours so an async caller can fetch it; the
        REQUEST only under `full`; `purge_results` clears the result unless the retention is `full` (then payload_retention_days)."""
        await self.pdb.execute(
            "INSERT INTO payloads (run_id, stored_at, retention, request_json, response_json, chunks_json) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id) DO UPDATE SET response_json = COALESCE(excluded.response_json, payloads.response_json), "
            "chunks_json = COALESCE(excluded.chunks_json, payloads.chunks_json), request_json = COALESCE(excluded.request_json, payloads.request_json)",
            (run_id, now_iso(), retention, json.dumps(request) if (request is not None and retention == "full") else None,
             json.dumps(response) if response is not None else None, json.dumps(chunks) if chunks is not None else None))
        await self.pdb.commit()

    async def payload(self, run_id: str) -> dict | None:
        cur = await self.pdb.execute("SELECT * FROM payloads WHERE run_id = ?", (run_id,))
        r = _row(await cur.fetchone())
        if not r:
            return None
        return {"run_id": run_id, "stored_at": r["stored_at"],
                "request": json.loads(r["request_json"]) if r["request_json"] else None,
                "response": json.loads(r["response_json"]) if r["response_json"] else None,
                "chunks": json.loads(r["chunks_json"]) if r["chunks_json"] else None}

    async def purge_payloads(self, older_than_days: int) -> int:
        cutoff = datetime.fromtimestamp(time.time() - older_than_days * 86400, timezone.utc).isoformat()
        cur = await self.pdb.execute("DELETE FROM payloads WHERE stored_at < ?", (cutoff,))
        await self.pdb.commit()
        return cur.rowcount

    async def purge_results(self, older_than_hours: int) -> int:
        """Done jobs lose their stored result after result_retention_hours (metadata stays for ever)."""
        cutoff = datetime.fromtimestamp(time.time() - older_than_hours * 3600, timezone.utc).isoformat()
        cur = await self.pdb.execute("DELETE FROM payloads WHERE stored_at < ? AND retention <> 'full'", (cutoff,))
        await self.pdb.commit()
        return cur.rowcount

    # -- models -----------------------------------------------------------------------------------------------------------------
    async def model(self, name: str) -> dict | None:
        r = await self.fetchone("SELECT * FROM models WHERE name = ? OR alias = ?", (name, name))
        if r:
            r["capabilities"] = json.loads(r["capabilities"])
        return r

    async def models(self) -> list[dict]:
        rows = await self.fetchall("SELECT * FROM models ORDER BY name")
        for r in rows:
            r["capabilities"] = json.loads(r["capabilities"])
        return rows

    async def upsert_model(self, name: str, **fields) -> int:
        if "capabilities" in fields and not isinstance(fields["capabilities"], str):
            fields["capabilities"] = json.dumps(fields["capabilities"])
        existing = await self.fetchone("SELECT name FROM models WHERE name = ?", (name,))
        ts = now_iso()
        if existing:
            if not fields:
                return self._seq
            sets = ", ".join(f"{k} = ?" for k in fields)
            async with self._lock:
                seq = self._next_seq()
                await self.db.execute(f"UPDATE models SET {sets}, updated_at = ?, seq = ? WHERE name = ?", (*fields.values(), ts, seq, name))
                await self._commit_seq(); await self.db.commit()
                return seq
        cols = ["name", "first_seen_at", "updated_at", *fields.keys()]
        return await self.write(f"INSERT INTO models ({', '.join(cols)}, seq) VALUES ({', '.join('?' * len(cols))}, ?)", [name, ts, ts, *fields.values()])

    async def delete_model(self, name: str) -> int:
        """A deleted model leaves the catalogue; the seq is bumped (the manager sees it gone on its next /models)."""
        async with self._lock:
            seq = self._next_seq()
            await self.db.execute("DELETE FROM models WHERE name = ?", (name,))
            await self._commit_seq(); await self.db.commit()
            return seq

    # -- keys -------------------------------------------------------------------------------------------------------------------
    async def key_by_hash(self, h: str) -> dict | None:
        r = await self.fetchone("SELECT * FROM keys WHERE hash = ?", (h,))
        if r and r["allowed_models"]:
            r["allowed_models"] = json.loads(r["allowed_models"])
        return r

    async def key(self, key_id: str) -> dict | None:
        r = await self.fetchone("SELECT * FROM keys WHERE id = ?", (key_id,))
        if r and r["allowed_models"]:
            r["allowed_models"] = json.loads(r["allowed_models"])
        return r

    async def keys(self, kind: str | None = None) -> list[dict]:
        rows = await self.fetchall("SELECT * FROM keys WHERE (? IS NULL OR kind = ?) ORDER BY created_at", (kind, kind))
        for r in rows:
            if r["allowed_models"]:
                r["allowed_models"] = json.loads(r["allowed_models"])
        return rows

    async def insert_key(self, row: dict) -> int:
        if row.get("allowed_models") is not None and not isinstance(row["allowed_models"], str):
            row["allowed_models"] = json.dumps(row["allowed_models"])
        cols = list(row.keys())
        return await self.write(f"INSERT INTO keys ({', '.join(cols)}, seq) VALUES ({', '.join('?' * len(cols))}, ?)", [row[c] for c in cols])

    async def update_key(self, key_id: str, changes: dict) -> int:
        if "allowed_models" in changes and changes["allowed_models"] is not None and not isinstance(changes["allowed_models"], str):
            changes["allowed_models"] = json.dumps(changes["allowed_models"])
        sets = ", ".join(f"{k} = ?" for k in changes)
        async with self._lock:
            seq = self._next_seq()
            await self.db.execute(f"UPDATE keys SET {sets}, seq = ? WHERE id = ?", (*changes.values(), seq, key_id))
            await self._commit_seq(); await self.db.commit()
            return seq

    async def touch_key(self, key_id: str) -> None:
        """last_used_at at most once a minute, with no seq (a stamp, not a change the manager syncs)."""
        r = await self.fetchone("SELECT last_used_at FROM keys WHERE id = ?", (key_id,))
        if r and r["last_used_at"] and time.time() - (parse_iso(r["last_used_at"]) or 0) < 60:
            return
        await self.db.execute("UPDATE keys SET last_used_at = ? WHERE id = ?", (now_iso(), key_id))
        await self.db.commit()

    async def tokens_today(self, key_id: str) -> int:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
        r = await self.fetchone("SELECT COALESCE(SUM(COALESCE(prompt_tokens,0) + COALESCE(output_tokens,0)), 0) AS n FROM runs WHERE key_id = ? AND accepted_at >= ?", (key_id, day))
        return int(r["n"])

    # -- usage ------------------------------------------------------------------------------------------------------------------
    async def usage(self, from_: str | None, to: str | None, group_by: str = "key", key_id: str | None = None) -> list[dict]:
        groups = {"key": "key_name", "model": "served_model", "lane": "lane_served", "day": "substr(accepted_at, 1, 10)",
                  "hour": "substr(accepted_at, 1, 13)", "tag": "tags"}
        g = groups.get(group_by, "key_name")
        where, params = ["kind IN ('chat','embed')"], []
        if from_:
            where.append("accepted_at >= ?"); params.append(from_)
        if to:
            where.append("accepted_at < ?"); params.append(to)
        if key_id:
            where.append("key_id = ?"); params.append(key_id)
        rows = await self.fetchall(
            f"SELECT {g} AS grp, count(*) AS runs, sum(status = 'failed') AS failures, sum(fallback_reason IS NOT NULL) AS fallbacks, "
            f"COALESCE(sum(prompt_tokens),0) AS tokens_in, COALESCE(sum(output_tokens),0) AS tokens_out FROM runs WHERE {' AND '.join(where)} GROUP BY grp ORDER BY grp", params)
        # percentiles by group (SQLite has no percentile function; small groups — compute in Python)
        for r in rows:
            r["tokens_total"] = r["tokens_in"] + r["tokens_out"]
            vals = await self.fetchall(f"SELECT queue_wait_ms AS q, total_ms AS t, output_tokens_per_second AS tps FROM runs WHERE {' AND '.join(where)} AND {g} IS ? AND status = 'done'",
                                       params + [r["grp"]])
            for col, name in (("q", "queue_wait_ms"), ("t", "duration_ms"), ("tps", "tokens_per_second")):
                xs = sorted(v[col] for v in vals if v[col] is not None)
                r[name + "_p50"] = xs[len(xs) // 2] if xs else None
                r[name + "_p95"] = xs[min(len(xs) - 1, int(len(xs) * 0.95))] if xs else None
            if group_by == "tag":
                r["grp"] = json.loads(r["grp"]) if r["grp"] else []
        return rows

    async def counters(self) -> dict:
        rows = await self.fetchall("SELECT status, count(*) AS n, COALESCE(sum(prompt_tokens),0) AS tin, COALESCE(sum(output_tokens),0) AS tout FROM runs GROUP BY status")
        return {"jobs": {r["status"]: r["n"] for r in rows}, "tokens_in": sum(r["tin"] for r in rows), "tokens_out": sum(r["tout"] for r in rows)}
