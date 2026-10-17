"""Keys are the one wall (design §2): `sllm_` + 48 hex, shown once, only the SHA-256 kept. Two kinds — `inference` (the faces) and
`node` (the node API). Limits live on the row."""
from __future__ import annotations

import hashlib
import secrets
import uuid

from .store import Store, now_iso

PREFIX = "sllm_"
KEY_FIELDS = ("name", "max_priority", "max_concurrent", "cloud_allowed", "fallback_allowed", "payload_retention", "allowed_models",
              "daily_token_cap", "status")


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def present(row: dict) -> dict:
    """What the node API answers about a key — never the hash."""
    out = {k: row[k] for k in ("id", "seq", "kind", "name", "prefix", "max_priority", "max_concurrent", "payload_retention",
                               "allowed_models", "daily_token_cap", "status", "created_at", "last_used_at", "revoked_at")}
    out["cloud_allowed"] = bool(row["cloud_allowed"])
    out["fallback_allowed"] = bool(row["fallback_allowed"])
    return out


class Keys:
    def __init__(self, store: Store):
        self.store = store

    async def mint(self, kind: str, name: str, **limits) -> tuple[str, dict]:
        """Returns (the raw key — shown once, never stored — and the row)."""
        if kind not in ("inference", "node"):
            raise ValueError("kind must be inference or node")
        raw = PREFIX + secrets.token_hex(24)
        row = {"id": str(uuid.uuid4()), "kind": kind, "name": name, "hash": hash_key(raw), "prefix": raw[:10], "created_at": now_iso(),
               "max_priority": 5, "max_concurrent": 4, "cloud_allowed": 0, "fallback_allowed": 0, "payload_retention": "metadata",
               "allowed_models": None, "daily_token_cap": None, "status": "active", "webhook_secret": secrets.token_hex(16)}
        for k, v in limits.items():
            if k not in KEY_FIELDS:
                raise ValueError(f"unknown key field {k}")
            row[k] = int(bool(v)) if k in ("cloud_allowed", "fallback_allowed") else v
        self._validate(row)
        await self.store.insert_key(dict(row))
        return raw, await self.store.key(row["id"])

    @staticmethod
    def _validate(row: dict) -> None:
        if not (1 <= int(row["max_priority"]) <= 9):
            raise ValueError("max_priority must be 1–9")
        if int(row["max_concurrent"]) < 1:
            raise ValueError("max_concurrent must be at least 1")
        if row["payload_retention"] not in ("none", "metadata", "full"):
            raise ValueError("payload_retention must be none, metadata or full")
        if row["status"] not in ("active", "paused", "revoked"):
            raise ValueError("status must be active, paused or revoked")

    async def update(self, key_id: str, changes: dict) -> dict:
        row = await self.store.key(key_id)
        if not row:
            raise KeyError(key_id)
        for k, v in changes.items():
            if k not in KEY_FIELDS:
                raise ValueError(f"unknown key field {k}")
            row[k] = int(bool(v)) if k in ("cloud_allowed", "fallback_allowed") else v
        self._validate(row)
        await self.store.update_key(key_id, {k: row[k] for k in changes})
        return await self.store.key(key_id)

    async def revoke(self, key_id: str) -> None:
        await self.store.update_key(key_id, {"status": "revoked", "revoked_at": now_iso()})

    async def authenticate(self, raw: str | None, kind: str) -> dict | None:
        """The live key row for a presented bearer of the given kind, else None. Stamps last_used_at (at most once a minute)."""
        if not raw or not raw.startswith(PREFIX):
            return None
        row = await self.store.key_by_hash(hash_key(raw.strip()))
        if not row or row["kind"] != kind or row["status"] != "active":
            return None
        await self.store.touch_key(row["id"])
        return row

    async def rotate_node_key(self) -> str:
        """A new node key shown once; every older node key dies at the next call (§6.6)."""
        for k in await self.store.keys("node"):
            if k["status"] == "active":
                await self.revoke(k["id"])
        raw, _ = await self.mint("node", "manager")
        return raw
