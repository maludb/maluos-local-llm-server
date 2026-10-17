"""Node settings: config/node.toml (the port, the backend URL, the name) + the settings table (everything the manager may change).
Secrets never live here — keys are hashes in the database."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The settings the manager may PATCH (design §6.6), with their defaults. Types are enforced by `coerce_setting`.
SETTING_DEFAULTS: dict[str, object] = {
    "name": "node",
    "local_parallel": 1,            # S5
    "cloud_parallel": 0,            # 0 = unbounded
    "affinity_window_jobs": 10,     # §3 step 3
    "affinity_window_seconds": 30,
    "fallback_after_seconds": 60,   # S4
    "max_run_seconds": 3600,
    "result_retention_hours": 72,
    "payload_retention_days": 30,
    "auto_pull": False,             # S7
    "default_model": "",
    "keep_alive_default": "5m",
    "default_expected_ms": 30000,   # an unknown model's expected duration
    "default_load_ms": 5000,
    "cloud_route": "daemon",        # S9: daemon | direct
    "timezone": "UTC",
    "sync_keepalive_seconds": 10,   # the SSE comment cadence while queued
}

BOOL_SETTINGS = {"auto_pull"}
INT_SETTINGS = {"local_parallel", "cloud_parallel", "affinity_window_jobs", "affinity_window_seconds", "fallback_after_seconds",
                "max_run_seconds", "result_retention_hours", "payload_retention_days", "default_expected_ms", "default_load_ms",
                "sync_keepalive_seconds"}


def coerce_setting(key: str, value) -> object:
    if key not in SETTING_DEFAULTS:
        raise ValueError(f"unknown setting {key}")
    if key in BOOL_SETTINGS:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("1", "true", "yes", "on"):
            return True
        if str(value).lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key} must be a boolean")
    if key in INT_SETTINGS:
        try:
            v = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer")
        if v < 0 or (key == "local_parallel" and v < 1):
            raise ValueError(f"{key} out of range")
        return v
    if key == "cloud_route" and value not in ("daemon", "direct"):
        raise ValueError("cloud_route must be daemon or direct")
    return str(value)


@dataclass
class NodeConfig:
    port: int = 11435
    bind: str = "0.0.0.0"
    backend_url: str = "http://127.0.0.1:11434"
    backend_kind: str = "ollama"
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    name: str = "node"

    @classmethod
    def load(cls, path: Path | None = None) -> "NodeConfig":
        path = path or Path(os.environ.get("SLLM_CONFIG", ROOT / "config" / "node.toml"))
        raw: dict = {}
        if path.exists():
            raw = tomllib.loads(path.read_text())
        cfg = cls()
        cfg.port = int(os.environ.get("SLLM_PORT", raw.get("port", cfg.port)))
        cfg.bind = os.environ.get("SLLM_BIND", raw.get("bind", cfg.bind))
        cfg.backend_url = os.environ.get("OLLAMA_HOST", raw.get("backend_url", cfg.backend_url))
        if cfg.backend_url and not cfg.backend_url.startswith("http"):
            cfg.backend_url = "http://" + cfg.backend_url
        cfg.backend_kind = raw.get("backend_kind", cfg.backend_kind)
        cfg.data_dir = Path(os.environ.get("SLLM_DATA_DIR", raw.get("data_dir", cfg.data_dir)))
        cfg.name = raw.get("name", cfg.name)
        return cfg
