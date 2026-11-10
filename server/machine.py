"""What the node knows about the machine it runs on: the facts `/node/v1/status` answers and the run record keeps
(memory free at a job's start). Linux reads /proc; macOS asks sysctl and vm_stat. Every reader is best effort: a missing
fact is absent, never a failure. The GPU readers arrive with the estate client (docs/estate-node-plan.md N3)."""
from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess


def _run(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def memory() -> dict:
    """{"memory_total_bytes", "memory_free_bytes"} — free is what a new model could take (MemAvailable on Linux)."""
    out: dict = {}
    if os.path.exists("/proc/meminfo"):
        try:
            for line in open("/proc/meminfo"):
                k, _, v = line.partition(":")
                if k == "MemTotal":
                    out["memory_total_bytes"] = int(v.split()[0]) * 1024
                elif k == "MemAvailable":
                    out["memory_free_bytes"] = int(v.split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        return out
    if platform.system() == "Darwin":
        try:
            out["memory_total_bytes"] = int(_run(["sysctl", "-n", "hw.memsize"]).strip())
        except ValueError:
            pass
        page, free = 4096, 0
        for line in _run(["vm_stat"]).splitlines():
            if "page size of" in line:
                try:
                    page = int(line.split("page size of")[1].split()[0])
                except (ValueError, IndexError):
                    pass
            name, _, val = line.partition(":")
            if name.strip() in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                try:
                    free += int(val.strip().rstrip("."))
                except ValueError:
                    pass
        if free:
            out["memory_free_bytes"] = free * page
    return out


def facts(model_store: str | None = None) -> dict:
    """The machine block of /node/v1/status."""
    f: dict = {"hostname": socket.gethostname(), "os": f"{platform.system()} {platform.release()}", "arch": platform.machine(),
               "cpu_count": os.cpu_count(), "gpus": []}
    f.update(memory())
    try:
        f["cpu_load_1m"] = os.getloadavg()[0]
    except (OSError, AttributeError):
        pass
    try:
        f["disk_free_bytes"] = shutil.disk_usage(model_store or os.path.expanduser("~")).free
    except OSError:
        pass
    return f
