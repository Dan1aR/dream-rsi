# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared agent types and helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# --- live subprocess registry (for clean Ctrl-C) ---------------------------
# Agents register every spawned subprocess here. On SIGINT the CLI calls
# shutdown_agents(): it sets SHUTDOWN (agents stop retrying/spawning) and
# SIGKILLs every live process group so no orphan keeps burning API quota.

SHUTDOWN = threading.Event()
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


def register_proc(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.add(proc)


def unregister_proc(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(proc)


def kill_proc_group(proc: subprocess.Popen) -> None:
    """SIGKILL the process group of a child started with start_new_session=True."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def shutdown_agents() -> int:
    """Stop all agents: no new spawns/retries, kill every live subprocess group."""
    SHUTDOWN.set()
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    killed = 0
    for proc in procs:
        if proc.poll() is None:
            kill_proc_group(proc)
            killed += 1
    return killed


@dataclass
class AgentResult:
    status: str = "ok"
    message: str = ""


class Agent(Protocol):
    name: str

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        node_dir: Path,
        log_dir: Path,
        guide: dict[str, Any],
        run_cwd: Path | None = None,
        stage: str | None = None,
    ) -> AgentResult: ...


def save_prompt(log_dir: Path, prompt: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "prompt.txt"
    path.write_text(prompt, encoding="utf-8")
    return path


def save_run_logs(log_dir: Path, *, stdout: str = "", stderr: str = "", command: list[str] | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    if command is not None:
        (log_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    (log_dir / "stdout.txt").write_text(stdout or "", encoding="utf-8")
    (log_dir / "stderr.txt").write_text(stderr or "", encoding="utf-8")


def streaming_enabled() -> bool:
    """When true, agents echo tool calls / eval output to the terminal live."""
    return os.environ.get("DRSI_AGENT_STREAM", "").strip().lower() in ("1", "true", "yes", "on")
