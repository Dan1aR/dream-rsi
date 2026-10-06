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

"""Accumulated replay traces for multi-environment policy evaluation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

TRACE_POOL_NAME = "trace_pool"
MANIFEST_KEY = "trace_pool"


def workspace_trace_pool(workspace: Path) -> Path:
    return workspace.resolve() / TRACE_POOL_NAME


def learning_trace_pool(job_root: Path) -> Path:
    return job_root.resolve() / TRACE_POOL_NAME


def list_pool_traces(pool_root: Path) -> list[Path]:
    """Return sorted ``*/raw_scores.json`` paths under a trace pool."""
    root = pool_root.resolve()
    if not root.is_dir():
        return []
    out: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        raw = entry / "raw_scores.json"
        if raw.is_file():
            out.append(raw.resolve())
    return out


def append_trace(pool_root: Path, name: str, raw_scores: Path) -> Path | None:
    """Copy one frozen trace and its optional live-cycle metadata sidecar."""
    src = raw_scores.resolve()
    if not src.is_file():
        return None
    dest = pool_root.resolve() / name / "raw_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    sidecar = src.parent / "live_cycle_manifest.json"
    if sidecar.is_file():
        shutil.copy2(sidecar, dest.parent / sidecar.name)
    return dest


def append_workspace_trace(workspace: Path, name: str, raw_scores: Path) -> Path | None:
    return append_trace(workspace_trace_pool(workspace), name, raw_scores)


def write_pool_manifest(pool_root: Path) -> Path:
    """Write ``manifest.json`` listing traces (for eval / debugging)."""
    pool_root = pool_root.resolve()
    pool_root.mkdir(parents=True, exist_ok=True)
    traces = list_pool_traces(pool_root)
    manifest = {
        "n_traces": len(traces),
        "traces": [{"name": p.parent.name, "path": str(p)} for p in traces],
    }
    out = pool_root / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def sync_workspace_pool_to_job(workspace: Path, job_root: Path) -> Path:
    """Mirror workspace ``trace_pool/`` into a learning job (includes current cycle)."""
    ws_pool = workspace_trace_pool(workspace)
    job_pool = learning_trace_pool(job_root)
    job_pool.mkdir(parents=True, exist_ok=True)

    for src in list_pool_traces(ws_pool):
        append_trace(job_pool, src.parent.name, src)

    # Current-cycle trace on workspace (pre-archive) wins over same name in pool.
    ws_raw = workspace.resolve() / "raw_scores.json"
    if ws_raw.is_file():
        append_trace(job_pool, "_current", ws_raw)

    write_pool_manifest(job_pool)
    return job_pool
