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

"""Allocate a self-contained learning job (frozen trace + agent sandbox)."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from dream_rsi.learning.pipeline import (
    ROOT,
    export_workspace_trace,
    load_raw,
    resolve_baseline_config,
)
from dream_rsi.policy.constants import METHOD_REL
from dream_rsi.policy.load import promoted_policy_path

PROMPT_REL = "dream_rsi/prompts/heuristic_proposer.txt"
DIRECTION_SUMMARY_REL = "directions/direction_summary.json"

_COPY_TREES: tuple[tuple[str, str], ...] = (
    ("dream_rsi/policy", "dream_rsi/policy"),
    ("dream_rsi/learning/heuristic_search", "dream_rsi/learning/heuristic_search"),
)
_COPY_FILES: tuple[tuple[str, str], ...] = (
    (PROMPT_REL, PROMPT_REL),
)


def workspace_learning_root(workspace: Path) -> Path:
    """Directory holding the learning jobs of one exploration workspace."""
    return workspace.resolve() / "learning"


def allocate_learning_workspace(source: Path, *, dest: Path | None = None) -> Path:
    """Create an isolated learning job directory from a workspace or trace.

    Layout::

        <job>/
          manifest.json
          trace/raw_scores.json
          dream_rsi/policy/                    # read-only API + editable discovered/method.py
          dream_rsi/learning/heuristic_search/
          dream_rsi/prompts/heuristic_proposer.txt
          heuristic_history/                   # one archived r####_* dir per candidate
          heuristic_results/
          logs/

    ``source`` is an explore workspace (its trace is exported) or a path to
    ``raw_scores.json``.  ``dest`` defaults to ``<workspace>/learning/<UTC stamp>``
    and is required for trace sources.  A workspace's promoted policy seeds
    ``discovered/method.py`` so it is evaluated as the incumbent candidate.

    Returns the new job root path.
    """
    source = source.resolve()
    if source.is_file():
        trace_src: Path | None = source
        source_workspace: Path | None = None
    elif source.is_dir():
        trace_src = None
        source_workspace = source
    else:
        raise FileNotFoundError(f"source not found: {source}")

    if dest is None:
        if source_workspace is None:
            raise ValueError("pass dest when allocating a learning job from a trace file")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        dest = workspace_learning_root(source_workspace) / stamp
    job_root = dest.resolve()
    if job_root.exists() and any(job_root.iterdir()):
        raise FileExistsError(f"learning job already exists and is not empty: {job_root}")

    job_root.mkdir(parents=True, exist_ok=True)
    for name in ("trace", "heuristic_history", "heuristic_results", "logs"):
        (job_root / name).mkdir(parents=True, exist_ok=True)

    trace_out = job_root / "trace" / "raw_scores.json"
    direction_summary: Path | None = None
    if trace_src is not None:
        shutil.copy2(trace_src, trace_out)
    else:
        export_workspace_trace(source_workspace, out=trace_out)
        direction_summary = copy_direction_summary_to_job(source_workspace, job_root)

    for src_rel, dst_rel in _COPY_TREES:
        src = ROOT / src_rel
        dst = job_root / dst_rel
        if not src.is_dir():
            raise FileNotFoundError(f"missing source tree to copy: {src}")
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "method.py"),
            dirs_exist_ok=True,
        )

    for src_rel, dst_rel in _COPY_FILES:
        src = ROOT / src_rel
        dst = job_root / dst_rel
        if not src.is_file():
            raise FileNotFoundError(f"missing source file to copy: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    seed_method: Path | None = None
    if source_workspace is not None and promoted_policy_path(source_workspace).is_file():
        seed_method = promoted_policy_path(source_workspace)
        dst = job_root / METHOD_REL
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_method, dst)

    preset = resolve_baseline_config(trace=load_raw(trace_out))
    manifest = {
        "job_id": job_root.name,
        "source_workspace": str(source_workspace) if source_workspace else None,
        "source_trace": str(trace_src) if trace_src is not None else None,
        "seed_method": str(seed_method) if seed_method is not None else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_preset": preset,
        "trace": "trace/raw_scores.json",
        "method_file": METHOD_REL,
        "prompt": PROMPT_REL,
        "history_dir": "heuristic_history",
        "results_dir": "heuristic_results",
        "logs_dir": "logs",
        "trace_pool": "trace_pool",
        "direction_summary": DIRECTION_SUMMARY_REL if direction_summary else None,
    }
    (job_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return job_root


def copy_direction_summary_to_job(workspace: Path, job_root: Path) -> Path | None:
    """Copy the workspace's validated current catalog into an isolated job."""
    src = workspace.resolve() / "directions" / "current.json"
    if not src.is_file():
        return None
    # Loading validates before the artifact enters a learning job.
    from dream_rsi.learning.directions.schema import load_catalog

    load_catalog(src)
    dst = job_root.resolve() / DIRECTION_SUMMARY_REL
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    manifest_path = job_root.resolve() / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            manifest["direction_summary"] = DIRECTION_SUMMARY_REL
            manifest_path.write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
    return dst


def load_manifest(job_root: Path) -> dict:
    path = job_root.resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"not a learning job (missing manifest.json): {job_root}")
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_learning_baseline_preset(
    job_root: Path,
    *,
    overrides: dict | None = None,
) -> dict:
    """Refresh a job's baseline grid from its trace plus runtime overrides.

    Explicit evolve ``--branches``/``--refine`` values remain authoritative
    when provided.
    """
    root = job_root.resolve()
    manifest = load_manifest(root)
    raw = load_raw(root / manifest.get("trace", "trace/raw_scores.json"))
    preset = resolve_baseline_config(overrides, trace=raw)
    manifest["baseline_preset"] = preset
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return preset


def promote_method_to_workspace(learning_root: Path, workspace: Path) -> Path:
    """Copy a learning job's selected ``method.py`` to the workspace's promoted policy."""
    manifest = load_manifest(learning_root)
    src = (learning_root / manifest.get("method_file", METHOD_REL)).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"discovered method missing: {src}")
    dst = promoted_policy_path(workspace)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
