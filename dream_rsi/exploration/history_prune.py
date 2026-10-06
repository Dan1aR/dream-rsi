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

"""Prune low-scoring attempts inside archived explore history (``history/iterNNN/``)."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

_ITER_ARCHIVE_RE = re.compile(r"^iter(\d{3})$")
ARCHIVE_FILTERS = ("all", "prev_global_best")


def archive_filter_mode() -> str:
    """``DRSI_HISTORY_ARCHIVE_FILTER``: ``all`` (default) or ``prev_global_best``."""
    mode = os.environ.get("DRSI_HISTORY_ARCHIVE_FILTER", "all").strip().lower()
    return mode if mode in ARCHIVE_FILTERS else "all"


def read_attempt_score(attempt_dir: Path) -> float | None:
    score_file = attempt_dir / "eval" / "score.json"
    if not score_file.is_file():
        return None
    try:
        data = json.loads(score_file.read_text(encoding="utf-8"))
        if data.get("score") is not None:
            return float(data["score"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def iter_archive_global_best(archive_dir: Path) -> float | None:
    """Highest ``score`` across all ``explore_b*/attempt_*`` under an iter archive."""
    best: float | None = None
    if not archive_dir.is_dir():
        return None
    for attempt in archive_dir.glob("explore_b*/attempt_*"):
        if not attempt.is_dir():
            continue
        score = read_attempt_score(attempt)
        if score is not None and (best is None or score > best):
            best = score
    return best


def iter_archive_run_best(archive_dir: Path) -> float | None:
    """Run-best for one iter archive (``raw_scores.json`` first, else attempt scores)."""
    raw = archive_dir / "raw_scores.json"
    if raw.is_file():
        try:
            from dream_rsi.env.trace.raw import trace_from_raw_file

            return float(trace_from_raw_file(raw).best_score)
        except (OSError, ValueError, TypeError, KeyError):
            pass
    return iter_archive_global_best(archive_dir)


def prior_iters_global_best(workspace: Path, cycle: int) -> float | None:
    """Max run-best across all prior ``iter000 … iter{cycle-1}`` archives."""
    if cycle <= 0:
        return None
    ws = workspace.resolve()
    best: float | None = None
    for i in range(cycle):
        arch = ws / "history" / f"iter{i:03d}"
        if not arch.is_dir():
            continue
        score = iter_archive_run_best(arch)
        if score is not None and (best is None or score > best):
            best = score
    return best


def prune_iter_archive(archive_dir: Path, *, threshold: float) -> dict[str, Any]:
    """Drop ``attempt_*`` dirs with score missing or ``<= threshold`` (scheme A: global)."""
    archive_dir = archive_dir.resolve()
    removed: list[str] = []
    kept: list[str] = []

    for branch in sorted(p for p in archive_dir.glob("explore_b*") if p.is_dir()):
        for attempt in sorted(p for p in branch.glob("attempt_*") if p.is_dir()):
            rel = attempt.relative_to(archive_dir).as_posix()
            score = read_attempt_score(attempt)
            if score is not None and score > threshold:
                kept.append(rel)
                continue
            shutil.rmtree(attempt, ignore_errors=True)
            removed.append(rel)

        # Remove branch dir if no attempts remain (baseline-only shells add no value).
        if not any(branch.glob("attempt_*")):
            shutil.rmtree(branch, ignore_errors=True)

    return {
        "archive": str(archive_dir),
        "threshold": threshold,
        "kept": kept,
        "removed": removed,
        "n_kept": len(kept),
        "n_removed": len(removed),
    }


def maybe_prune_archived_iter(
    workspace: Path,
    archive_dir: Path,
    *,
    cycle: int,
    filter_mode: str | None = None,
) -> dict[str, Any] | None:
    """Apply archive filter after ``history/iterNNN/`` is written. Returns prune stats or None."""
    mode = (filter_mode or archive_filter_mode()).strip().lower()
    if mode == "all":
        return None
    if mode != "prev_global_best":
        print(f"[history_prune] unknown filter {mode!r}; expected {ARCHIVE_FILTERS}", flush=True)
        return None

    parsed = _ITER_ARCHIVE_RE.match(archive_dir.name)
    if parsed is None:
        return None

    threshold = prior_iters_global_best(workspace, cycle)
    if threshold is None:
        print(
            f"[history_prune] skip {archive_dir.name}: no scored prior iter archives",
            flush=True,
        )
        return None

    stats = prune_iter_archive(archive_dir, threshold=threshold)
    print(
        f"[history_prune] {archive_dir.name} vs prior iters best={threshold:.6f}: "
        f"kept {stats['n_kept']}, removed {stats['n_removed']}",
        flush=True,
    )
    return stats
