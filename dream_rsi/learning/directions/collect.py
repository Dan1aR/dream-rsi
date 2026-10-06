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

"""Collect current and archived attempts into bounded summarizer inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_PROPOSAL_CHARS = 8_000
MAX_PROGRAM_CHARS = 16_000


@dataclass(frozen=True)
class AttemptSource:
    attempt_id: str
    cycle: str
    branch: int
    attempt: int
    parent_attempt_id: str | None
    status: str
    score: float | None
    parent_score: float | None
    error: str | None
    proposal: str
    program_path: str | None
    program_excerpt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_text(path: Path, *, limit: int) -> str:
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n[truncated after {limit} characters]"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _score(node_dir: Path) -> float | None:
    data = _read_json(node_dir / "eval" / "score.json") or {}
    value = data.get("combined_score", data.get("score"))
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _program_file(node_dir: Path, meta: dict[str, Any]) -> Path | None:
    rel = str(meta.get("artifact_relpath") or "").strip()
    candidates: list[Path] = []
    if rel:
        candidates.append(node_dir / rel)
    candidates.extend(
        [
            node_dir / "artifact" / "program.py",
            node_dir / "artifact" / "main.py",
        ]
    )
    artifact = node_dir / "artifact"
    if artifact.is_dir():
        candidates.extend(sorted(path for path in artifact.rglob("*") if path.is_file()))
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _parent_score(node_dir: Path, parent_id: str | None) -> float | None:
    if not parent_id:
        return None
    for meta_path in sorted(node_dir.parent.glob("attempt_*/meta.json")):
        data = _read_json(meta_path)
        if data is not None and data.get("id") == parent_id:
            return _score(meta_path.parent)
    return None


def _source_from_node(
    workspace: Path,
    node_dir: Path,
    *,
    cycle_label: str,
) -> AttemptSource | None:
    meta = _read_json(node_dir / "meta.json")
    if meta is None:
        return None
    try:
        branch = int(meta.get("branch", 0))
        seq = int(meta.get("seq", 1))
    except (TypeError, ValueError):
        return None
    attempt = max(0, seq - 1)
    attempt_id = f"{cycle_label}/b{branch}_a{attempt}"
    parent_attempt_id = (
        f"{cycle_label}/{meta['parent_id']}" if meta.get("parent_id") else None
    )
    program = _program_file(node_dir, meta)
    error = _read_text(node_dir / "error.txt", limit=8_000).strip() or None
    try:
        program_label = str(program.relative_to(workspace)) if program else None
    except ValueError:
        program_label = str(program) if program else None
    return AttemptSource(
        attempt_id=attempt_id,
        cycle=cycle_label,
        branch=branch,
        attempt=attempt,
        parent_attempt_id=parent_attempt_id,
        status=str(meta.get("status") or "unknown"),
        score=_score(node_dir),
        parent_score=_parent_score(node_dir, meta.get("parent_id")),
        error=error,
        proposal=_read_text(node_dir / "proposal.md", limit=MAX_PROPOSAL_CHARS),
        program_path=program_label,
        program_excerpt=_read_text(program, limit=MAX_PROGRAM_CHARS) if program else "",
    )


def collect_attempts(workspace: Path, *, current_cycle: int) -> list[AttemptSource]:
    """Return every archived attempt plus the current unarchived round."""
    ws = workspace.resolve()
    out: list[AttemptSource] = []
    seen: set[str] = set()

    history = ws / "history"
    if history.is_dir():
        for archive in sorted(path for path in history.iterdir() if path.is_dir()):
            for meta_path in sorted(archive.glob("explore_b*/attempt_*/meta.json")):
                source = _source_from_node(ws, meta_path.parent, cycle_label=archive.name)
                if source is not None and source.attempt_id not in seen:
                    out.append(source)
                    seen.add(source.attempt_id)

    current_label = f"iter{current_cycle:03d}"
    for meta_path in sorted(ws.glob("explore_b*/attempt_*/meta.json")):
        source = _source_from_node(ws, meta_path.parent, cycle_label=current_label)
        if source is not None and source.attempt_id not in seen:
            out.append(source)
            seen.add(source.attempt_id)

    return sorted(out, key=lambda item: (item.cycle, item.branch, item.attempt))
