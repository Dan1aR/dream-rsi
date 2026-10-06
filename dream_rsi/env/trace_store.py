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

"""In-memory ``raw_scores.json`` store with scaffold + workspace sync."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dream_rsi.exploration.store import NodeStore, node_id as make_node_id
from dream_rsi.env.trace.raw import (
    RAW_TRACE_VERSION,
    CellRaw,
    RawNode,
    RawTrace,
    export_raw_scores,
    load_raw_scores,
    write_raw_scores,
)


def grid_limits(
    *,
    max_width: int | None = None,
    max_refine: int | None = None,
) -> tuple[int, int]:
    m = max_width if max_width is not None else int(os.environ.get("DRSI_PARALLEL_EXPLORE", "2"))
    n = max_refine if max_refine is not None else int(os.environ.get("DRSI_REFINE_PER_BRANCH", "2"))
    return m, n


def scaffold_grid_nodes(
    workspace: Path,
    *,
    max_width: int,
    max_refine: int,
) -> list[RawNode]:
    """Pre-register M×N cells (unevaluated) so live/replay share the same tree."""
    ws = workspace.resolve()
    store = NodeStore(ws)
    nodes: list[RawNode] = []
    for branch in range(max_width):
        for attempt in range(max_refine + 1):
            seq = attempt + 1
            parent_id = make_node_id(branch, seq - 1) if attempt > 0 else None
            nid = make_node_id(branch, seq)
            node_dir = store.branch_dir(branch) / f"attempt_{seq - 1:02d}"
            tags = ["explore"] if attempt == 0 else ["refine"]
            intent = f"explore {branch}" if attempt == 0 else f"refine b{branch} attempt_{attempt}"
            nodes.append(
                RawNode(
                    id=nid,
                    branch=branch,
                    attempt=attempt,
                    parent_id=parent_id,
                    intent=intent,
                    tags=tags,
                    path=str(node_dir),
                    created_at=None,
                    status="pending",
                    raw=CellRaw(evaluated=False),
                )
            )
    return nodes


class TraceStore:
    """Canonical explore state as :class:`RawTrace` (``raw_scores.json`` format)."""

    def __init__(self, raw: RawTrace, *, path: Path | None = None) -> None:
        self.raw = raw
        self.path = path.resolve() if path is not None else None
        self._by_id = {n.id: n for n in raw.nodes}

    @classmethod
    def from_frozen(cls, path: Path) -> TraceStore:
        return cls(load_raw_scores(path), path=path)

    @classmethod
    def from_raw(cls, raw: RawTrace) -> TraceStore:
        """Wrap an already-loaded raw trace (no backing file; used for replay)."""
        return cls(raw)

    @classmethod
    def for_live(
        cls,
        workspace: Path,
        *,
        trace_path: Path | None = None,
        max_width: int | None = None,
        max_refine: int | None = None,
        n_problems: int | None = None,
    ) -> TraceStore:
        ws = workspace.resolve()
        out = trace_path or (ws / "raw_scores.json")
        if out.is_file():
            store = cls.from_frozen(out)
            store.raw.workspace = str(ws)
            return store

        exported = export_raw_scores(ws, n_problems=n_problems)
        m, n = grid_limits(max_width=max_width, max_refine=max_refine)
        scaffold = scaffold_grid_nodes(ws, max_width=m, max_refine=n)
        existing = {node.id for node in exported.nodes}
        merged = list(exported.nodes)
        for cell in scaffold:
            if cell.id not in existing:
                merged.append(cell)
                existing.add(cell.id)

        raw = replace(
            exported,
            workspace=str(ws),
            exported_at=datetime.now(timezone.utc).isoformat(),
            nodes=merged,
        )
        store = cls(raw, path=out)
        store.flush()
        return store

    def flush(self) -> Path | None:
        if self.path is None:
            return None
        self.raw.exported_at = datetime.now(timezone.utc).isoformat()
        write_raw_scores(self.raw, self.path)
        return self.path

    def get_node(self, cell_id: str) -> RawNode | None:
        return self._by_id.get(cell_id)

    def is_evaluated(self, cell_id: str) -> bool:
        node = self._by_id.get(cell_id)
        return node is not None and node.raw.evaluated

    def refresh_from_workspace(self, workspace: Path, *, n_problems: int | None = None) -> None:
        """Merge latest workspace node outcomes into the in-memory raw trace."""
        ws = workspace.resolve()
        fresh = export_raw_scores(ws, n_problems=n_problems or self.raw.n_problems or None)
        self.raw.baseline = fresh.baseline
        self.raw.n_problems = fresh.n_problems
        fresh_by_id = {n.id: n for n in fresh.nodes}
        updated: list[RawNode] = []
        for old in self.raw.nodes:
            if old.id in fresh_by_id:
                updated.append(fresh_by_id[old.id])
            else:
                updated.append(old)
        for nid, node in fresh_by_id.items():
            if nid not in self._by_id:
                updated.append(node)
        self.raw.nodes = updated
        self.raw.workspace = str(ws)
        self._by_id = {n.id: n for n in self.raw.nodes}
