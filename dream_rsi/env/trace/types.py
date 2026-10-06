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

"""In-memory trace tree used by :class:`~dream_rsi.env.TraceEnv`."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Node:
    id: str
    branch: int
    seq: int
    parent_id: str | None
    intent: str
    tags: list[str]
    path: Path
    score: float = 0.0
    valid: bool = False
    fail_class: str = "no_score"
    error: str | None = None
    evaluated: bool = False
    n_valid: int | None = None
    n_total: int | None = None
    sol_times_ms: list[float | None] = field(default_factory=list)
    valids: list[bool | None] = field(default_factory=list)
    max_gaps: list[float | None] = field(default_factory=list)
    created_at: str | None = None
    status: str | None = None
    parent_score: float | None = None
    delta_vs_parent: float | None = None
    delta_vs_baseline: float | None = None

    @property
    def attempt(self) -> int:
        return self.seq - 1

    @property
    def is_root(self) -> bool:
        return self.parent_id is None


@dataclass
class Trace:
    nodes: list[Node]
    baseline_score: float = 0.0
    by_id: dict[str, Node] = field(default_factory=dict)
    children: dict[str | None, list[str]] = field(default_factory=dict)

    @property
    def roots(self) -> list[str]:
        return self.children.get(None, [])

    def branch_chain(self, branch: int) -> list[Node]:
        ns = [n for n in self.nodes if n.branch == branch]
        return sorted(ns, key=lambda n: n.seq)

    @property
    def branches(self) -> list[int]:
        return sorted({n.branch for n in self.nodes})

    @property
    def best_score(self) -> float:
        return max((n.score for n in self.nodes), default=0.0)

    @property
    def recorded_order(self) -> list[str]:
        def key(n: Node):
            return (n.created_at is None, n.created_at or "", n.branch, n.seq)

        return [n.id for n in sorted(self.nodes, key=key)]
