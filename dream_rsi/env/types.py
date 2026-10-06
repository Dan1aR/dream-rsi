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

"""Shared types for :class:`~dream_rsi.env.TraceEnv` (live + replay).

Branch-level interface:
  TraceEnv.probe_new()                  → (Observation, branch_idx, isFinish)
  TraceEnv.probe_more(branch_idx)       → (Observation, isFinish)
  TraceEnv.get_cost_per_branch_active() → probes per opened branch
  CellMeta.branch / CellMeta.attempt    = branch index / probe depth

Cell-level extensions (parallel explore waves):
  env.legal_actions() / env.probe(cell_id) / env.observed() → rich Observation prefix
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class CellMeta:
    """Structural info for one matrix cell — safe before reveal (no outcomes)."""

    id: str
    branch: int
    seq: int
    attempt: int
    parent_id: str | None
    tags: list[str]


# Backward-compatible alias
ActionMeta = CellMeta


@dataclass
class Observation:
    """Outcome revealed after visiting one cell (one explore/refine attempt)."""

    cell_id: str
    branch: int
    attempt: int
    score: float
    evaluated: bool = False
    valid: bool = False
    fail_class: str = "no_score"
    error: str | None = None
    n_valid: int | None = None
    n_total: int | None = None
    sol_times_ms: list[float | None] = field(default_factory=list)
    valids: list[bool | None] = field(default_factory=list)
    max_gaps: list[float | None] = field(default_factory=list)
    status: str | None = None
    delta_vs_baseline: float | None = None
    delta_vs_parent: float | None = None
    budget_spent: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class SearchEnv(Protocol):
    """Branch-level probe interface + cell-level explore extensions."""

    @property
    def baseline_score(self) -> float: ...

    @property
    def budget_spent(self) -> int: ...

    @property
    def best_so_far(self) -> float: ...

    @property
    def max_parallelism(self) -> int: ...

    @property
    def done(self) -> bool: ...

    def reset(self) -> None: ...
    def legal_actions(self) -> list[str]: ...
    def legal_roots(self) -> list[str]: ...
    def opened_branches(self) -> list[int]: ...
    def is_legal(self, cell_id: str) -> bool: ...
    def meta(self, cell_id: str) -> CellMeta: ...
    def observed(self) -> dict[str, Observation]: ...
    def step(self, cell_id: str) -> Observation: ...
    def probe(self, cell_id: str) -> Observation: ...
    def probe_batch(
        self,
        cell_ids: list[str],
        *,
        max_workers: int | None = None,
        on_reveal: Callable[[Observation], None] | None = None,
    ) -> list[Observation]: ...
    def probe_new(self) -> tuple[Observation, int, bool]: ...
    def probe_more(self, branch: int) -> tuple[Observation, bool]: ...
    def get_cost_per_branch_active(self) -> list[int]: ...
