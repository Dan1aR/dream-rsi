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

"""Exploration policy API: probe helpers, metrics, and the policy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from dream_rsi.env.types import SearchEnv
from dream_rsi.policy.observation_signal import probe_value


@dataclass
class SimResult:
    best: float = 0.0
    budget: int = 0
    curve: list[tuple[int, float]] = field(default_factory=list)
    order: list[str] = field(default_factory=list)
    decision_rounds: int = 0
    effective_sequential_rounds: int = 0
    batch_sizes: list[int] = field(default_factory=list)

    def budget_to_reach(self, target: float) -> int | None:
        for spent, best in self.curve:
            if best >= target:
                return spent
        return None

    def best_at(self, budget: int) -> float:
        best = 0.0
        for spent, b in self.curve:
            if spent > budget:
                break
            best = b
        return best


@dataclass(frozen=True)
class GridPlan:
    """Requested shape for the *next* live exploration grid.

    ``branch_count`` is the number of root branches. ``refine_count`` is the
    number of refinement attempts allowed after each root, so a plan ``(7, 6)``
    creates branches ``0..6`` with attempts ``0..6``.
    """

    branch_count: int
    refine_count: int
    reason: str = ""
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_count": self.branch_count,
            "refine_count": self.refine_count,
            "reason": self.reason,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class GridPlanningContext:
    """Prefix-safe facts available before a new exploration grid is created.

    ``history`` only contains manifests from completed earlier cycles. It never
    contains outcomes from the grid that is about to be explored.
    """

    cycle: int
    mode: str
    fallback_branch_count: int
    fallback_refine_count: int
    hard_max_branch_count: int
    hard_max_refine_count: int
    max_workers: int | None = None
    workspace: str | None = None
    history: tuple[dict[str, Any], ...] = ()
    trace_branch_count: int | None = None
    trace_refine_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle": self.cycle,
            "mode": self.mode,
            "fallback_branch_count": self.fallback_branch_count,
            "fallback_refine_count": self.fallback_refine_count,
            "hard_max_branch_count": self.hard_max_branch_count,
            "hard_max_refine_count": self.hard_max_refine_count,
            "max_workers": self.max_workers,
            "workspace": self.workspace,
            "history": list(self.history),
            "trace_branch_count": self.trace_branch_count,
            "trace_refine_count": self.trace_refine_count,
        }


class LLMDesignedMethod(ABC):
    """A complete exploration policy — replay or live via :class:`dream_rsi.env.TraceEnv`."""

    NAME = "unnamed_method"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def solve(self, question: SearchEnv, budget: int | None = None) -> SimResult:
        pass

    def __call__(self, question: SearchEnv, budget: int | None = None) -> SimResult:
        return self.solve(question, budget=budget)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan | None:
        """Optionally choose the next cycle's grid before the environment exists.

        Returning ``None`` deliberately preserves the caller's fallback grid.
        This default keeps baseline and previously discovered policies compatible.
        """
        del context
        return None

    def description(self) -> str:
        if not self.config:
            return self.NAME
        return f"{self.NAME} | config={self.config}"


def finalize_result(env: SearchEnv, res: SimResult) -> SimResult:
    res.best = env.best_so_far
    res.budget = env.budget_spent
    res.order = env.revealed_order
    res.decision_rounds = int(getattr(env, "decision_rounds", 0))
    res.effective_sequential_rounds = int(
        getattr(env, "effective_sequential_rounds", res.decision_rounds)
    )
    res.batch_sizes = list(getattr(env, "batch_sizes", []))
    return res


def _budget_done(env: SearchEnv, budget: int | None) -> bool:
    return budget is not None and env.budget_spent >= budget


def _record_curve(res: SimResult, env: SearchEnv) -> None:
    res.curve.append((env.budget_spent, env.best_so_far))


def _safe_probe_new(question: SearchEnv):
    try:
        obs, idx, is_finish = question.probe_new()
    except (IndexError, ValueError):
        return None
    return probe_value(obs), idx, is_finish


def _safe_probe_more(question: SearchEnv, index: int):
    try:
        obs, is_finish = question.probe_more(index)
    except (IndexError, ValueError):
        return None
    return probe_value(obs), is_finish


def _last_observation(question: SearchEnv):
    """Raw rich observation after the most recent probe."""
    return question.last_observation
