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

"""Discovered exploration controller (reset from this template every round).

FRAMEWORK ONLY — fill in the adaptive logic in ``solve`` and ``_schedule``:

- one ``beta`` knob (high = explore, low = save budget), read from config;
- route ALL behaviour thresholds through ``_schedule(beta)``;
- optionally implement ``plan_grid`` to choose arbitrary integer grid dimensions
  for the *next* cycle from completed-cycle history only;
- decide prefix-only (from ``question.observed()``) — no run-best / absolute cutoffs;
- keep ``class OptimalPolicy`` and ``NAME = "OptimalPolicy"``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from dream_rsi.env.types import SearchEnv
from dream_rsi.policy.api import GridPlan, GridPlanningContext, LLMDesignedMethod, SimResult


class OptimalPolicy(LLMDesignedMethod):
    NAME = "OptimalPolicy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # Single exploration <-> budget knob (high beta = explore more).
        self.beta = float(self.config.get("beta", 1.0))
        self.sch = self._schedule(self.beta)

    def _schedule(self, beta: float) -> Dict[str, Any]:
        """Map the single knob to every behaviour threshold. (fill in)"""
        return {}

    def plan_grid(self, context: GridPlanningContext) -> GridPlan | None:
        """Optionally plan the next live grid before it is created.

        Return arbitrary valid integer ``GridPlan(branch_count, refine_count)``
        within ``context.hard_max_*``. Use only ``context.history`` from completed
        cycles. Return ``None`` when history is insufficient; the runner will use
        its configured fallback grid.
        """
        del context
        return None

    def solve(self, question: SearchEnv, budget: int | None = None) -> SimResult:
        """Adaptive exploration loop. (you need implement this function)"""
        question.reset()
        return SimResult()
