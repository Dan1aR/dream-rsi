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

"""Build :class:`Observation` from trace nodes (used by :class:`~dream_rsi.env.TraceEnv`)."""

from __future__ import annotations

from dream_rsi.env.types import Observation
from dream_rsi.env.trace.types import Node


def observation_from_node(
    n: Node,
    *,
    baseline: float,
    parent_score: float | None,
    budget_spent: int,
) -> Observation:
    parent_delta = (n.score - parent_score) if parent_score is not None else None
    return Observation(
        cell_id=n.id,
        branch=n.branch,
        attempt=n.attempt,
        score=n.score,
        evaluated=n.evaluated,
        valid=n.valid,
        fail_class=n.fail_class,
        error=n.error,
        n_valid=n.n_valid,
        n_total=n.n_total,
        sol_times_ms=list(n.sol_times_ms),
        valids=list(n.valids),
        max_gaps=list(n.max_gaps),
        status=n.status,
        delta_vs_baseline=n.score - baseline,
        delta_vs_parent=parent_delta,
        budget_spent=budget_spent,
    )
