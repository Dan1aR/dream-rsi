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

"""Map rich :class:`Observation` → scalar / boolean signals for controllers.

Read-only during heuristic search — agents use these helpers from
``discovered/method.py`` but must not edit this file.
"""

from __future__ import annotations

from dream_rsi.env.types import Observation

_COMPILE_FAIL = frozenset({"compile_other", "compile_env"})


def probe_value(obs: Observation) -> float:
    """Primary scalar after a probe."""
    return obs.score


def probe_improved_vs_baseline(obs: Observation, *, baseline: float) -> float:
    """How much this cell beat the starting program."""
    if obs.delta_vs_baseline is not None:
        return obs.delta_vs_baseline
    return obs.score - baseline


def probe_improved_vs_parent(obs: Observation) -> float | None:
    return obs.delta_vs_parent


def branch_promising(obs: Observation, *, baseline: float) -> bool:
    """Whether a root observation justifies spending refine budget on its branch."""
    if not obs.evaluated:
        return False
    if obs.fail_class in _COMPILE_FAIL:
        return False
    if obs.n_valid is not None and obs.n_valid == 0:
        return False
    if obs.fail_class == "runtime" and not obs.valid:
        return False
    if obs.valid or obs.score >= baseline:
        return True
    return obs.fail_class == "ok"


def branch_failed_hard(obs: Observation) -> bool:
    """Compile / zero-valid style failures — usually skip refine."""
    if obs.fail_class in _COMPILE_FAIL:
        return True
    if obs.n_valid is not None and obs.n_valid == 0:
        return True
    return False
