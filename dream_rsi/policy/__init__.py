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

"""Exploration policies: API, baseline, discovered OptimalPolicy, and loader.

Online (:func:`dream_rsi.env.runner.run_live`) and offline (:func:`dream_rsi.learning.pipeline.run_learning`)
both load policies from here and run ``policy.solve(dream_rsi.env.TraceEnv)``.
"""

from dream_rsi.policy.api import (
    GridPlan,
    GridPlanningContext,
    LLMDesignedMethod,
    SimResult,
    _budget_done,
    _last_observation,
    _record_curve,
    _safe_probe_more,
    _safe_probe_new,
    finalize_result,
)
from dream_rsi.policy.baseline import ParallelRefineMethod
from dream_rsi.policy.constants import METHOD_REL, METHOD_TEMPLATE_REL, PROMOTED_POLICY_REL
from dream_rsi.policy.load import default_policy_source, load_policy, load_policy_class
from dream_rsi.policy.observation_signal import (
    branch_failed_hard,
    branch_promising,
    probe_improved_vs_baseline,
    probe_improved_vs_parent,
    probe_value,
)

__all__ = [
    "LLMDesignedMethod",
    "GridPlan",
    "GridPlanningContext",
    "ParallelRefineMethod",
    "SimResult",
    "METHOD_REL",
    "METHOD_TEMPLATE_REL",
    "PROMOTED_POLICY_REL",
    "finalize_result",
    "_budget_done",
    "_record_curve",
    "_safe_probe_new",
    "_safe_probe_more",
    "_last_observation",
    "default_policy_source",
    "load_policy",
    "load_policy_class",
    "probe_value",
    "probe_improved_vs_baseline",
    "probe_improved_vs_parent",
    "branch_promising",
    "branch_failed_hard",
]
