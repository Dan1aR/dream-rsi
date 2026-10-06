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

"""Policy solve + observation helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from dream_rsi.env.types import Observation
from dream_rsi.learning.heuristic_search import eval as heuristic_eval
from dream_rsi.policy import ParallelRefineMethod
from dream_rsi.policy.constants import METHOD_TEMPLATE_REL
from dream_rsi.policy.observation_signal import branch_failed_hard, branch_promising


def test_parallel_refine_on_mini_trace(replay_env):
    result = ParallelRefineMethod({"max_width": 2, "max_refine": 1}).solve(replay_env)
    assert result.best == pytest.approx(0.9)
    assert result.budget == 4


def test_template_policy_loads_and_spends_no_budget(replay_env):
    """The proposer's starting point is a valid but empty controller."""
    template = Path(__file__).resolve().parents[1] / METHOD_TEMPLATE_REL
    spec = importlib.util.spec_from_file_location("tpl", str(template))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.OptimalPolicy()
    assert policy.NAME == "OptimalPolicy"
    result = policy.solve(replay_env)
    assert result.budget == 0


def test_observation_signals():
    obs = Observation(
        cell_id="b0_a0",
        branch=0,
        attempt=0,
        score=0.5,
        evaluated=True,
        valid=True,
        fail_class="ok",
        delta_vs_baseline=0.1,
        budget_spent=1,
    )
    assert branch_promising(obs, baseline=0.4)
    assert branch_failed_hard(
        Observation(
            cell_id="b0_a0",
            branch=0,
            attempt=0,
            score=0.0,
            fail_class="compile_other",
            budget_spent=1,
        )
    )


def test_heuristic_eval_resolve_target(mini_trace):
    assert heuristic_eval._resolve_target("run_best", mini_trace) == pytest.approx(0.9)
