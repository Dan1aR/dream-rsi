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

"""Per-cycle grid planning: fallback, validation, and live/replay enforcement."""

from __future__ import annotations

import json
from pathlib import Path

from dream_rsi.env.runner import run_live, run_replay
from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.learning.heuristic_search.eval import evaluate_on_trace
from dream_rsi.policy import GridPlan, GridPlanningContext, LLMDesignedMethod, ParallelRefineMethod
from dream_rsi.policy.api import SimResult
from dream_rsi.policy.grid_planning import resolve_grid_plan


class _Planner(LLMDesignedMethod):
    NAME = "planner"

    def __init__(self, plan: GridPlan | None):
        super().__init__()
        self.plan = plan
        self.seen_legal: list[str] = []

    def plan_grid(self, context: GridPlanningContext) -> GridPlan | None:
        return self.plan

    def solve(self, question, budget=None) -> SimResult:
        del budget
        question.reset()
        self.seen_legal = question.legal_actions()
        return SimResult()


def _context() -> GridPlanningContext:
    return GridPlanningContext(
        cycle=1,
        mode="live",
        fallback_branch_count=10,
        fallback_refine_count=10,
        hard_max_branch_count=10,
        hard_max_refine_count=10,
    )


def test_grid_plan_accepts_arbitrary_integers_and_clamps_to_hard_caps():
    policy = _Planner(GridPlan(branch_count=17, refine_count=4, reason="wider"))
    resolved = resolve_grid_plan(policy, _context())
    assert resolved.requested is not None
    assert resolved.requested.branch_count == 17
    assert resolved.effective.branch_count == 10
    assert resolved.effective.refine_count == 4
    assert resolved.source == "policy"
    assert "branch_count_clamped" in resolved.clamp_reasons


def test_invalid_grid_plan_falls_back():
    policy = _Planner({"branch_count": 0, "refine_count": 2})  # type: ignore[arg-type]
    resolved = resolve_grid_plan(policy, _context())
    assert resolved.source == "fallback"
    assert resolved.effective == GridPlan(10, 10, reason="fallback grid")
    assert resolved.fallback_reason and resolved.fallback_reason.startswith("invalid_plan:")


def test_live_runner_creates_planned_grid_after_completed_history(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n", encoding="utf-8")
    archived = ws / "history" / "iter000"
    archived.mkdir(parents=True)
    (archived / "live_cycle_manifest.json").write_text(
        json.dumps({"cycle": 0, "explore": {"best_score": 0.5}}),
        encoding="utf-8",
    )

    policy = _Planner(GridPlan(branch_count=1, refine_count=0, reason="one root"))
    run = run_live(
        ws,
        agent=None,
        policy=policy,
        config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
        max_width=2,
        max_refine=1,
        prepare_workspace=False,
    )

    assert policy.seen_legal == ["b0_a0"]
    assert run.grid_plan["source"] == "policy"
    assert run.grid_plan["effective"]["branch_count"] == 1
    assert run.grid_plan["effective"]["refine_count"] == 0
    raw = json.loads((ws / "raw_scores.json").read_text(encoding="utf-8"))
    assert [node["id"] for node in raw["nodes"]] == ["b0_a0"]


def test_live_runner_uses_preset_grid_for_first_cycle(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n", encoding="utf-8")
    policy = _Planner(GridPlan(branch_count=1, refine_count=0, reason="must wait"))

    run = run_live(
        ws,
        agent=None,
        policy=policy,
        config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
        max_width=2,
        max_refine=1,
        prepare_workspace=False,
    )

    assert policy.seen_legal == ["b0_a0", "b1_a0"]
    assert run.grid_plan["source"] == "fallback"
    assert run.grid_plan["fallback_reason"] == "no_completed_history"


def test_replay_enforces_policy_grid_plan(mini_raw_trace_file: Path):
    class NarrowBaseline(ParallelRefineMethod):
        def plan_grid(self, context: GridPlanningContext) -> GridPlan | None:
            del context
            return GridPlan(branch_count=1, refine_count=0, reason="narrow replay")

    result = run_replay(NarrowBaseline({"max_width": 2, "max_refine": 1}), mini_raw_trace_file)
    assert result.order == ["b0_a0"]
    assert result.budget == 1


def test_replay_marks_unseen_expansion_out_of_support(mini_raw_trace_file: Path):
    policy = _Planner(GridPlan(branch_count=3, refine_count=1, reason="unsupported"))
    context = GridPlanningContext(
        cycle=0,
        mode="replay",
        fallback_branch_count=2,
        fallback_refine_count=1,
        hard_max_branch_count=2,
        hard_max_refine_count=1,
        trace_branch_count=2,
        trace_refine_count=1,
    )
    resolved = resolve_grid_plan(
        policy,
        context,
        support_branch_count=2,
        support_refine_count=1,
    )
    assert resolved.out_of_support
    assert resolved.effective.branch_count == 2


def _write_policy(path: Path, plan: tuple[int, int]) -> Path:
    path.write_text(
        "\n".join(
            [
                "from dream_rsi.policy.api import GridPlan, GridPlanningContext",
                "from dream_rsi.policy.baseline import ParallelRefineMethod",
                "class OptimalPolicy(ParallelRefineMethod):",
                "    def plan_grid(self, context: GridPlanningContext):",
                "        del context",
                f"        return GridPlan(branch_count={plan[0]}, refine_count={plan[1]}, reason='test')",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_evaluator_replays_effective_plan(mini_raw_trace_file: Path, tmp_path: Path):
    method = _write_policy(tmp_path / "narrow.py", (1, 0))
    out = evaluate_on_trace(method, mini_raw_trace_file, target_spec="run_best")
    assert out["status"] == "ok"
    assert out["grid_plan"]["effective"]["branch_count"] == 1
    assert out["heuristic"]["total_budget"] == 1


def test_evaluator_rejects_plan_outside_frozen_trace(
    mini_raw_trace_file: Path, tmp_path: Path
):
    method = _write_policy(tmp_path / "wide.py", (3, 1))
    out = evaluate_on_trace(method, mini_raw_trace_file, target_spec="run_best")
    assert out["status"] == "out_of_support"
    assert out["grid_plan"]["out_of_support"] is True
