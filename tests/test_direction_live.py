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

"""Global direction memory reaches live explore prompts without changing scheduling."""

from __future__ import annotations

import json
from pathlib import Path

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.learning.directions.schema import load_catalog
from dream_rsi.learning.directions.context import write_direction_context
from dream_rsi.policy import ParallelRefineMethod
from dream_rsi.policy.api import GridPlan
from dream_rsi.env.runner import run_live


class _GlobalPromptCapturingAgent:
    name = "global_capture"

    def __init__(self):
        self.prompts: dict[str, str] = {}

    def run(self, **kwargs):
        from dream_rsi.agents.base import AgentResult

        assert kwargs["stage"] == "explore"
        node_dir = kwargs["node_dir"]
        self.prompts[str(node_dir)] = kwargs["prompt"]
        (node_dir / "proposal.md").write_text("used global direction memory\n")
        program = node_dir / "artifact/program.py"
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text("x = 1\n")
        return AgentResult(status="ok")


def _catalog(path: Path) -> Path:
    data = {
        "schema_version": 1,
        "summary_id": "iter000",
        "task": "toy",
        "generated_from": {"cycles": ["iter000"], "attempt_count": 1},
        "directions": [],
        "candidates": [
            {
                "id": "dir_a",
                "title": "Mechanism A",
                "instruction": "Implement mechanism A.",
                "priority": 0.9,
                "reason": "First priority.",
                "origin": "new",
                "target_bottleneck": "Toy bottleneck A.",
            },
            {
                "id": "dir_b",
                "title": "Mechanism B",
                "instruction": "Implement mechanism B.",
                "priority": 0.8,
                "reason": "Second priority.",
                "origin": "new",
                "target_bottleneck": "Toy bottleneck B.",
            },
        ],
    }
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_global_direction_context_reaches_all_prompts(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    catalog = load_catalog(_catalog(ws / "directions/current.json"))
    write_direction_context(catalog, ws / "directions/current.md")
    agent = _GlobalPromptCapturingAgent()
    policy = ParallelRefineMethod({"max_width": 2, "max_refine": 1})

    run = run_live(
        ws,
        agent,
        policy=policy,
        config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
        max_width=2,
        max_refine=1,
        max_workers=2,
        direction_mode="global",
    )

    assert run.direction_mode == "global"
    assert run.direction_summary_id == "iter000"
    assert run.sim.order == ["b0_a0", "b1_a0", "b0_a1", "b1_a1"]
    assert len(agent.prompts) == 4
    for prompt in agent.prompts.values():
        assert "Cumulative exploration direction memory" in prompt
        assert "Mechanism A" in prompt
        assert "Mechanism B" in prompt


def test_global_catalog_size_cannot_cap_next_grid(tmp_path: Path):
    class _CandidateCapPlanner(ParallelRefineMethod):
        def __init__(self):
            super().__init__({"max_width": 3, "max_refine": 0})
            self.context_keys: set[str] = set()
            self.has_candidate_count = True

        def plan_grid(self, context):
            self.context_keys = set(context.to_dict())
            self.has_candidate_count = hasattr(context, "direction_candidate_count")
            return GridPlan(
                branch_count=3,
                refine_count=0,
                reason="test global catalog is not a grid cap",
            )

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    catalog = load_catalog(_catalog(ws / "directions/current.json"))
    write_direction_context(catalog, ws / "directions/current.md")
    archived = ws / "history" / "iter000"
    archived.mkdir(parents=True)
    (archived / "live_cycle_manifest.json").write_text(
        json.dumps({"cycle": 0, "explore": {"best_score": 0.5}}),
        encoding="utf-8",
    )
    policy = _CandidateCapPlanner()

    run = run_live(
        ws,
        _GlobalPromptCapturingAgent(),
        policy=policy,
        config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
        max_width=3,
        max_refine=0,
        max_workers=3,
        cycle=1,
        direction_mode="global",
    )

    assert "direction_candidate_count" not in policy.context_keys
    assert not policy.has_candidate_count
    assert run.grid_plan["effective"]["branch_count"] == 3


def test_global_direction_context_is_strict_when_missing(tmp_path: Path):
    import pytest

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    with pytest.raises(FileNotFoundError, match="global direction context missing"):
        run_live(
            ws,
            _GlobalPromptCapturingAgent(),
            policy=ParallelRefineMethod({"max_width": 1, "max_refine": 0}),
            config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
            max_width=1,
            max_refine=0,
            cycle=1,
            direction_mode="global",
        )


def test_global_mode_allows_only_the_unguided_e0_bootstrap(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    agent = _GlobalPromptCapturingAgent()
    run = run_live(
        ws,
        agent,
        policy=ParallelRefineMethod({"max_width": 1, "max_refine": 0}),
        config=ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
        max_width=1,
        max_refine=0,
        cycle=0,
        direction_mode="global",
    )
    assert run.direction_mode == "global"
    assert not run.direction_context_active
    assert "Cumulative exploration direction memory" not in next(iter(agent.prompts.values()))
