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

"""Exploration: NodeStore + run_live workflow."""

from __future__ import annotations

import json
import os
import shutil
from unittest.mock import patch

import pytest

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.store import NodeStore, node_id
from dream_rsi.env.runner import run_live
from dream_rsi.env.trace_store import grid_limits


def test_node_store_lifecycle(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    store = NodeStore(ws)
    node_dir = store.create_node(branch=0, seq=1, intent="explore", tags=["explore"])
    assert store.read_meta(node_id(0, 1)).status == "draft"
    assert node_dir.name == "attempt_00"


def test_grid_limits_from_env():
    with patch.dict(os.environ, {"DRSI_PARALLEL_EXPLORE": "3", "DRSI_REFINE_PER_BRANCH": "3"}):
        assert grid_limits() == (3, 3)


class _SkipAgent:
    name = "skip"

    def run(self, **kwargs):
        from dream_rsi.agents.base import AgentResult

        node_dir = kwargs["node_dir"]
        (node_dir / "proposal.md").write_text("skip")
        prog = node_dir / "artifact" / "program.py"
        prog.parent.mkdir(parents=True, exist_ok=True)
        prog.write_text("x=1\n")
        return AgentResult(status="ok")


def test_run_live_baseline_grid(tmp_path):
    ws = tmp_path / "wf"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")

    config = ExploreConfig(evaluator=None, eval_program="artifact/program.py")
    with patch.dict(os.environ, {"DRSI_PARALLEL_EXPLORE": "2", "DRSI_REFINE_PER_BRANCH": "1"}):
        run = run_live(
            ws,
            _SkipAgent(),
            policy_source="baseline",
            config=config,
            max_workers=2,
            max_width=2,
            max_refine=1,
        )

    assert len(run.node_ids) + len(run.failed) == 4


def test_explore_run_cli_records_the_cycle_manifest(tmp_path, monkeypatch):
    # Later grid planning and learning read this manifest back from history/,
    # so a round explored with the CLI (e.g. a shared E0) must write it too.
    from dream_rsi.cli import explore as explore_cli

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    monkeypatch.setattr("dream_rsi.agents.load_agent", lambda name, **_: _SkipAgent())
    monkeypatch.setattr(
        explore_cli,
        "build_explore_config",
        lambda args: ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
    )
    args = explore_cli.build_parser().parse_args(
        ["run", "-w", str(ws), "--policy-source", "baseline", "--cycle", "3"]
        + ["--branches", "2", "--refine", "1", "--max-workers", "2"]
    )

    assert explore_cli.cmd_run(args) == 0

    manifest = json.loads((ws / "live_cycle_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cycle"] == 3
    assert manifest["policy"]["name"] == "parallel_refine"
    assert manifest["policy"]["config"] == {"max_width": 2, "max_refine": 1}
    assert manifest["explore"]["max_width"] == 2
    assert manifest["explore"]["probe_budget"] == 4
    assert manifest["grid_plan"]["effective"]["branch_count"] == 2


def test_explore_archive_cli_moves_the_live_round_to_history(tmp_path, monkeypatch, capsys):
    from dream_rsi.cli import explore as explore_cli

    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n")
    monkeypatch.setattr("dream_rsi.agents.load_agent", lambda name, **_: _SkipAgent())
    monkeypatch.setattr(
        explore_cli,
        "build_explore_config",
        lambda args: ExploreConfig(evaluator=None, eval_program="artifact/program.py"),
    )
    parser = explore_cli.build_parser()
    run = parser.parse_args(
        ["run", "-w", str(ws), "--policy-source", "baseline", "--branches", "2", "--refine", "0"]
    )
    assert explore_cli.cmd_run(run) == 0
    raw = (ws / "raw_scores.json").read_bytes()

    archive = parser.parse_args(["archive", "-w", str(ws), "--cycle", "0"])
    assert explore_cli.cmd_archive(archive) == 0

    dest = ws / "history" / "iter000"
    assert sorted(p.name for p in dest.glob("explore_b*")) == ["explore_b0", "explore_b1"]
    assert (dest / "raw_scores.json").read_bytes() == raw
    assert (dest / "live_cycle_manifest.json").is_file()
    assert not list(ws.glob("explore_b*")) and not (ws / "raw_scores.json").exists()
    # Nothing is live any more, and an archived round is never overwritten.
    assert explore_cli.cmd_archive(archive) == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.integration
def test_run_live_gemini_1x1_lasso(tmp_path, real_agent, lasso_config):
    ws = tmp_path / "gemini_1x1"
    ws.mkdir()
    shutil.copy2(lasso_config.problem, ws / "problem.txt")
    env = {
        "DRSI_PARALLEL_EXPLORE": "1",
        "DRSI_REFINE_PER_BRANCH": "1",
        "GEMINI_MAX_TURNS": os.environ.get("DRSI_TEST_GEMINI_MAX_TURNS", "12"),
    }
    with patch.dict(os.environ, env, clear=False):
        run = run_live(
            ws,
            real_agent,
            policy_source="baseline",
            config=lasso_config,
            max_workers=1,
            max_width=1,
            max_refine=1,
        )
    assert len(run.node_ids) == 2 and not run.failed
