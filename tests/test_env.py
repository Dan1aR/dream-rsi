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

"""TraceEnv + trace store (live/replay parity)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dream_rsi.env import TraceEnv, run_live, run_replay
from dream_rsi.env.trace_store import TraceStore, load_raw_scores, scaffold_grid_nodes
from dream_rsi.policy import ParallelRefineMethod
from tests.conftest import materialize_node


def test_replay_policy_matches_run_replay(mini_raw_trace_file: Path):
    policy = ParallelRefineMethod({"max_width": 2, "max_refine": 1})
    direct = policy.solve(TraceEnv.replay(mini_raw_trace_file))
    via_runner = run_replay(policy, mini_raw_trace_file)
    assert via_runner.best == pytest.approx(direct.best)
    assert via_runner.order == direct.order


def test_replay_tracks_batch_parallelism_metrics(mini_raw_trace_file: Path):
    """The replay metric must distinguish two dependency waves from serial probes."""
    env = TraceEnv.replay(mini_raw_trace_file, max_workers=2)
    env.probe_batch(["b0_a0", "b1_a0"])
    env.probe_batch(["b0_a1", "b1_a1"])
    assert env.decision_rounds == 2
    assert env.effective_sequential_rounds == 2
    assert env.batch_sizes == [2, 2]

    serial = TraceEnv.replay(mini_raw_trace_file, max_workers=2)
    for cell_id in ("b0_a0", "b1_a0", "b0_a1", "b1_a1"):
        serial.probe(cell_id)
    assert serial.decision_rounds == 4
    assert serial.effective_sequential_rounds == 4
    assert serial.batch_sizes == [1, 1, 1, 1]

    worker_limited = TraceEnv.replay(mini_raw_trace_file, max_workers=1)
    worker_limited.probe_batch(["b0_a0", "b1_a0"])
    worker_limited.probe_batch(["b0_a1", "b1_a1"])
    assert worker_limited.decision_rounds == 2
    assert worker_limited.effective_sequential_rounds == 4


def test_replay_execution_trace_records_decisions_and_outcomes(mini_raw_trace_file: Path):
    env = TraceEnv.replay(mini_raw_trace_file, max_workers=2)
    env.probe_batch(["b0_a0", "b1_a0"])
    env.probe_batch(["b0_a1", "b1_a1"])

    trace = env.execution_trace()
    assert trace["schema_version"] == 1
    assert trace["max_parallelism"] == 2
    assert len(trace["events"]) == 2

    first = trace["events"][0]
    assert first["before"] == {
        "observed_count": 0,
        "budget": 0,
        "best": 0.0,
        "legal_count": 2,
        "root_count": 2,
        "opened_branches": [],
    }
    assert first["batch"] == [
        {"branch": 0, "attempt": 0, "seq": 1, "tags": ["explore"]},
        {"branch": 1, "attempt": 0, "seq": 1, "tags": ["explore"]},
    ]
    assert [outcome["score"] for outcome in first["outcomes"]] == [0.5, 0.3]
    assert first["parallelism"]["batch_size"] == 2
    assert first["parallelism"]["effective_sequential_rounds"] == 1

    final = trace["final"]
    assert final["exhausted"] is True
    assert final["decision_rounds"] == 2
    assert final["effective_sequential_rounds"] == 2
    assert final["curve"][-1] == {"budget": 4, "best": 0.9}


def test_live_replay_parity(tmp_path: Path, mini_raw_trace_dict: dict):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n", encoding="utf-8")
    raw_path = tmp_path / "raw_scores.json"
    raw_path.write_text(json.dumps(mini_raw_trace_dict), encoding="utf-8")
    golden = load_raw_scores(raw_path)

    def fake_executor(env: TraceEnv, cell_id: str) -> None:
        node = golden.nodes[next(i for i, n in enumerate(golden.nodes) if n.id == cell_id)]
        materialize_node(ws, node)

    policy = ParallelRefineMethod({"max_width": 2, "max_refine": 1})
    live_run = run_live(
        ws,
        agent=None,
        policy=policy,
        max_width=2,
        max_refine=1,
        live_executor=fake_executor,
        prepare_workspace=False,
    )
    replay_run = run_replay(policy, raw_path)
    assert live_run.sim.best == pytest.approx(replay_run.best)
    assert live_run.sim.order == replay_run.order
    assert (ws / "raw_scores.json").is_file()


def test_trace_store_refresh(tmp_path: Path, mini_raw_trace_dict: dict):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy\n", encoding="utf-8")
    raw_path = tmp_path / "raw.json"
    raw_path.write_text(json.dumps(mini_raw_trace_dict), encoding="utf-8")
    golden = load_raw_scores(raw_path)

    store = TraceStore.for_live(ws, max_width=2, max_refine=1)
    materialize_node(ws, golden.nodes[0])
    store.refresh_from_workspace(ws)
    assert store.get_node(golden.nodes[0].id).raw.evaluated


def test_replay_semantics(mini_raw_trace_file: Path):
    env = TraceEnv.replay(mini_raw_trace_file)
    assert env.baseline_score == pytest.approx(0.4)
    assert set(env.legal_actions()) == {"b0_a0", "b1_a0"}
    with pytest.raises(ValueError, match="illegal"):
        env.probe("b0_a1")
    obs, branch, done = env.probe_new()
    assert branch == 0 and obs.cell_id == "b0_a0" and not done
    obs, done = env.probe_more(branch)
    assert obs.score == pytest.approx(0.7) and done


def test_scaffold_grid():
    nodes = scaffold_grid_nodes(Path("/tmp/ws"), max_width=2, max_refine=1)
    assert {n.id for n in nodes} == {"b0_a0", "b1_a0", "b0_a1", "b1_a1"}


def test_replay_excludes_unevaluated_tail(tmp_path: Path, mini_raw_trace_dict: dict):
    """Unevaluated scaffold at branch tail must not appear in replay legal_actions."""
    data = dict(mini_raw_trace_dict)
    data["nodes"] = list(data["nodes"]) + [
        {
            "id": "b0_a2",
            "branch": 0,
            "attempt": 2,
            "parent_id": "b0_a1",
            "intent": "refine",
            "tags": ["refine"],
            "path": "/tmp/mini/b0_a2",
            "created_at": "2026-01-01T00:00:02+00:00",
            "status": "pending",
            "raw": {
                "evaluated": False,
                "combined_score": None,
                "n_valid": None,
                "n_total": 1,
                "sol_times_ms": [None],
                "valids": [None],
                "max_gaps": [None],
                "error": None,
            },
        }
    ]
    raw_path = tmp_path / "raw_scores.json"
    raw_path.write_text(json.dumps(data), encoding="utf-8")

    env = TraceEnv.replay(raw_path)
    env.probe("b0_a0")
    env.probe("b0_a1")
    assert "b0_a2" not in env.legal_actions()
    assert not any(env.meta(a).branch == 0 for a in env.legal_actions())

    result = ParallelRefineMethod({"max_width": 1, "max_refine": 5}).solve(
        TraceEnv.replay(raw_path)
    )
    assert result.budget == 2
    assert result.order == ["b0_a0", "b0_a1"]
