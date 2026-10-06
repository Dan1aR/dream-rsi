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

"""Shared fixtures for DREAM-RSI module tests."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LASSO_TASK = ROOT / "tasks/lasso_path"

from dream_rsi.dotenv import load_env  # noqa: E402

load_env()


def _require_gemini() -> None:
    from dream_rsi.dotenv import gemini_ready

    if not gemini_ready():
        pytest.skip("GEMINI_API_KEY / GOOGLE_API_KEY not set")
    if shutil.which(os.environ.get("GEMINI_BIN", "gemini").strip() or "gemini") is None:
        pytest.skip("gemini CLI not found (set GEMINI_BIN or add gemini to PATH)")


def _require_lasso_build() -> None:
    if not (LASSO_TASK / "evaluator.py").is_file():
        pytest.skip("lasso_path task files missing: run scripts/fetch_tasks.sh lasso_path")
    if shutil.which("g++") is None:
        pytest.skip("g++ not available")
    eigen_core = LASSO_TASK / "eigen" / "Eigen" / "Core"
    if not eigen_core.is_file():
        pytest.skip(f"eigen headers missing: run {LASSO_TASK}/setup.sh")


@pytest.fixture
def replay_env(mini_raw_trace_file):
    from dream_rsi.env import TraceEnv

    return TraceEnv.replay(mini_raw_trace_file)


def materialize_node(ws: Path, raw_node) -> None:
    """Write one node dir so export_raw_scores can read it (fake live run)."""
    from dream_rsi.exploration.store import NodeStore

    store = NodeStore(ws)
    node_dir = store.branch_dir(raw_node.branch) / f"attempt_{raw_node.attempt:02d}"
    for sub in ("artifact", "logs", "eval", "agent_log"):
        (node_dir / sub).mkdir(parents=True, exist_ok=True)
    meta = {
        "id": raw_node.id,
        "parent_id": raw_node.parent_id,
        "branch": raw_node.branch,
        "seq": raw_node.attempt + 1,
        "cycle": 0,
        "intent": raw_node.intent,
        "status": raw_node.status or "done",
        "created_at": raw_node.created_at or "2026-01-01T00:00:00+00:00",
        "tags": list(raw_node.tags),
        "artifact_relpath": "artifact/main.md",
    }
    (node_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    raw = raw_node.raw
    score = {
        "combined_score": raw.combined_score,
        "n_valid": raw.n_valid,
        "n_total": raw.n_total,
        "sol_times_ms": raw.sol_times_ms,
        "valids": raw.valids,
        "max_gaps": raw.max_gaps,
        "error": raw.error,
    }
    (node_dir / "eval" / "score.json").write_text(json.dumps(score), encoding="utf-8")
    (node_dir / "proposal.md").write_text("ok\n", encoding="utf-8")
    (node_dir / "artifact" / "program.py").write_text("pass\n", encoding="utf-8")


@pytest.fixture
def real_agent():
    _require_gemini()
    from dream_rsi.agents import load_agent

    return load_agent(os.environ.get("DRSI_TEST_AGENT", "gemini"))


@pytest.fixture
def lasso_config():
    _require_lasso_build()
    from dream_rsi.exploration.config import ExploreConfig

    return ExploreConfig(
        problem=LASSO_TASK / "lasso_path.txt",
        evaluator=LASSO_TASK / "evaluator.py",
        init_program=LASSO_TASK / "init_program.py",
        eval_program="artifact/program.py",
        eval_timeout_sec=float(os.environ.get("DRSI_TEST_EVAL_TIMEOUT", "600")),
    )


@pytest.fixture
def mini_raw_trace_dict() -> dict:
    """2 branches × (explore + refine); run_best=0.9 on b1_a1."""
    return {
        "version": 1,
        "workspace": "mini",
        "exported_at": "2026-01-01T00:00:00+00:00",
        "n_problems": 1,
        "baseline": {
            "evaluated": True,
            "combined_score": 0.4,
            "n_valid": 1,
            "n_total": 1,
            "sol_times_ms": [None],
            "valids": [None],
            "max_gaps": [None],
            "error": None,
        },
        "nodes": [
            _node("b0_a0", 0, 0, None, 0.5),
            _node("b1_a0", 1, 0, None, 0.3),
            _node("b0_a1", 0, 1, "b0_a0", 0.7),
            _node("b1_a1", 1, 1, "b1_a0", 0.9),
        ],
    }


def _node(nid: str, branch: int, attempt: int, parent_id: str | None, score: float) -> dict:
    return {
        "id": nid,
        "branch": branch,
        "attempt": attempt,
        "parent_id": parent_id,
        "intent": "explore" if attempt == 0 else "refine",
        "tags": ["explore"] if attempt == 0 else ["refine"],
        "path": f"/tmp/mini/{nid}",
        "created_at": f"2026-01-01T00:00:{attempt:02d}+00:00",
        "status": "done",
        "raw": {
            "evaluated": True,
            "combined_score": score,
            "n_valid": 1,
            "n_total": 1,
            "sol_times_ms": [None],
            "valids": [None],
            "max_gaps": [None],
            "error": None,
        },
    }


@pytest.fixture
def mini_raw_trace_file(tmp_path: Path, mini_raw_trace_dict: dict) -> Path:
    path = tmp_path / "raw_scores.json"
    path.write_text(json.dumps(mini_raw_trace_dict), encoding="utf-8")
    return path


@pytest.fixture
def baseline_policy_file(tmp_path: Path) -> Path:
    """An ``OptimalPolicy`` file that replays the 2x2 parallel-refine baseline."""
    from dream_rsi.learning.heuristic_search.workflow import baseline_incumbent_source

    path = tmp_path / "baseline_policy.py"
    path.write_text(
        baseline_incumbent_source({"max_width": 2, "max_refine": 1}), encoding="utf-8"
    )
    return path


@pytest.fixture
def mini_trace(mini_raw_trace_file: Path):
    from dream_rsi.env.trace.raw import trace_from_raw_file

    return trace_from_raw_file(mini_raw_trace_file)


@pytest.fixture
def mini_workspace(tmp_path: Path) -> Path:
    """Minimal explore workspace: 2 branches, one attempt each, with scores."""
    ws = tmp_path / "mini_ws"
    ws.mkdir()
    (ws / "problem.txt").write_text("toy problem\n", encoding="utf-8")

    baseline = ws / ".baseline"
    (baseline / "eval").mkdir(parents=True)
    (baseline / "eval" / "score.json").write_text(
        json.dumps({"combined_score": 0.4, "n_valid": 1, "n_total": 1}),
        encoding="utf-8",
    )

    specs = [
        ("b0_a0", 0, 1, None, 0.5),
        ("b1_a0", 1, 1, None, 0.3),
    ]
    attempts = []
    for nid, branch, seq, parent_id, score in specs:
        node_dir = ws / f"explore_b{branch}" / f"attempt_{seq - 1:02d}"
        for sub in ("artifact", "logs", "eval", "agent_log"):
            (node_dir / sub).mkdir(parents=True)
        meta = {
            "id": nid,
            "parent_id": parent_id,
            "branch": branch,
            "seq": seq,
            "cycle": 0,
            "intent": "explore",
            "status": "done",
            "created_at": "2026-01-01T00:00:00+00:00",
            "tags": ["explore"],
            "artifact_relpath": "artifact/main.md",
        }
        (node_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        (node_dir / "eval" / "score.json").write_text(
            json.dumps({"combined_score": score, "n_valid": 1, "n_total": 1}),
            encoding="utf-8",
        )
        attempts.append(
            {
                "id": nid,
                "branch": branch,
                "seq": seq,
                "path": str(node_dir),
                "parent_id": parent_id,
                "intent": "explore",
                "cycle": 0,
                "tags": ["explore"],
            }
        )

    (ws / "history.json").write_text(
        json.dumps({"count": len(attempts), "attempts": attempts}),
        encoding="utf-8",
    )
    return ws
