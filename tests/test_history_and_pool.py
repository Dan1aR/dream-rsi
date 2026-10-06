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

"""History seeding for explore + trace pool for multi-trace discover eval."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dream_rsi.exploration.history_prune import (
    iter_archive_global_best,
    maybe_prune_archived_iter,
    prior_iters_global_best,
)
from dream_rsi.exploration.history_seed import list_workspace_archives, seed_history_to_branches
from dream_rsi.learning.heuristic_search.eval import evaluate_multi_trace
from dream_rsi.learning.trace_pool import (
    append_workspace_trace,
    list_pool_traces,
    sync_workspace_pool_to_job,
    workspace_trace_pool,
)
from dream_rsi.orchestrate import _archive_and_reset_workspace, write_live_cycle_manifest
from dream_rsi.policy.constants import PROMOTED_POLICY_REL


def test_list_workspace_archives(tmp_path: Path):
    ws = tmp_path / "ws"
    (ws / "history" / "iter000").mkdir(parents=True)
    (ws / "history" / "manual_x").mkdir()
    assert [p.name for p in list_workspace_archives(ws)] == ["iter000", "manual_x"]


def test_seed_history_symlink(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRSI_HISTORY_SEED_MODE", "symlink")
    ws = tmp_path / "ws"
    arch = ws / "history" / "iter000"
    arch.mkdir(parents=True)
    (arch / "marker.txt").write_text("prior cycle\n", encoding="utf-8")

    touched = seed_history_to_branches(ws, max_width=2, max_refine=1)
    assert len(touched) == 2

    link = ws / "explore_b0" / "history" / "iter000"
    assert link.is_symlink()
    assert (link / "marker.txt").read_text(encoding="utf-8") == "prior cycle\n"


def test_seed_history_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRSI_HISTORY_SEED_MODE", "copy")
    ws = tmp_path / "ws"
    arch = ws / "history" / "iter000"
    arch.mkdir(parents=True)
    (arch / "marker.txt").write_text("copy me\n", encoding="utf-8")

    seed_history_to_branches(ws, max_width=1, max_refine=1)
    dst = ws / "explore_b0" / "history" / "iter000"
    assert dst.is_dir() and not dst.is_symlink()
    assert (dst / "marker.txt").read_text(encoding="utf-8") == "copy me\n"


def test_trace_pool_append_and_sync(tmp_path: Path, mini_raw_trace_file: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    job = tmp_path / "learning" / "job1"
    job.mkdir(parents=True)

    (ws / "raw_scores.json").write_bytes(mini_raw_trace_file.read_bytes())
    live_manifest = {"cycle": 0, "policy": {"beta_default": 0.6}}
    (ws / "live_cycle_manifest.json").write_text(
        json.dumps(live_manifest), encoding="utf-8"
    )
    append_workspace_trace(ws, "iter000", ws / "raw_scores.json")

    pool = sync_workspace_pool_to_job(ws, job)
    names = [p.parent.name for p in list_pool_traces(pool)]
    assert "iter000" in names
    assert "_current" in names
    assert (pool / "manifest.json").is_file()
    assert json.loads(
        (pool / "iter000/live_cycle_manifest.json").read_text(encoding="utf-8")
    ) == live_manifest
    assert json.loads(
        (pool / "_current/live_cycle_manifest.json").read_text(encoding="utf-8")
    ) == live_manifest


def test_live_cycle_manifest_records_and_archives_with_trace(
    tmp_path: Path, mini_raw_trace_file: Path
):
    ws = tmp_path / "ws"
    policy_path = ws / PROMOTED_POLICY_REL
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("# promoted policy\n", encoding="utf-8")
    policy = SimpleNamespace(NAME="OptimalPolicy", beta=0.6, config={"max_width": 2})
    sim = SimpleNamespace(
        best=0.9,
        budget=4,
        decision_rounds=2,
        effective_sequential_rounds=2,
        batch_sizes=[2, 2],
    )
    run = SimpleNamespace(
        sim=sim,
        grid_plan={
            "source": "policy",
            "effective": {"branch_count": 2, "refine_count": 1},
        },
        direction_mode="off",
        direction_summary_id=None,
        direction_context_active=False,
    )
    manifest_path = write_live_cycle_manifest(
        ws,
        cycle=0,
        policy_source=str(policy_path),
        policy=policy,
        run=run,
        max_workers=2,
        max_width=None,
        max_refine=None,
        max_steps=4,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"]["beta_default"] == pytest.approx(0.6)
    assert manifest["explore"]["max_steps"] == 4
    assert manifest["explore"]["direction_mode"] == "off"
    assert manifest["policy"]["source"] == str(policy_path.resolve())
    assert manifest["policy"]["sha256"]
    assert manifest["explore"]["best_score"] == pytest.approx(0.9)
    assert manifest["grid_plan"]["effective"] == {
        "branch_count": 2,
        "refine_count": 1,
    }

    (ws / "raw_scores.json").write_bytes(mini_raw_trace_file.read_bytes())
    archived = _archive_and_reset_workspace(ws, cycle=0)
    assert archived is not None
    archive = Path(archived)
    assert (archive / "raw_scores.json").is_file()
    assert (
        json.loads((archive / "live_cycle_manifest.json").read_text(encoding="utf-8"))
        == manifest
    )
    assert json.loads(
        (ws / "trace_pool/iter000/live_cycle_manifest.json").read_text(encoding="utf-8")
    ) == manifest
    # The promoted policy outlives the per-cycle reset.
    assert policy_path.is_file()


def test_evaluate_multi_trace(
    mini_raw_trace_file: Path, baseline_policy_file: Path, tmp_path: Path
):
    pool = tmp_path / "pool"
    for name in ("iter000", "iter001"):
        dest = pool / name / "raw_scores.json"
        dest.parent.mkdir(parents=True)
        dest.write_bytes(mini_raw_trace_file.read_bytes())

    out = evaluate_multi_trace(
        baseline_policy_file, list_pool_traces(pool), target_spec="run_best"
    )
    assert out["trace_pool"] is True
    assert out["aggregate"]["n_traces"] == 2
    assert out["status"] == "ok"
    assert out["aggregate"]["n_reached"] == 2
    assert out["heuristic"]["reached_target"] is True
    assert out["heuristic"]["best"] == pytest.approx(0.9)
    assert out["heuristic"]["total_budget"] == 4


def _write_attempt(archive: Path, branch: str, attempt: str, score: float) -> None:
    attempt_dir = archive / branch / attempt / "eval"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "score.json").write_text(
        json.dumps({"score": score}),
        encoding="utf-8",
    )


def test_prune_iter_archive_prev_global_best(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRSI_HISTORY_ARCHIVE_FILTER", "prev_global_best")
    ws = tmp_path / "ws"
    iter000 = ws / "history" / "iter000"
    iter001 = ws / "history" / "iter001"
    _write_attempt(iter000, "explore_b0", "attempt_00", 0.50)
    _write_attempt(iter000, "explore_b1", "attempt_02", 0.70)
    _write_attempt(iter001, "explore_b0", "attempt_00", 0.60)
    _write_attempt(iter001, "explore_b0", "attempt_01", 0.65)
    _write_attempt(iter001, "explore_b1", "attempt_00", 0.68)
    _write_attempt(iter001, "explore_b2", "attempt_00", 0.75)

    stats = maybe_prune_archived_iter(ws, iter001, cycle=1)
    assert stats is not None
    assert stats["threshold"] == 0.70
    assert stats["n_kept"] == 1
    assert stats["n_removed"] == 3
    assert (iter001 / "explore_b2" / "attempt_00").is_dir()
    assert not (iter001 / "explore_b0").exists()
    assert not (iter001 / "explore_b1").exists()


def test_prune_skips_first_iter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DRSI_HISTORY_ARCHIVE_FILTER", "prev_global_best")
    ws = tmp_path / "ws"
    iter000 = ws / "history" / "iter000"
    _write_attempt(iter000, "explore_b0", "attempt_00", 0.40)

    stats = maybe_prune_archived_iter(ws, iter000, cycle=0)
    assert stats is None
    assert (iter000 / "explore_b0" / "attempt_00").is_dir()


def test_iter_archive_global_best(tmp_path: Path):
    arch = tmp_path / "iter000"
    _write_attempt(arch, "explore_b0", "attempt_00", 0.41)
    _write_attempt(arch, "explore_b3", "attempt_01", 0.82)
    assert iter_archive_global_best(arch) == 0.82


def test_prior_iters_global_best_uses_cumulative_max(tmp_path: Path):
    ws = tmp_path / "ws"
    _write_attempt(ws / "history" / "iter000", "explore_b0", "attempt_00", 0.70)
    _write_attempt(ws / "history" / "iter001", "explore_b0", "attempt_00", 0.60)
    assert prior_iters_global_best(ws, cycle=2) == 0.70


def test_prune_when_prev_iter_empty_but_earlier_best_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Bug #1: iter001 pruned empty must still compare iter002 against iter000 best."""
    monkeypatch.setenv("DRSI_HISTORY_ARCHIVE_FILTER", "prev_global_best")
    ws = tmp_path / "ws"
    iter000 = ws / "history" / "iter000"
    iter001 = ws / "history" / "iter001"
    iter002 = ws / "history" / "iter002"
    _write_attempt(iter000, "explore_b0", "attempt_00", 0.70)
    # iter001 had nothing above 0.70 — no attempts left after its own archive prune.
    _write_attempt(iter002, "explore_b0", "attempt_00", 0.68)
    _write_attempt(iter002, "explore_b1", "attempt_00", 0.72)

    stats = maybe_prune_archived_iter(ws, iter002, cycle=2)
    assert stats is not None
    assert stats["threshold"] == 0.70
    assert stats["n_kept"] == 1
    assert (iter002 / "explore_b1" / "attempt_00").is_dir()
    assert not (iter002 / "explore_b0").exists()
