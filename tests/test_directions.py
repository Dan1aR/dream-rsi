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

"""Direction catalog, collection, and summarization tests."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from dream_rsi.learning.directions.collect import collect_attempts
from dream_rsi.learning.directions.schema import AttemptExtraction, DirectionCatalog, load_catalog
from dream_rsi.learning.directions.summarize import infer_next_cycle, run_direction_summary
from dream_rsi.learning.directions.context import (
    load_global_direction_context,
    render_direction_context,
    resolve_direction_mode,
)
from dream_rsi.learning.directions.metrics import compute_direction_metrics
from dream_rsi.learning.directions.validate import validate_catalog


def _catalog_dict() -> dict:
    return {
        "schema_version": 1,
        "summary_id": "iter000",
        "task": "toy",
        "generated_from": {"cycles": ["iter000"], "attempt_count": 2},
        "directions": [
            {
                "id": "dir_existing",
                "canonical_name": "existing",
                "title": "Existing mechanism",
                "description": "A previously explored mechanism.",
                "target_bottleneck": "A toy bottleneck.",
                "status": "underexplored",
                "intensity": {
                    "n_roots": 1,
                    "n_refinements": 0,
                    "n_rounds": 1,
                    "n_successes": 1,
                    "n_failures": 0,
                    "n_improvements": 1,
                },
                "outcomes": {"best_score": 0.5, "latest_score": 0.5},
            }
        ],
        "candidates": [
            {
                "id": "candidate_b",
                "title": "Lower priority",
                "instruction": "Implement mechanism B.",
                "priority": 0.4,
                "reason": "Useful secondary option.",
                "origin": "new",
                "target_bottleneck": "Toy bottleneck B.",
            },
            {
                "id": "candidate_a",
                "title": "Higher priority",
                "instruction": "Implement mechanism A.",
                "priority": 0.9,
                "reason": "Best unused direction.",
                "origin": "new",
                "target_bottleneck": "Toy bottleneck A.",
                "avoid_repeating": ["existing mechanism"],
                "evidence_attempts": ["iter000/b0_a0"],
            },
        ],
    }


def _write_catalog(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_catalog_dict()), encoding="utf-8")
    return path


def test_catalog_validation_and_duplicate_ids(tmp_path: Path):
    catalog = load_catalog(_write_catalog(tmp_path / "catalog.json"))
    assert catalog.summary_id == "iter000"
    assert [item.id for item in catalog.candidates] == ["candidate_b", "candidate_a"]

    data = _catalog_dict()
    data["candidates"][0]["id"] = "dir_existing"
    with pytest.raises(ValueError, match="duplicate direction ids"):
        validate_catalog(DirectionCatalog.from_dict(data))


def test_collects_archived_and_current_attempts(mini_workspace: Path):
    ws = mini_workspace
    archived = ws / "history" / "iter000"
    archived.mkdir(parents=True)
    current = ws / "explore_b0" / "attempt_00"
    target = archived / "explore_b0" / "attempt_00"
    target.mkdir(parents=True)
    # Copy just enough files to create a historical record while retaining current.
    target.joinpath("meta.json").write_bytes(current.joinpath("meta.json").read_bytes())
    target.joinpath("proposal.md").write_text("historical proposal\n")
    (target / "eval").mkdir()
    (target / "eval" / "score.json").write_bytes(
        (current / "eval" / "score.json").read_bytes()
    )

    attempts = collect_attempts(ws, current_cycle=1)
    ids = {item.attempt_id for item in attempts}
    assert "iter000/b0_a0" in ids
    assert "iter001/b0_a0" in ids
    assert "iter001/b1_a0" in ids


class _SummaryAgent:
    name = "summary_fake"

    @staticmethod
    def _path(prompt: str, label: str) -> Path:
        match = re.search(rf"^{re.escape(label)}: (.+)$", prompt, re.MULTILINE)
        assert match is not None
        return Path(match.group(1).strip())

    def run(self, **kwargs):
        from dream_rsi.agents.base import AgentResult

        prompt = kwargs["prompt"]
        assert kwargs["stage"] == "learning"
        input_path = self._path(prompt, "Input JSON")
        output_path = self._path(prompt, "Required output JSON")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if "Mode: extract" in prompt:
            attempts = json.loads(input_path.read_text(encoding="utf-8"))
            rows = []
            for item in attempts:
                rows.append(
                    {
                        "attempt_id": item["attempt_id"],
                        "cycle": item["cycle"],
                        "branch": item["branch"],
                        "attempt": item["attempt"],
                        "parent_attempt_id": item["parent_attempt_id"],
                        "assigned_direction_id": None,
                        "primary_mechanism": "toy mechanism",
                        "supporting_mechanisms": [],
                        "outcome_class": "success",
                        "score": item["score"],
                        "improved_over_parent": None,
                        "evidence": "toy evidence",
                    }
                )
            output_path.write_text(json.dumps(rows), encoding="utf-8")
        else:
            summary_id = re.search(r"^Summary id: (.+)$", prompt, re.MULTILINE).group(1)
            task = re.search(r"^Task: (.+)$", prompt, re.MULTILINE).group(1)
            data = _catalog_dict()
            data["summary_id"] = summary_id
            data["task"] = task
            output_path.write_text(json.dumps(data), encoding="utf-8")
        return AgentResult(status="ok")


class _PartialSummaryAgent(_SummaryAgent):
    """Simulate a valid extraction response that accidentally omits one row."""

    def run(self, **kwargs):
        result = super().run(**kwargs)
        prompt = kwargs["prompt"]
        if "Mode: extract" in prompt:
            output_path = self._path(prompt, "Required output JSON")
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            output_path.write_text(json.dumps(rows[:-1]), encoding="utf-8")
        return result


class _DroppedCanonicalAgent(_SummaryAgent):
    """Simulate an aggregate response that omits an old canonical direction."""

    def run(self, **kwargs):
        result = super().run(**kwargs)
        prompt = kwargs["prompt"]
        if "Mode: aggregate" in prompt:
            output_path = self._path(prompt, "Required output JSON")
            data = json.loads(output_path.read_text(encoding="utf-8"))
            data["directions"] = []
            data["candidates"] = []
            output_path.write_text(json.dumps(data), encoding="utf-8")
        return result


class _TransientAggregateFailureAgent(_SummaryAgent):
    """Fail the first aggregate call as Gemini does after retry exhaustion."""

    def __init__(self) -> None:
        self.aggregate_calls = 0

    def run(self, **kwargs):
        prompt = kwargs["prompt"]
        if "Mode: aggregate" in prompt:
            self.aggregate_calls += 1
            if self.aggregate_calls == 1:
                from dream_rsi.agents.base import AgentResult

                return AgentResult(
                    status="failed",
                    message="gemini empty/invalid stream after 3 attempts",
                )
        return super().run(**kwargs)


def test_direction_summary_promotes_valid_catalog(mini_workspace: Path):
    result = run_direction_summary(mini_workspace, _SummaryAgent(), cycle=0, task="toy")
    assert result.ok, result.error
    assert result.n_attempts == 2
    assert result.n_extracted == 2
    catalog = load_catalog(Path(result.catalog_path))
    assert catalog.summary_id == "iter000"
    assert result.context_path is not None
    context_path = Path(result.context_path)
    assert context_path == mini_workspace / "directions/current.md"
    context, loaded = load_global_direction_context(mini_workspace)
    assert loaded.summary_id == "iter000"
    assert "Cumulative exploration direction memory" in context
    assert "Higher priority" in context
    assert (mini_workspace / "directions/extractions/iter000.jsonl").is_file()
    assert (mini_workspace / "directions/history/iter000.json").is_file()

    resumed = run_direction_summary(mini_workspace, _SummaryAgent(), cycle=0, task="toy")
    assert resumed.ok, resumed.error
    assert resumed.summary_id == "iter000_r001"
    assert (mini_workspace / "directions/history/iter000_r001.json").is_file()
    assert infer_next_cycle(mini_workspace) == 1


def test_direction_summary_warns_but_promotes_when_catalog_drops_canonical_ids(
    mini_workspace: Path,
):
    initial = run_direction_summary(mini_workspace, _SummaryAgent(), cycle=0, task="toy")
    assert initial.ok, initial.error

    with pytest.warns(RuntimeWarning, match="dropped canonical ids: dir_existing"):
        result = run_direction_summary(
            mini_workspace, _DroppedCanonicalAgent(), cycle=1, task="toy"
        )

    assert result.ok, result.error
    catalog = load_catalog(Path(result.catalog_path))
    assert catalog.directions == []


def test_direction_summary_retries_transient_aggregate_agent_failure(
    mini_workspace: Path,
):
    agent = _TransientAggregateFailureAgent()

    result = run_direction_summary(mini_workspace, agent, cycle=0, task="toy")

    assert result.ok, result.error
    assert agent.aggregate_calls == 2
    assert result.metrics["aggregation"] == {"attempt_count": 2, "retry_count": 1}
    assert (
        mini_workspace
        / "directions/runs/iter000/aggregate/retry_001/catalog.proposed.json"
    ).is_file()


def test_direction_summary_promotes_partial_extractions_and_retries_gap(
    mini_workspace: Path,
):
    partial = run_direction_summary(
        mini_workspace, _PartialSummaryAgent(), cycle=0, task="toy"
    )
    assert partial.ok, partial.error
    assert partial.n_attempts == 2
    assert partial.n_extracted == 1
    assert partial.metrics["coverage"] == {
        "available_attempt_count": 2,
        "extracted_attempt_count": 1,
        "missing_attempt_count": 1,
        "complete": False,
        "failed_batch_count": 0,
    }
    context, catalog = load_global_direction_context(mini_workspace)
    assert catalog.generated_from["coverage_complete"] is False
    assert "Coverage: partial (1/2 extracted; 1 pending)." in context

    resumed = run_direction_summary(mini_workspace, _SummaryAgent(), cycle=0, task="toy")
    assert resumed.ok, resumed.error
    assert resumed.n_extracted == 1
    assert resumed.metrics["coverage"]["complete"] is True


def test_direction_summary_extracts_independent_batches_concurrently(
    mini_workspace: Path, monkeypatch
):
    """Batch size controls prompt size; workers control parallel extraction."""
    from dream_rsi.learning.directions import extract as direction_extract

    monkeypatch.setenv("DRSI_DIRECTION_EXTRACT_BATCH", "1")
    monkeypatch.setenv("DRSI_DIRECTION_EXTRACT_WORKERS", "2")
    real_extract = direction_extract.extract_batch
    barrier = threading.Barrier(2, timeout=3)
    called_dirs: list[Path] = []
    called_lock = threading.Lock()

    def synchronized_extract(*args, **kwargs):
        # With serial execution this times out before batch two can start.
        barrier.wait()
        with called_lock:
            called_dirs.append(kwargs["run_dir"])
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(direction_extract, "extract_batch", synchronized_extract)
    result = run_direction_summary(mini_workspace, _SummaryAgent(), cycle=0, task="toy")

    assert result.ok, result.error
    assert result.n_extracted == 2
    assert sorted(path.name for path in called_dirs) == ["extract_0000", "extract_0001"]


def test_direction_context_renderer_and_mode_resolution(tmp_path: Path, monkeypatch):
    catalog = load_catalog(_write_catalog(tmp_path / "catalog.json"))
    context = render_direction_context(catalog)
    assert "Cumulative exploration direction memory" in context
    assert "Existing mechanism" in context
    assert "Higher priority" in context
    assert resolve_direction_mode("global") == "global"
    monkeypatch.delenv("DRSI_DIRECTION_MODE", raising=False)
    assert resolve_direction_mode() == "off"
    monkeypatch.setenv("DRSI_DIRECTION_MODE", "global")
    assert resolve_direction_mode() == "global"
    assert resolve_direction_mode("off") == "off"
    with pytest.raises(ValueError, match="invalid direction mode"):
        resolve_direction_mode("assigned")


def _extraction(cycle: str, branch: int, attempt: int, mechanism: str) -> AttemptExtraction:
    return AttemptExtraction(
        attempt_id=f"{cycle}/b{branch}_a{attempt}",
        cycle=cycle,
        branch=branch,
        attempt=attempt,
        parent_attempt_id=None,
        assigned_direction_id=None,
        primary_mechanism=mechanism,
        supporting_mechanisms=[],
        outcome_class="success",
        score=None,
        improved_over_parent=None,
        evidence="test",
    )


def test_direction_metrics_count_duplicate_root_mechanisms():
    extractions = [
        _extraction("iter001", 0, 0, "same"),
        _extraction("iter001", 1, 0, "same"),
        _extraction("iter001", 2, 0, "different"),
        _extraction("iter001", 3, 0, "unknown_no_mechanism"),
        _extraction("iter001", 0, 1, "refinement"),
        _extraction("iter000", 0, 0, "older cycle"),
    ]
    metrics = compute_direction_metrics(cycle="iter001", extractions=extractions)
    assert metrics["n_roots"] == 4
    assert metrics["actual"]["count"] == 3
    assert metrics["actual"]["unique"] == 2
    assert metrics["actual"]["duplicate_rate"] == pytest.approx(1 / 3)


def test_learning_job_copies_valid_direction_catalog(
    mini_workspace: Path, tmp_path: Path
):
    from dream_rsi.learning.allocate import allocate_learning_workspace

    _write_catalog(mini_workspace / "directions/current.json")
    job = allocate_learning_workspace(
        mini_workspace,
        dest=tmp_path / "learning/job",
    )
    copied = job / "directions/direction_summary.json"
    assert copied.is_file()
    assert load_catalog(copied).summary_id == "iter000"
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["direction_summary"] == "directions/direction_summary.json"
