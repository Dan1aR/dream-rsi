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

"""Offline learning: pipeline, discover, allocate, incumbent selection."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from dream_rsi import orchestrate as orchestrate_module
from dream_rsi.learning.allocate import (
    allocate_learning_workspace,
    refresh_learning_baseline_preset,
)
from dream_rsi.learning.discover import (
    EVAL_SCRIPT,
    DiscoverConfig,
    baseline_benchmark,
    build_workflow_config,
    paths_for_learning_job,
)
from dream_rsi.learning.heuristic_search.eval import (
    _pareto_summary,
    _write_execution_traces,
    run_beta_sweep,
)
from dream_rsi.learning.heuristic_search import workflow as workflow_module
from dream_rsi.learning.heuristic_search.workflow import (
    BASELINE_INCUMBENT_HEADER,
    WorkflowConfig,
    _seeded_method_text,
    _select_best_round,
    archive_round,
    baseline_incumbent_source,
    run_workflow,
)
from dream_rsi.learning.pipeline import (
    format_learning_result,
    load_raw,
    resolve_baseline_config,
    run_learning,
)
from dream_rsi.orchestrate import PipelineConfig, _discover_ok, _grid_caps
from dream_rsi.policy.constants import METHOD_REL, METHOD_TEMPLATE_REL, PROMOTED_POLICY_REL

TEMPLATE_TEXT = "# template\n"


def test_run_learning_on_mini_trace(mini_raw_trace_file, baseline_policy_file):
    result = run_learning(
        trace_source=mini_raw_trace_file,
        export=False,
        method_file=baseline_policy_file,
    )
    assert result.trace.n_nodes == 4
    assert result.parallel_refine.ok is True
    assert result.discovered.ok is True
    assert result.parallel_refine.hit_run_best is True
    assert result.discovered.hit_run_best is True
    # The generated incumbent replays exactly the parallel-refine baseline.
    assert result.discovered.budget == result.parallel_refine.budget
    assert result.discovered.order_preview == result.parallel_refine.order_preview

    text = format_learning_result(result)
    assert "parallel_refine" in text
    assert "OptimalPolicy" in text


def test_run_learning_skips_discovered_policy_without_a_file(mini_raw_trace_file):
    result = run_learning(trace_source=mini_raw_trace_file, export=False)
    assert result.parallel_refine.ok is True
    assert result.discovered is None
    assert "[skip] OptimalPolicy" in format_learning_result(result)


def test_run_learning_requires_source():
    with pytest.raises(ValueError, match="workspace or trace_source"):
        run_learning()


def test_baseline_benchmark_runs_parallel_refine(mini_raw_trace_file):
    out = baseline_benchmark(mini_raw_trace_file)
    assert out["ok"] is True
    assert out["error"] is None
    assert out["run_best"] == pytest.approx(0.9)
    assert out["budget"] > 0
    assert out["best"] > 0
    assert out["config"] == {"max_width": 2, "max_refine": 1}


def test_baseline_config_infers_trace_grid_and_keeps_explicit_overrides(
    mini_raw_trace_file,
):
    raw = load_raw(mini_raw_trace_file)
    assert resolve_baseline_config(trace=raw) == {"max_width": 2, "max_refine": 1}
    overridden = resolve_baseline_config({"max_width": 32, "max_refine": 19}, trace=raw)
    assert overridden == {"max_width": 32, "max_refine": 19}


def test_allocate_from_raw_trace(mini_raw_trace_file, tmp_path):
    job = allocate_learning_workspace(
        mini_raw_trace_file,
        dest=tmp_path / "learning" / "test_job",
    )
    assert job.is_dir()
    assert (job / "manifest.json").is_file()
    assert (job / "trace/raw_scores.json").is_file()
    assert (job / METHOD_TEMPLATE_REL).is_file()
    assert not (job / METHOD_REL).exists()

    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["job_id"] == "test_job"
    assert manifest["seed_method"] is None
    assert manifest["baseline_preset"] == {"max_width": 2, "max_refine": 1}

    preset = refresh_learning_baseline_preset(
        job,
        overrides={"max_width": 32, "max_refine": 19},
    )
    assert preset == {"max_width": 32, "max_refine": 19}
    refreshed = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert refreshed["baseline_preset"] == preset


def test_allocate_from_trace_requires_dest(mini_raw_trace_file):
    with pytest.raises(ValueError, match="pass dest"):
        allocate_learning_workspace(mini_raw_trace_file)


def test_allocate_from_workspace_seeds_the_promoted_policy(mini_workspace):
    promoted = mini_workspace / PROMOTED_POLICY_REL
    promoted.parent.mkdir(parents=True)
    promoted.write_text("# promoted policy\n", encoding="utf-8")

    job = allocate_learning_workspace(mini_workspace)

    assert job.parent == mini_workspace.resolve() / "learning"
    assert (job / "trace/raw_scores.json").is_file()
    assert (job / METHOD_REL).read_text(encoding="utf-8") == "# promoted policy\n"
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed_method"] == str(promoted.resolve())


def test_evolve_grid_caps_forward_to_learning_manifest(tmp_path):
    cfg = PipelineConfig(
        workspace=tmp_path,
        explore_max_width=32,
        explore_max_refine=19,
    )
    assert _grid_caps(cfg) == {"max_width": 32, "max_refine": 19}
    assert _grid_caps(PipelineConfig(workspace=tmp_path)) is None


def test_build_workflow_config_wires_replay_eval_and_incumbent(
    mini_raw_trace_file,
    tmp_path,
    monkeypatch,
):
    job = allocate_learning_workspace(
        mini_raw_trace_file,
        dest=tmp_path / "learning" / "worker_job",
    )
    manifest = json.loads((job / "manifest.json").read_text(encoding="utf-8"))
    paths = paths_for_learning_job(job)
    monkeypatch.setenv("DRSI_REPLAY_MAX_WORKERS", "10")
    # Registered so monkeypatch restores it after build_workflow_config sets it.
    monkeypatch.setenv("DRSI_AGENT_STREAM", "0")

    wf = build_workflow_config(
        DiscoverConfig(learning_root=job, max_workers=32),
        trace_path=paths["trace"],
        paths=paths,
        manifest=manifest,
    )

    # The worker cap reaches the replay subprocess, not this process.
    assert wf.eval_env == {"DRSI_REPLAY_MAX_WORKERS": "32"}
    assert os.environ["DRSI_REPLAY_MAX_WORKERS"] == "10"
    assert wf.eval_cmd[1:4] == (
        str(EVAL_SCRIPT),
        str((job / METHOD_REL).resolve()),
        str(paths["trace"]),
    )
    assert wf.eval_cmd[-5:] == (
        "--result-dir",
        str(paths["results"].resolve()),
        "--target",
        "run_best",
        "--beta-sweep",
    )
    assert wf.eval_cwd == job.resolve()
    assert wf.evaluate_incumbent is True
    assert wf.incumbent_fallback.startswith(BASELINE_INCUMBENT_HEADER)
    assert "{'max_width': 2, 'max_refine': 1}" in wf.incumbent_fallback


def test_pareto_reward_penalizes_serial_replay(monkeypatch):
    """Equal attainment/work curves prefer the policy with fewer sequential rounds."""
    monkeypatch.setenv("DRSI_REPLAY_PARALLELISM_PENALTY", "0.2")

    def grid(rounds: list[int]) -> list[dict]:
        return [
            {
                "status": "ok",
                "beta": beta,
                "per_trace": [
                    {
                        "name": "toy",
                        "best": best,
                        "total_budget": budget,
                        "effective_sequential_rounds": sequential_rounds,
                        "run_best": 1.0,
                        "n_evaluated": 4,
                    }
                ],
            }
            for beta, best, budget, sequential_rounds in (
                (0.0, 0.5, 2, rounds[0]),
                (1.0, 1.0, 4, rounds[1]),
            )
        ]

    parallel = _pareto_summary(grid([1, 2]))
    serial = _pareto_summary(grid([2, 4]))

    assert parallel["auc"] == pytest.approx(serial["auc"])
    assert parallel["parallel_penalty"] == pytest.approx(0.5)
    assert serial["parallel_penalty"] == pytest.approx(1.0)
    assert parallel["reward"] == pytest.approx(
        parallel["auc"] - 0.2 * parallel["parallel_penalty"]
    )
    assert parallel["reward"] > serial["reward"]


def test_beta_sweep_records_replay_parallelism(
    mini_raw_trace_file, baseline_policy_file, monkeypatch, tmp_path
):
    monkeypatch.setenv("DRSI_REPLAY_MAX_WORKERS", "2")
    sweep = run_beta_sweep(
        baseline_policy_file,
        pool_dir=None,
        trace_src=mini_raw_trace_file,
        target_spec="run_best",
        betas=[0.5],
    )

    assert sweep["max_parallelism"] == 2
    assert sweep["pareto"]["reward"] == pytest.approx(
        sweep["pareto"]["auc"]
        - sweep["pareto"]["parallelism_weight"]
        * sweep["pareto"]["parallel_penalty"]
    )
    point = sweep["grid"][0]["per_trace"][0]
    batch_sizes = point["batch_sizes"]
    assert point["decision_rounds"] == len(batch_sizes)
    assert point["effective_sequential_rounds"] == sum(
        (batch_size + sweep["max_parallelism"] - 1) // sweep["max_parallelism"]
        for batch_size in batch_sizes
    )

    episodes = sweep["execution_traces"]
    assert len(episodes) == 1
    assert episodes[0]["beta"] == pytest.approx(0.5)
    assert episodes[0]["events"]
    assert episodes[0]["final"]["decision_rounds"] == point["decision_rounds"]

    trace_path = tmp_path / "policy_execution_traces.jsonl"
    _write_execution_traces(trace_path, episodes)
    saved = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    assert saved == episodes


def test_best_round_uses_reward_and_ignores_legacy_auc_only_sweeps(tmp_path):
    history = tmp_path / "history"
    method_file = METHOD_REL

    def archive(round_id: str, pareto: dict) -> None:
        root = history / round_id
        method = root / method_file
        method.parent.mkdir(parents=True)
        method.write_text("# policy\n", encoding="utf-8")
        results = root / "proposal_results"
        results.mkdir()
        (results / "beta_sweep.json").write_text(
            json.dumps(
                {
                    "pareto": {**pareto, "n_traces_scored": 1},
                    "grid": [{"status": "ok"}],
                    "n_execution_episodes": 1,
                }
            ),
            encoding="utf-8",
        )
        (results / "heuristic_eval.json").write_text(
            json.dumps({"status": "ok", "traces": [{"status": "ok"}]}),
            encoding="utf-8",
        )

    archive("r0000_20260101_000000_deadbeef", {"auc": 0.99})
    archive(
        "r0001_20260101_000001_feedface",
        {"auc": 0.30, "reward": 0.25, "parallel_penalty": 1.0},
    )
    archive(
        "r0002_20260101_000002_1234abcd",
        {"auc": 0.28, "reward": 0.26, "parallel_penalty": 0.4},
    )
    (history / "workflow_index.jsonl").write_text(
        "\n".join(
            json.dumps({"round_id": round_id, "eval_returncode": 0})
            for round_id in (
                "r0000_20260101_000000_deadbeef",
                "r0001_20260101_000001_feedface",
                "r0002_20260101_000002_1234abcd",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    best = _select_best_round(tmp_path, "history", method_file)
    assert best is not None
    assert best["round_id"] == "r0002_20260101_000002_1234abcd"
    assert best["reward"] == pytest.approx(0.26)


def test_best_round_excludes_failed_replay_even_with_higher_reward(tmp_path):
    history = tmp_path / "history"
    method_file = METHOD_REL

    def archive(round_id: str, reward: float, *, ok: bool) -> None:
        root = history / round_id
        method = root / method_file
        method.parent.mkdir(parents=True)
        method.write_text(f"# {round_id}\n", encoding="utf-8")
        results = root / "proposal_results"
        results.mkdir()
        (results / "heuristic_eval.json").write_text(
            json.dumps({"status": "ok" if ok else "error", "traces": [{"status": "ok"}]}),
            encoding="utf-8",
        )
        (results / "beta_sweep.json").write_text(
            json.dumps(
                {
                    "pareto": {
                        "reward": reward,
                        "auc": reward,
                        "parallel_penalty": 0.1,
                        "n_traces_scored": 1,
                    },
                    "grid": [{"status": "ok" if ok else "error"}],
                    "n_execution_episodes": 1 if ok else 0,
                }
            ),
            encoding="utf-8",
        )

    failed = "r0000_20260101_000000_deadbeef"
    valid = "r0001_20260101_000001_feedface"
    archive(failed, 0.99, ok=False)
    archive(valid, 0.25, ok=True)
    (history / "workflow_index.jsonl").write_text(
        "\n".join(
            (
                json.dumps({"round_id": failed, "eval_returncode": 1}),
                json.dumps({"round_id": valid, "eval_returncode": 0}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    best = _select_best_round(tmp_path, "history", method_file)
    assert best is not None
    assert best["round_id"] == valid
    assert best["reward"] == pytest.approx(0.25)


class _FakeDiscovery:
    """Fake proposer and replay evaluator for :func:`run_workflow`.

    A candidate's reward is looked up by the first line of ``method.py``; a
    candidate without an entry fails replay.
    """

    def __init__(self, workdir: Path, rewards: dict[str, float]):
        self.method = workdir / METHOD_REL
        self.results = workdir / "heuristic_results"
        self.rewards = rewards
        self.started_from: list[str] = []
        self.evaluated: list[str] = []

    def install(self, monkeypatch) -> "_FakeDiscovery":
        monkeypatch.setattr(workflow_module, "_run_proposer_for_round", self.propose)
        monkeypatch.setattr(workflow_module, "run_eval_subprocess", self.evaluate)
        return self

    def propose(self, **_kwargs):
        index = len(self.started_from)
        self.started_from.append(self.method.read_text(encoding="utf-8"))
        self.method.write_text(f"# proposal {index}\n", encoding="utf-8")
        return {"status": "ok", "message": "fake"}

    def evaluate(self, **_kwargs):
        text = self.method.read_text(encoding="utf-8")
        self.evaluated.append(text)
        reward = self.rewards.get(text.splitlines()[0])
        ok = reward is not None
        (self.results / "heuristic_eval.json").write_text(
            json.dumps({"status": "ok" if ok else "error", "traces": [{"status": "ok"}]}),
            encoding="utf-8",
        )
        (self.results / "beta_sweep.json").write_text(
            json.dumps(
                {
                    "pareto": {
                        "reward": reward if ok else -0.05,
                        "auc": reward + 0.05 if ok else 0.0,
                        "parallel_penalty": 1.0,
                        "n_traces_scored": 1 if ok else 0,
                    },
                    "grid": [{"status": "ok" if ok else "error"}],
                    "n_execution_episodes": 1 if ok else 0,
                }
            ),
            encoding="utf-8",
        )
        return (0 if ok else 1), "", ""


def _discovery_job(workdir: Path, *, method_text: str = TEMPLATE_TEXT) -> Path:
    method = workdir / METHOD_REL
    method.parent.mkdir(parents=True)
    (workdir / METHOD_TEMPLATE_REL).write_text(TEMPLATE_TEXT, encoding="utf-8")
    method.write_text(method_text, encoding="utf-8")
    (workdir / "prompt.txt").write_text("write a policy", encoding="utf-8")
    (workdir / "heuristic_results").mkdir()
    return workdir


def _workflow_config(workdir: Path, **overrides) -> WorkflowConfig:
    fields = {
        "workdir": workdir,
        "method_file": METHOD_REL,
        "history_dir": "history",
        "proposer_prompt_path": workdir / "prompt.txt",
        "rounds": 2,
        "log_parent": workdir / "logs",
        "result_dir": workdir / "heuristic_results",
        "eval_cmd": ("fake-eval",),
    }
    fields.update(overrides)
    return WorkflowConfig(**fields)


def _best_round(workdir: Path) -> dict:
    return json.loads((workdir / "history/best_round.json").read_text(encoding="utf-8"))


def test_workflow_retries_until_a_qualified_candidate_then_promotes(tmp_path, monkeypatch):
    """A failed initial proposal must not block a later valid retry."""
    workdir = _discovery_job(tmp_path)
    fake = _FakeDiscovery(workdir, {"# proposal 1": 0.25}).install(monkeypatch)
    cfg = _workflow_config(workdir, rounds=1, retry_rounds=2, max_retry_batches=1)

    rows = asyncio.run(run_workflow(cfg))

    assert [row["eval_returncode"] for row in rows] == [1, 0]
    assert [row["kind"] for row in rows] == ["proposal", "proposal"]
    assert len(fake.started_from) == 2
    best = _best_round(workdir)
    assert best["round_index"] == 1
    assert best["incumbent"] is False
    method = workdir / METHOD_REL
    assert method.read_text(encoding="utf-8") == "# proposal 1\n"
    assert _discover_ok({"best_round": best, "method_file": str(method)})


def test_incumbent_baseline_is_replayed_first_and_can_be_kept(tmp_path, monkeypatch):
    workdir = _discovery_job(tmp_path)
    fake = _FakeDiscovery(
        workdir,
        {BASELINE_INCUMBENT_HEADER: 0.5, "# proposal 0": 0.2, "# proposal 1": 0.3},
    ).install(monkeypatch)
    incumbent = baseline_incumbent_source({"max_width": 2, "max_refine": 1})
    cfg = _workflow_config(workdir, evaluate_incumbent=True, incumbent_fallback=incumbent)

    rows = asyncio.run(run_workflow(cfg))

    # M = rounds + 1 candidates; the agent is not called for the incumbent.
    assert [row["kind"] for row in rows] == ["incumbent", "proposal", "proposal"]
    assert fake.evaluated[0] == incumbent
    # Before any promotion, every proposal starts from the template.
    assert fake.started_from == [TEMPLATE_TEXT, TEMPLATE_TEXT]
    best = _best_round(workdir)
    assert best["round_index"] == 0
    assert best["incumbent"] is True
    method = workdir / METHOD_REL
    assert method.read_text(encoding="utf-8") == incumbent
    assert _discover_ok({"best_round": best, "method_file": str(method)})


def test_seeded_incumbent_is_the_first_proposal_starting_point(tmp_path, monkeypatch):
    workdir = _discovery_job(tmp_path, method_text="# promoted policy\n")
    fake = _FakeDiscovery(
        workdir,
        {"# promoted policy": 0.1, "# proposal 0": 0.2, "# proposal 1": 0.4},
    ).install(monkeypatch)
    cfg = _workflow_config(
        workdir,
        evaluate_incumbent=True,
        incumbent_fallback=baseline_incumbent_source(),
    )

    rows = asyncio.run(run_workflow(cfg))

    assert [row["kind"] for row in rows] == ["incumbent", "proposal", "proposal"]
    # A seeded policy wins over the fallback as the incumbent.
    assert fake.evaluated[0] == "# promoted policy\n"
    assert fake.started_from == ["# promoted policy\n", TEMPLATE_TEXT]
    best = _best_round(workdir)
    assert best["round_index"] == 2
    assert best["incumbent"] is False
    assert (workdir / METHOD_REL).read_text(encoding="utf-8") == "# proposal 1\n"


def test_incumbent_needs_a_seed_or_a_fallback(tmp_path):
    workdir = _discovery_job(tmp_path)
    cfg = _workflow_config(workdir, evaluate_incumbent=True)
    with pytest.raises(ValueError, match="incumbent_fallback"):
        asyncio.run(run_workflow(cfg))


class _Interrupted(Exception):
    pass


@pytest.mark.parametrize("seed", [None, "# promoted policy\n"])
def test_resume_after_the_incumbent_round(tmp_path, monkeypatch, seed):
    workdir = _discovery_job(tmp_path, method_text=seed or TEMPLATE_TEXT)
    rewards = {
        BASELINE_INCUMBENT_HEADER: 0.1,
        "# promoted policy": 0.1,
        "# proposal 0": 0.2,
        "# proposal 1": 0.3,
    }
    fake = _FakeDiscovery(workdir, rewards).install(monkeypatch)
    cfg = _workflow_config(
        workdir,
        resume=True,
        evaluate_incumbent=True,
        incumbent_fallback=baseline_incumbent_source(),
    )

    def interrupted(**_kwargs):
        raise _Interrupted

    monkeypatch.setattr(workflow_module, "_run_proposer_for_round", interrupted)
    with pytest.raises(_Interrupted):
        asyncio.run(run_workflow(cfg))
    (workdir / METHOD_REL).write_text("# half-written proposal\n", encoding="utf-8")

    fake.install(monkeypatch)
    rows = asyncio.run(run_workflow(cfg))

    assert [row["round_index"] for row in rows] == [1, 2]
    assert len({row["round_id"].split("_", 1)[1] for row in rows}) == 1
    # The resumed first proposal starts where a fresh run would have.
    assert fake.started_from == [seed or TEMPLATE_TEXT, TEMPLATE_TEXT]
    assert _best_round(workdir)["round_index"] == 2


def test_seeded_method_text_ignores_template_and_generated_baseline(tmp_path):
    template = tmp_path / "method.template.py"
    template.write_text(TEMPLATE_TEXT, encoding="utf-8")
    method = tmp_path / "method.py"
    assert _seeded_method_text(method, template) is None
    method.write_text(TEMPLATE_TEXT, encoding="utf-8")
    assert _seeded_method_text(method, template) is None
    method.write_text(baseline_incumbent_source({"max_width": 2}), encoding="utf-8")
    assert _seeded_method_text(method, template) is None
    method.write_text("# promoted policy\n", encoding="utf-8")
    assert _seeded_method_text(method, template) == "# promoted policy\n"


@pytest.mark.parametrize("incumbent", [True, False])
def test_cycle_promotes_only_a_policy_that_beat_the_incumbent(
    tmp_path, monkeypatch, incumbent
):
    ws = tmp_path / "ws"
    ws.mkdir()
    job = tmp_path / "learning_job"
    method = job / METHOD_REL
    method.parent.mkdir(parents=True)
    method.write_text("# selected\n", encoding="utf-8")
    archived = job / "heuristic_history/r0000_20260101_000000_deadbeef" / METHOD_REL
    archived.parent.mkdir(parents=True)
    archived.write_text("# selected\n", encoding="utf-8")
    discover = {
        "best_round": {
            "method_src": str(archived),
            "eval_returncode": 0,
            "incumbent": incumbent,
        },
        "method_file": str(method),
    }
    promoted: list[Path] = []
    monkeypatch.setattr(
        orchestrate_module,
        "_run_learning_stage",
        lambda cfg, ws, *, cycle: (job, discover, None),
    )
    monkeypatch.setattr(
        orchestrate_module,
        "_run_promote_stage",
        lambda learning_root, ws: promoted.append(learning_root) or str(ws / PROMOTED_POLICY_REL),
    )

    row = orchestrate_module.run_cycle(
        PipelineConfig(workspace=ws), cycle=0, start_stage="learning"
    )

    assert promoted == ([] if incumbent else [job])
    assert (row["promoted_method"] is None) is incumbent


def test_cycle_from_learning_archives_a_round_explored_outside_the_run(
    tmp_path, monkeypatch, mini_raw_trace_file
):
    # A shared E0 copied into the workspace: learn from it, then archive it as
    # iter000 so the next cycle explores a fresh grid instead of resuming it.
    ws = tmp_path / "ws"
    (ws / "explore_b0" / "attempt_00").mkdir(parents=True)
    (ws / "raw_scores.json").write_bytes(mini_raw_trace_file.read_bytes())
    (ws / "live_cycle_manifest.json").write_text(json.dumps({"cycle": 0}), encoding="utf-8")
    monkeypatch.setattr(
        orchestrate_module,
        "_run_learning_stage",
        lambda cfg, ws, *, cycle: (tmp_path / "learning_job", {}, None),
    )

    row = orchestrate_module.run_cycle(
        PipelineConfig(workspace=ws), cycle=0, start_stage="learning"
    )

    archive = ws.resolve() / "history" / "iter000"
    assert row["archived_to"] == str(archive)
    assert (archive / "explore_b0" / "attempt_00").is_dir()
    assert (archive / "live_cycle_manifest.json").is_file()
    assert not (ws / "raw_scores.json").exists()
    assert (ws / "trace_pool" / "iter000" / "raw_scores.json").is_file()


def test_cycle_from_promote_reuses_the_finished_learning_job(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    job = orchestrate_module._learning_job_path(ws.resolve(), cycle=2)
    method = job / METHOD_REL
    method.parent.mkdir(parents=True)
    method.write_text("# selected\n", encoding="utf-8")
    archived = job / "heuristic_history/r0001_20260101_000000_deadbeef" / METHOD_REL
    archived.parent.mkdir(parents=True)
    archived.write_text("# selected\n", encoding="utf-8")
    (job / "discover_summary.json").write_text(
        json.dumps(
            {
                "best_round": {
                    "method_src": str(archived),
                    "eval_returncode": 0,
                    "incumbent": False,
                },
                "method_file": str(method),
            }
        ),
        encoding="utf-8",
    )

    def no_learning(*_args, **_kwargs):
        raise AssertionError("--from promote must not re-run learning")

    promoted: list[Path] = []
    monkeypatch.setattr(orchestrate_module, "_run_learning_stage", no_learning)
    monkeypatch.setattr(
        orchestrate_module,
        "_run_promote_stage",
        lambda learning_root, ws: promoted.append(learning_root) or str(ws / PROMOTED_POLICY_REL),
    )

    row = orchestrate_module.run_cycle(
        PipelineConfig(workspace=ws), cycle=2, start_stage="promote"
    )

    assert promoted == [job.resolve()]
    assert row["learning_root"] == str(job.resolve())


def test_cycle_from_promote_requires_a_finished_learning_job(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    with pytest.raises(FileNotFoundError, match="run the learning stage of cycle 0"):
        orchestrate_module.run_cycle(
            PipelineConfig(workspace=ws), cycle=0, start_stage="promote"
        )


def test_archive_round_copies_execution_traces(tmp_path):
    method_file = METHOD_REL
    method = tmp_path / method_file
    method.parent.mkdir(parents=True)
    method.write_text("# policy\n", encoding="utf-8")
    results = tmp_path / "heuristic_results"
    results.mkdir()
    (results / "policy_execution_traces.jsonl").write_text(
        '{"schema_version":1}\n', encoding="utf-8"
    )

    archived = archive_round(
        workdir=tmp_path,
        history_dir="heuristic_history",
        round_id="r0000_20260101_000000_deadbeef",
        method_file=method_file,
        method_src=method,
        result_dir=results,
    )

    copied = archived / "proposal_results/policy_execution_traces.jsonl"
    assert copied.read_text(encoding="utf-8") == '{"schema_version":1}\n'
