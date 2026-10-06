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

"""Agent-driven policy discovery on one learning job (the offline phase).

The incumbent policy and ``rounds`` agent rewrites are each replayed on the
job's frozen traces and archived; the best replay-qualified candidate is
selected, so one phase compares ``M = rounds + 1`` candidates.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from shutil import which

from dream_rsi.env.trace.raw import raw_trace_to_trace
from dream_rsi.learning.allocate import PROMPT_REL, load_manifest
from dream_rsi.learning.heuristic_search.workflow import (
    WorkflowConfig,
    baseline_incumbent_source,
    run_workflow,
    scan_history_resume,
)
from dream_rsi.learning.pipeline import (
    ROOT,
    load_raw,
    resolve_baseline_config,
    run_parallel_refine,
)
from dream_rsi.learning.trace_pool import TRACE_POOL_NAME, list_pool_traces
from dream_rsi.policy.constants import METHOD_REL

EVAL_SCRIPT = ROOT / "dream_rsi/learning/heuristic_search/eval.py"


@dataclass
class DiscoverConfig:
    """Configuration for multi-round agent discovery on one learning job."""

    learning_root: Path
    agent: str = "claude"
    rounds: int = 3
    retry_rounds: int = 3
    max_retry_batches: int = 3
    evaluate_incumbent: bool = True
    target: str = "run_best"
    resume: bool = True
    eval_timeout_sec: float = 1800.0
    stream: bool = True
    max_workers: int | None = None

    def __post_init__(self) -> None:
        if self.max_workers is not None and self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")


def _gemini_ready() -> bool:
    keys = os.environ.get("GEMINI_API_KEYS", "").strip()
    if keys:
        return True
    return bool(
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    )


def _replay_worker_default() -> str:
    return os.environ.get("DRSI_REPLAY_MAX_WORKERS", "10")


def preflight(agent: str) -> str | None:
    agent = agent.lower()
    if agent == "gemini":
        if not _gemini_ready():
            return "GEMINI_API_KEY / GOOGLE_API_KEY not set"
        gemini_bin = os.environ.get("GEMINI_BIN", "gemini").strip() or "gemini"
        if which(gemini_bin) is None:
            return f"gemini CLI not found: {gemini_bin}"
    return None


def paths_for_learning_job(job_root: Path) -> dict[str, Path]:
    manifest = load_manifest(job_root)
    return {
        "trace": job_root / manifest.get("trace", "trace/raw_scores.json"),
        "results": job_root / manifest.get("results_dir", "heuristic_results"),
        "history": job_root / manifest.get("history_dir", "heuristic_history"),
        "trace_pool": job_root / manifest.get("trace_pool", TRACE_POOL_NAME),
    }


def baseline_benchmark(trace_path: Path, *, preset: dict | None = None) -> dict:
    raw = load_raw(trace_path)
    cfg = resolve_baseline_config(preset, trace=raw)
    rep = run_parallel_refine(raw, config=cfg)
    trace = raw_trace_to_trace(raw)
    return {
        "name": "parallel_refine",
        "config": cfg,
        "budget": rep.budget,
        "best": rep.best,
        "budget_to_run_best": rep.reach_run_best,
        "hit_run_best": rep.hit_run_best,
        "run_best": trace.best_score,
        "ok": rep.ok,
        "error": rep.error,
    }


def write_baseline_reference(history_dir: Path, baseline: dict) -> None:
    """Seed ``{history_dir}/baseline/`` with the baseline method + its score.

    The proposer reads this as the floor to beat. We deliberately drop
    ``run_best`` / ``hit_run_best`` / ``budget_to_run_best`` — those reveal the
    grid ceiling and must never leak to the agent.
    """
    base_dir = history_dir / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)
    src = ROOT / "dream_rsi/policy/baseline.py"
    if src.is_file():
        (base_dir / "baseline_method.py").write_text(
            src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8"
        )
    summary = {
        "name": baseline.get("name", "parallel_refine"),
        "config": baseline.get("config"),
        "best": baseline.get("best"),
        "budget": baseline.get("budget"),
        "note": (
            "Reference baseline (fixed parallel-refine). Beat it: reach this 'best' "
            "for less 'budget', or push 'best' higher. run-best/target omitted on purpose."
        ),
    }
    (base_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def build_workflow_config(
    cfg: DiscoverConfig,
    *,
    trace_path: Path,
    paths: dict[str, Path],
    manifest: dict,
) -> WorkflowConfig:
    workdir = cfg.learning_root.resolve()
    method_file = manifest.get("method_file", METHOD_REL)
    method_path = (workdir / method_file).resolve()
    prompt_path = (workdir / manifest.get("prompt", PROMPT_REL)).resolve()
    results_dir = paths["results"].resolve()

    pool_dir = paths.get("trace_pool")
    use_pool = pool_dir is not None and pool_dir.is_dir() and bool(list_pool_traces(pool_dir))
    traces = ["--trace-pool", str(pool_dir.resolve())] if use_pool else [str(trace_path)]
    # Writes beta_sweep.json each round: its pareto.reward ranks the candidates.
    eval_cmd = (
        sys.executable,
        str(EVAL_SCRIPT),
        str(method_path),
        *traces,
        "--result-dir",
        str(results_dir),
        "--target",
        cfg.target,
        "--beta-sweep",
    )
    eval_env = {}
    if cfg.max_workers is not None:
        eval_env["DRSI_REPLAY_MAX_WORKERS"] = str(cfg.max_workers)

    if cfg.stream:
        os.environ["DRSI_AGENT_STREAM"] = "1"
    else:
        os.environ.pop("DRSI_AGENT_STREAM", None)

    # Only keys the proposer prompt actually substitutes. Ranking fitness is
    # pareto.reward in beta_sweep.json, not a budget/run-best metric.
    prompt_vars = {
        "results_dir": str(results_dir.relative_to(workdir)),
        "trace_pool": str(pool_dir.resolve().relative_to(workdir)) if use_pool else "",
    }

    return WorkflowConfig(
        workdir=workdir,
        method_file=method_file,
        history_dir=str(paths["history"].resolve().relative_to(workdir)),
        proposer_prompt_path=prompt_path,
        rounds=cfg.rounds,
        log_parent=workdir / manifest.get("logs_dir", "logs"),
        proposer_backend=cfg.agent.lower(),
        resume=cfg.resume,
        result_dir=results_dir,
        eval_cmd=eval_cmd,
        eval_cwd=workdir,
        eval_env=eval_env,
        eval_timeout_sec=cfg.eval_timeout_sec,
        stream=cfg.stream,
        prompt_vars=prompt_vars,
        retry_rounds=cfg.retry_rounds,
        max_retry_batches=cfg.max_retry_batches,
        evaluate_incumbent=cfg.evaluate_incumbent,
        incumbent_fallback=baseline_incumbent_source(manifest.get("baseline_preset")),
    )


async def run_discover(cfg: DiscoverConfig) -> dict:
    err = preflight(cfg.agent)
    if err:
        raise RuntimeError(err)

    workdir = cfg.learning_root.resolve()
    manifest = load_manifest(workdir)
    paths = paths_for_learning_job(workdir)
    trace_path = paths["trace"].resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace not found in learning job: {trace_path}")

    baseline = baseline_benchmark(trace_path, preset=manifest.get("baseline_preset"))
    (workdir / "baseline_benchmark.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    write_baseline_reference(paths["history"], baseline)

    n_pool = len(list_pool_traces(paths["trace_pool"]))
    print("[discover] job       :", workdir)
    print("[discover] trace     :", trace_path)
    if n_pool:
        print("[discover] trace pool:", paths["trace_pool"], f"({n_pool} traces)")
    print("[discover] agent     :", cfg.agent)
    print(
        "[discover] candidates:",
        f"{'incumbent + ' if cfg.evaluate_incumbent else ''}{cfg.rounds} proposal(s)",
    )
    print(
        "[discover] retry     :",
        f"up to {cfg.retry_rounds} extra proposal(s) per batch, "
        f"{cfg.max_retry_batches} batch(es)",
    )
    print("[discover] target    :", cfg.target)
    print("[discover] stream    :", cfg.stream)
    replay_workers = (
        cfg.max_workers if cfg.max_workers is not None else _replay_worker_default()
    )
    print("[discover] workers   :", replay_workers)
    print("[discover] history   :", paths["history"])
    print("[discover] results   :", paths["results"])
    print(
        "[discover] baseline  : parallel_refine "
        f"best={baseline['best']:.6f} "
        f"total={baseline['budget']} "
        f"hit_run_best={baseline['hit_run_best']}"
    )
    print()

    wf = build_workflow_config(
        cfg,
        trace_path=trace_path,
        paths=paths,
        manifest=manifest,
    )
    rounds = await run_workflow(wf)

    # run_workflow returns [] when all planned rounds were already completed in a
    # prior (resumed) run. Flag that so callers can still treat the finished job
    # as a success instead of mistaking the empty round list for a failure.
    already_complete = False
    if not rounds:
        _ts, _uid, nxt = scan_history_resume(workdir, wf.history_dir)
        already_complete = bool(_ts) and nxt >= wf.initial_rounds

    eval_json = paths["results"] / "heuristic_eval.json"
    last_eval = None
    if eval_json.is_file():
        last_eval = json.loads(eval_json.read_text(encoding="utf-8"))

    best_round = None
    best_round_json = paths["history"] / "best_round.json"
    if best_round_json.is_file():
        try:
            best_round = json.loads(best_round_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            best_round = None

    summary = {
        "task": workdir.name,
        "trace": str(trace_path),
        "agent": cfg.agent,
        "rounds_planned": cfg.rounds,
        "evaluate_incumbent": cfg.evaluate_incumbent,
        "retry_rounds": cfg.retry_rounds,
        "max_retry_batches": cfg.max_retry_batches,
        "rounds_run": len(rounds),
        "already_complete": already_complete,
        "target": cfg.target,
        "max_workers": cfg.max_workers,
        "baseline_benchmark": baseline,
        "rounds": rounds,
        "last_eval": last_eval,
        "best_round": best_round,
        "learning_root": str(workdir),
        "method_file": str((workdir / wf.method_file).resolve()),
        "history_dir": str(paths["history"]),
    }
    summary_path = workdir / "discover_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_discover_sync(cfg: DiscoverConfig) -> dict:
    return asyncio.run(run_discover(cfg))


def format_discover_summary(summary: dict) -> str:
    lines = [
        f"job           : {summary['learning_root']}",
        f"trace         : {summary['trace']}",
        f"agent         : {summary['agent']}",
        f"rounds        : {summary['rounds_run']} run, "
        f"{summary['rounds_planned']} proposal(s) planned"
        f"{' + incumbent' if summary.get('evaluate_incumbent') else ''}",
        f"method        : {summary['method_file']}",
        f"history       : {summary['history_dir']}",
    ]
    br = summary.get("best_round")
    if br:
        kept = "  (incumbent kept)" if br.get("incumbent") else ""
        lines.append(
            f"selected      : {br.get('round_id')}  reward={br.get('reward', 0):.4g}  "
            f"auc={br.get('pareto_auc', 0):.4g}{kept}"
        )
    b = summary.get("baseline_benchmark") or {}
    lines.append(
        f"baseline      : parallel_refine  best={b.get('best', 0):.6f}  "
        f"total={b.get('budget')}  hit_run_best={b.get('hit_run_best')}"
    )
    ev = summary.get("last_eval") or {}
    h = ev.get("heuristic") or {}
    if ev.get("trace_pool"):
        agg = ev.get("aggregate") or {}
        total = agg.get("total_budget_median")
        lines.append(
            f"last eval     : best={h.get('best', 0):.6f}  "
            f"median total={total if total is not None else '-'}  "
            f"reached {agg.get('n_reached', 0)}/{agg.get('n_traces', 0)} traces"
        )
    elif h:
        total = h.get("total_budget")
        lines.append(
            f"last eval     : best={h.get('best', 0):.6f}  "
            f"total={total if total is not None else '-'}  "
            f"reached={h.get('reached_target')}"
        )
    elif ev.get("status") == "error":
        lines.append(f"last eval     : ERROR — {ev.get('error', '')}")
    return "\n".join(lines) + "\n"
