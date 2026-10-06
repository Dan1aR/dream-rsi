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

"""Fitness evaluator for a discovered exploration heuristic.

Loads ``OptimalPolicy`` from a method file and replays it on offline trace(s) via
``TraceEnv`` (replay mode = the same surface online explore uses).

Two artifacts are written per run:
  * ``heuristic_eval.json`` — a per-run summary at the policy's default beta
    (best reached, probes actually spent, whether it reached run-best).
  * ``beta_sweep.json`` — the policy run at each beta in ``DRSI_REPLAY_BETA_GRID``.
    The ranking fitness is ``pareto.reward``: normalized budget<->attainment AUC minus
    a worker-capped sequential-round penalty. Higher reaches each trace's ceiling with
    fewer probes while preserving live parallelism.
  * ``policy_execution_traces.jsonl`` — one compact, decision-by-decision replay
    episode for every ``(trace, beta)`` used by the beta sweep.

Budget is always the probes actually spent (replay always finishes); whether a
trace reached run-best is reported separately (``reached_target`` / ``n_reached``)
and is NOT folded into the budget.

CLI arguments take precedence over the matching env vars:
    DRSI_REPLAY_METHOD_FILE   policy file defining ``OptimalPolicy``
    DRSI_REPLAY_TRACE         single frozen trace (file or workspace)
    DRSI_REPLAY_TRACE_POOL    directory of frozen traces (``iter*/raw_scores.json``)
    DRSI_REPLAY_TARGET        float, or "run_best" (default), or "baseline+0.02"
    DRSI_REPLAY_RESULT_DIR    where heuristic_eval.json / beta_sweep.json are written
    DRSI_REPLAY_BETA_SWEEP    truthy -> also write beta_sweep.json
    DRSI_REPLAY_BETA_GRID     comma list of betas (default 0.0,0.25,0.5,0.75,1.0)
    DRSI_REPLAY_MAX_WORKERS   replay parallelism assumption W (default 10)
    DRSI_REPLAY_PARALLELISM_PENALTY  penalty weight lambda (default 0.05)

Usage:
    python dream_rsi/learning/heuristic_search/eval.py method.py trace/raw_scores.json \
        --result-dir results/ --beta-sweep
    python dream_rsi/learning/heuristic_search/eval.py method.py --trace-pool trace_pool/ \
        --result-dir results/ --beta-sweep
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dream_rsi.env import TraceEnv  # noqa: E402
from dream_rsi.env.trace.raw import (  # noqa: E402
    export_raw_scores,
    trace_from_raw_file,
    trace_from_workspace,
    write_raw_scores,
)
from dream_rsi.learning.trace_pool import list_pool_traces  # noqa: E402
from dream_rsi.policy.grid_planning import make_replay_context, resolve_grid_plan  # noqa: E402
from dream_rsi.policy.load import load_policy_class  # noqa: E402

DEFAULT_REPLAY_MAX_WORKERS = 10
DEFAULT_PARALLELISM_PENALTY = 0.05


def _load_trace(src: Path):
    return trace_from_raw_file(src) if src.is_file() else trace_from_workspace(src)


def _resolve_trace_path(src: Path) -> Path:
    """Return a frozen ``raw_scores.json`` path for :meth:`TraceEnv.replay`."""
    if src.is_file():
        return src.resolve()
    ws = src.resolve()
    raw = ws / "raw_scores.json"
    if raw.is_file():
        return raw.resolve()
    write_raw_scores(export_raw_scores(ws), raw)
    return raw.resolve()


def _resolve_target(spec: str, trace) -> float:
    spec = (spec or "run_best").strip()
    if spec in ("run_best", "", "best"):
        return trace.best_score
    if spec.startswith("baseline"):
        rest = spec[len("baseline"):].strip()
        if rest.startswith("+"):
            return trace.baseline_score + float(rest[1:])
        if rest.startswith("*"):
            return trace.baseline_score * float(rest[1:])
        return trace.baseline_score
    return float(spec)


def _replay_parallelism() -> int:
    raw = os.environ.get("DRSI_REPLAY_MAX_WORKERS", str(DEFAULT_REPLAY_MAX_WORKERS))
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_REPLAY_MAX_WORKERS


def _parallelism_penalty_weight() -> float:
    raw = os.environ.get(
        "DRSI_REPLAY_PARALLELISM_PENALTY", str(DEFAULT_PARALLELISM_PENALTY)
    )
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_PARALLELISM_PENALTY


def _summarize(result, target: float) -> dict:
    return {
        "best": result.best,
        "total_budget": result.budget,
        "reached_target": result.best >= target - 1e-9,
        "decision_rounds": result.decision_rounds,
        "effective_sequential_rounds": result.effective_sequential_rounds,
        "batch_sizes": list(result.batch_sizes),
    }


def evaluate_on_trace(
    method_file: Path,
    trace_path: Path,
    *,
    target_spec: str,
    config: dict | None = None,
    parallelism: int | None = None,
    include_execution_trace: bool = False,
    planning_history: tuple[dict, ...] = (),
) -> dict:
    trace_path = _resolve_trace_path(trace_path)
    trace = _load_trace(trace_path)
    target = _resolve_target(target_spec, trace)
    Policy = load_policy_class(method_file)
    workers = parallelism if parallelism is not None else _replay_parallelism()
    policy = Policy(config)
    planning_context = make_replay_context(
        trace_path,
        max_workers=workers,
        history=planning_history,
    )
    grid_plan = resolve_grid_plan(
        policy,
        planning_context,
        support_branch_count=planning_context.trace_branch_count,
        support_refine_count=planning_context.trace_refine_count,
    )
    # n_evaluated = nodes that were actually run (done/failed). In replay only these
    # are probeable, so it is the exhaustive budget (what parallel-refine spends) and
    # the natural budget-normalizer. n_nodes also counts unrun `pending` scaffold.
    n_evaluated = sum(1 for n in trace.nodes if getattr(n, "evaluated", False))
    row: dict = {
        "trace": str(trace_path),
        "name": trace_path.parent.name,
        "n_nodes": len(trace.nodes),
        "n_evaluated": n_evaluated,
        "max_parallelism": workers,
        "baseline_score": trace.baseline_score,
        "run_best": trace.best_score,
        "target": target,
        "grid_plan": grid_plan.to_dict(),
    }
    if grid_plan.out_of_support:
        row.update(
            {
                "heuristic": None,
                "status": "out_of_support",
                "note": "planner requested a grid outside this frozen trace's support",
            }
        )
        return row

    env = TraceEnv.replay(
        trace_path,
        max_workers=workers,
        grid_plan=grid_plan.effective,
    )
    result = policy.solve(env)
    row.update({"heuristic": _summarize(result, target), "status": "ok"})
    if not row["heuristic"]["reached_target"]:
        row["note"] = "heuristic did NOT reach target on this trace"
    if include_execution_trace:
        row["execution_trace"] = env.execution_trace()
    return row


def _prior_manifest_for_trace(trace_path: Path) -> dict | None:
    """Read only the completed-cycle sidecar paired with a prior pool trace."""
    path = trace_path.resolve().parent / "live_cycle_manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _planning_trace_order(trace_path: Path) -> tuple[int, int, str]:
    """Chronological pool order; ``_current`` must be evaluated after archived iters."""
    name = trace_path.resolve().parent.name
    match = re.fullmatch(r"iter(\d+)", name)
    if match:
        return (0, int(match.group(1)), name)
    if name == "_current":
        return (2, 0, name)
    return (1, 0, name)


def evaluate_multi_trace(
    method_file: Path,
    trace_paths: list[Path],
    *,
    target_spec: str,
    config: dict | None = None,
    parallelism: int | None = None,
    include_execution_trace: bool = False,
) -> dict:
    per_trace: list[dict] = []
    rc = 0
    planning_history: list[dict] = []
    for tp in sorted(trace_paths, key=_planning_trace_order):
        try:
            row = evaluate_on_trace(
                method_file,
                tp,
                target_spec=target_spec,
                config=config,
                parallelism=parallelism,
                include_execution_trace=include_execution_trace,
                planning_history=tuple(planning_history),
            )
            per_trace.append(row)
            if row.get("status") != "ok":
                rc = 1
        except Exception as e:
            rc = 1
            per_trace.append(
                {
                    "trace": str(tp),
                    "name": tp.parent.name,
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                    "heuristic": None,
                }
            )
        manifest = _prior_manifest_for_trace(tp)
        if manifest is not None:
            planning_history.append(manifest)

    # Budget axis = probes actually spent (replay always finishes). Whether a trace
    # reached run-best belongs on the best axis / n_reached, NOT as a budget penalty —
    # inflating budget for misses would double-count against the pareto curve.
    budgets: list[float] = [
        float(row["heuristic"].get("total_budget", 0))
        for row in per_trace
        if row.get("status") == "ok" and row.get("heuristic")
    ]
    effective_rounds: list[float] = [
        float(row["heuristic"].get("effective_sequential_rounds", 0))
        for row in per_trace
        if row.get("status") == "ok" and row.get("heuristic")
    ]
    decision_rounds: list[float] = [
        float(row["heuristic"].get("decision_rounds", 0))
        for row in per_trace
        if row.get("status") == "ok" and row.get("heuristic")
    ]
    agg = {
        "n_traces": len(trace_paths),
        "total_budget_median": int(statistics.median(budgets)) if budgets else None,
        "total_budget_mean": statistics.mean(budgets) if budgets else None,
        "total_budget_worst": int(max(budgets)) if budgets else None,
        "n_reached": sum(
            1
            for row in per_trace
            if row.get("heuristic") and row["heuristic"].get("reached_target")
        ),
        "effective_sequential_rounds_median": (
            int(statistics.median(effective_rounds)) if effective_rounds else None
        ),
        "decision_rounds_median": (
            int(statistics.median(decision_rounds)) if decision_rounds else None
        ),
    }

    primary_total = agg["total_budget_median"]
    first = per_trace[0] if per_trace else {}
    out: dict = {
        "method_file": str(method_file.resolve()),
        "trace_pool": True,
        "traces": per_trace,
        "aggregate": agg,
        "baseline_score": first.get("baseline_score"),
        "run_best": first.get("run_best"),
        "target": first.get("target"),
        "n_nodes": sum(r.get("n_nodes") or 0 for r in per_trace if r.get("status") == "ok"),
        "status": "ok" if rc == 0 else "error",
        "heuristic": {
            "best": max(
                (r["heuristic"]["best"] for r in per_trace if r.get("heuristic")),
                default=0.0,
            ),
            "total_budget": primary_total,
            "reached_target": agg["n_reached"] == len(trace_paths) if trace_paths else False,
            "decision_rounds": agg["decision_rounds_median"],
            "effective_sequential_rounds": agg["effective_sequential_rounds_median"],
        },
    }
    if rc != 0:
        out["note"] = "one or more traces failed during eval"
    elif agg["n_reached"] < len(trace_paths):
        out["note"] = "heuristic did NOT reach target on all traces"
    return out


def _render(out: dict) -> str:
    lines = []
    if out.get("trace_pool"):
        agg = out.get("aggregate") or {}
        lines.append(
            f"trace pool: {agg.get('n_traces', 0)} traces | "
            f"median total={agg.get('total_budget_median')} | "
            f"reached {agg.get('n_reached', 0)}/{agg.get('n_traces', 0)}"
        )
        lines.append(
            "default-beta run; ranking fitness = pareto.reward (see beta_sweep.json)\n"
        )
        for row in out.get("traces") or []:
            name = row.get("name") or row.get("trace")
            if row.get("status") == "ok" and row.get("heuristic"):
                h = row["heuristic"]
                lines.append(
                    f"  {name}: best={h.get('best', 0):.4f}  "
                    f"total={h.get('total_budget')}  "
                    f"sequential={h.get('effective_sequential_rounds')}  "
                    f"reached={h.get('reached_target')}"
                )
            else:
                lines.append(f"  {name}: ERROR {row.get('error', '')}")
        lines.append("")
        if out.get("status") == "ok" and out.get("heuristic"):
            h = out["heuristic"]
            total = h.get("total_budget")
            total_s = str(total) if total is not None else "-"
            lines.append(
                f"OptimalPolicy  best={h.get('best', 0):.4f}  "
                f"total={total_s}  "
                f"sequential={h.get('effective_sequential_rounds')}  "
                f"reached={h.get('reached_target')}"
            )
        else:
            lines.append(f"OptimalPolicy  ERROR: {out.get('error', out.get('note', ''))}")
        return "\n".join(lines) + "\n"

    if out.get("status") != "ok":
        lines.append(f"OptimalPolicy  ERROR: {out.get('error', out.get('note', ''))}")
        return "\n".join(lines) + "\n"

    lines.append(
        f"trace: {out['n_nodes']} nodes | baseline={out['baseline_score']:.4f} "
        f"| run_best={out['run_best']:.4f} | TARGET={out['target']:.4f}"
    )
    lines.append(
        "default-beta run; ranking fitness = pareto.reward (see beta_sweep.json)\n"
    )

    h = out["heuristic"]
    lines.append(
        f"OptimalPolicy  best={h['best']:.4f}  total={h['total_budget']}  "
        f"sequential={h.get('effective_sequential_rounds')}  "
        f"reached={h['reached_target']}"
    )
    return "\n".join(lines) + "\n"


def _parse_beta_grid() -> list[float]:
    raw = os.environ.get("DRSI_REPLAY_BETA_GRID", "0.0,0.25,0.5,0.75,1.0").strip()
    grid: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            grid.append(round(float(tok), 6))
        except ValueError:
            continue
    return grid or [0.0, 0.5, 1.0]


def _sweep_point(res: dict) -> dict:
    """Condense one full eval result (single- or multi-trace) into a sweep row."""
    h = res.get("heuristic") or {}
    if res.get("trace_pool"):
        agg = res.get("aggregate") or {}
        per = [
            {
                "name": r.get("name"),
                "best": (r.get("heuristic") or {}).get("best"),
                "total_budget": (r.get("heuristic") or {}).get("total_budget"),
                "decision_rounds": (r.get("heuristic") or {}).get("decision_rounds"),
                "effective_sequential_rounds": (
                    (r.get("heuristic") or {}).get("effective_sequential_rounds")
                ),
                "batch_sizes": (r.get("heuristic") or {}).get("batch_sizes"),
                "reached": (r.get("heuristic") or {}).get("reached_target"),
                "run_best": r.get("run_best"),
                "n_evaluated": r.get("n_evaluated"),
                "grid_plan": r.get("grid_plan"),
            }
            for r in res.get("traces") or []
        ]
        return {
            "best_max": h.get("best"),
            "total_budget_median": agg.get("total_budget_median"),
            "total_budget_mean": agg.get("total_budget_mean"),
            "n_reached": agg.get("n_reached"),
            "n_traces": agg.get("n_traces"),
            "per_trace": per,
        }
    return {
        "best_max": h.get("best"),
        "total_budget_median": h.get("total_budget"),
        "total_budget_mean": h.get("total_budget"),
        "n_reached": 1 if h.get("reached_target") else 0,
        "n_traces": 1,
        "per_trace": [
            {
                "name": res.get("name") or res.get("trace"),
                "best": h.get("best"),
                "total_budget": h.get("total_budget"),
                "decision_rounds": h.get("decision_rounds"),
                "effective_sequential_rounds": h.get("effective_sequential_rounds"),
                "batch_sizes": h.get("batch_sizes"),
                "reached": h.get("reached_target"),
                "run_best": res.get("run_best"),
                "n_evaluated": res.get("n_evaluated"),
                "grid_plan": res.get("grid_plan"),
            }
        ],
    }


# Minimum evaluated (probeable) nodes for a trace to carry budget<->attainment signal.
# With fewer, budget == exhaustive and the normalized AUC is forced to 0 for any policy.
_MIN_EVAL_FOR_AUC = 2


def _trace_auc(pts: list[tuple[float, float, float]], ref_budget: float) -> tuple[float, list]:
    """Normalized budget<->attainment hypervolume for one trace across betas.

    ``pts`` are ``(budget, attainment, beta)`` with attainment in [0, 1]. We keep the
    non-dominated frontier (cheapest budget for each attainment level) and integrate the
    area it dominates versus the exhaustive reference ``(ref_budget, attain=0)``,
    normalized to [0, 1]. Reaching the ceiling with few probes -> ~1; needing the full
    exhaustive budget (parallel-refine) -> ~0; never scoring -> 0.
    """
    if ref_budget <= 0 or not pts:
        return 0.0, []
    best_by_budget: dict[float, tuple[float, float]] = {}
    for b, a, beta in sorted(pts, key=lambda p: (p[0], -p[1])):
        if b not in best_by_budget or a > best_by_budget[b][0]:
            best_by_budget[b] = (a, beta)
    front: list[tuple[float, float, float]] = []
    amax = -1.0
    for b in sorted(best_by_budget):
        a, beta = best_by_budget[b]
        if a > amax:  # strictly better attainment than anything cheaper -> on frontier
            front.append((b, a, beta))
            amax = a
    hv = 0.0
    prev_a = 0.0
    for b, a, _ in front:
        hv += (a - prev_a) * max(0.0, ref_budget - b)
        prev_a = a
    return hv / ref_budget, [
        {"beta": beta, "total_budget": b, "attain": a} for b, a, beta in front
    ]


def _pareto_summary(grid: list[dict]) -> dict:
    """Fitness = AUC minus normalized worker-capped sequential-round penalty.

    We do NOT aggregate best across the pool first (that let one easy trace saturate the
    old ``best_max`` axis and collapse the curve to auc=0). Instead, for each trace we
    look at how ``(budget, attainment)`` moves across the beta grid, where
    ``attainment = best / run_best`` in [0, 1] (each trace normalized by its OWN ceiling),
    take the non-dominated frontier, and integrate versus the exhaustive reference
    ``ref_budget = n_evaluated``. The fitness is the mean per-trace AUC. Traces with
    ``run_best <= 0`` or unknown ``n_evaluated`` (degenerate / all-fail, e.g. iter006) are
    skipped rather than scored. The parallel penalty is the mean, over beta values, of
    ``effective_sequential_rounds / total_budget``.  It is 1.0 for a serial policy
    and approaches ``1 / max_parallelism`` for full worker utilization.
    """
    # Pivot the per-beta grid into per-trace point sets across betas. run_best and
    # n_evaluated are trace properties (constant across betas) — keep the first seen.
    by_trace: dict[str, dict] = {}
    for g in grid:
        if g.get("status") != "ok":
            continue
        beta = g.get("beta")
        for r in g.get("per_trace") or []:
            name = r.get("name")
            if name is None:
                continue
            slot = by_trace.setdefault(
                name,
                {"run_best": None, "n_evaluated": None, "pts": []},
            )
            if slot["run_best"] is None:
                slot["run_best"] = r.get("run_best")
            if slot["n_evaluated"] is None:
                slot["n_evaluated"] = r.get("n_evaluated")
            budget, best = r.get("total_budget"), r.get("best")
            rounds = r.get("effective_sequential_rounds")
            if budget is not None and best is not None:
                slot["pts"].append((float(budget), float(best), beta, rounds))

    per_trace: list[dict] = []
    for name, slot in by_trace.items():
        run_best, n_eval = slot["run_best"], slot["n_evaluated"]
        # Skip traces with no budget<->attainment trade-off to measure:
        #  - run_best <= 0: degenerate / all-fail (can't normalize attainment).
        #  - n_eval < 2: only one probeable node, so budget==exhaustive always and the
        #    AUC is mathematically forced to 0 for any policy -> no signal, would only
        #    dilute the mean.
        if not run_best or run_best <= 0 or not n_eval or n_eval < _MIN_EVAL_FOR_AUC:
            continue
        pts = [
            (budget, max(0.0, min(best / run_best, 1.0)), beta)
            for (budget, best, beta, _) in slot["pts"]
        ]
        auc_t, front = _trace_auc(pts, float(n_eval))
        ratios = [
            float(rounds) / budget
            for (budget, _, _, rounds) in slot["pts"]
            if rounds is not None and budget > 0
        ]
        parallel_penalty = sum(ratios) / len(ratios) if ratios else 1.0
        per_trace.append(
            {
                "name": name,
                "run_best": run_best,
                "n_evaluated": n_eval,
                "auc": auc_t,
                "parallel_penalty": parallel_penalty,
                "front": front,
            }
        )

    if not per_trace:
        return {
            "auc": 0.0,
            "parallel_penalty": 1.0,
            "parallelism_weight": _parallelism_penalty_weight(),
            "reward": -_parallelism_penalty_weight(),
            "n_traces_scored": 0,
            "per_trace": [],
            "note": f"no trace with run_best>0 and n_evaluated>={_MIN_EVAL_FOR_AUC} to score",
        }

    mean_auc = sum(t["auc"] for t in per_trace) / len(per_trace)
    mean_penalty = sum(t["parallel_penalty"] for t in per_trace) / len(per_trace)
    weight = _parallelism_penalty_weight()
    return {
        "auc": mean_auc,
        "parallel_penalty": mean_penalty,
        "parallelism_weight": weight,
        "reward": mean_auc - weight * mean_penalty,
        "method": (
            "mean per-trace normalized budget<->attainment hypervolume "
            "minus worker-capped sequential-round penalty"
        ),
        "n_traces_scored": len(per_trace),
        "per_trace": per_trace,
    }


def run_beta_sweep(
    method_file: Path,
    *,
    pool_dir: Path | None,
    trace_src: Path | None,
    target_spec: str,
    betas: list[float],
) -> dict:
    """Run the policy at each beta and collect the budget/best trade-off per trace."""
    grid: list[dict] = []
    execution_traces: list[dict] = []
    for beta in betas:
        cfg = {"beta": beta}
        try:
            if pool_dir is not None:
                trace_paths = list_pool_traces(Path(pool_dir))
                if not trace_paths:
                    raise FileNotFoundError(f"no traces in pool: {pool_dir}")
                res = evaluate_multi_trace(
                    method_file,
                    trace_paths,
                    target_spec=target_spec,
                    config=cfg,
                    include_execution_trace=True,
                )
            else:
                res = evaluate_on_trace(
                    method_file,
                    trace_src,
                    target_spec=target_spec,
                    config=cfg,
                    include_execution_trace=True,
                )
            execution_traces.extend(_execution_trace_episodes(res, beta=beta))
            point = _sweep_point(res)
            point["status"] = res.get("status", "ok")
            if point["status"] != "ok":
                point["error"] = res.get("note", "planner replay evaluation failed")
        except Exception as e:  # keep sweeping the remaining betas
            point = {"status": "error", "error": repr(e)}
        point["beta"] = beta
        grid.append(point)
    pareto = _pareto_summary(grid)
    return {
        "method_file": str(method_file.resolve()),
        "target_spec": target_spec,
        "max_parallelism": _replay_parallelism(),
        "parallelism_penalty_weight": _parallelism_penalty_weight(),
        "betas": betas,
        "pareto": pareto,
        "grid": grid,
        "execution_traces": execution_traces,
    }


def _execution_trace_episodes(result: dict, *, beta: float) -> list[dict]:
    """Extract one serializable decision episode per trace from an eval result."""
    rows = result.get("traces") if result.get("trace_pool") else [result]
    episodes: list[dict] = []
    for row in rows or []:
        if row.get("status") != "ok":
            continue
        episode = row.get("execution_trace")
        if not isinstance(episode, dict):
            continue
        episodes.append(
            {
                "trace": row.get("trace"),
                "name": row.get("name"),
                "beta": beta,
                "run_best": row.get("run_best"),
                "n_evaluated": row.get("n_evaluated"),
                **episode,
            }
        )
    return episodes


def _write_execution_traces(path: Path, episodes: list[dict]) -> None:
    """Write independently readable replay episodes without bloating beta_sweep.json."""
    payload = "".join(
        json.dumps(episode, ensure_ascii=False, separators=(",", ":")) + "\n"
        for episode in episodes
    )
    path.write_text(payload, encoding="utf-8")


def _required(value, env_name: str, flag: str) -> str:
    value = value or os.environ.get(env_name, "").strip()
    if not value:
        raise SystemExit(f"eval: pass {flag} or set {env_name}")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a discovered exploration policy on frozen traces")
    parser.add_argument("method_file", nargs="?", default=None)
    parser.add_argument("trace", nargs="?", default=None, help="single trace file or workspace")
    parser.add_argument(
        "--trace-pool",
        type=Path,
        default=None,
        help="directory containing iter*/raw_scores.json frozen traces",
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--target", default=None, help='float, "run_best", or "baseline+0.02"')
    parser.add_argument(
        "--beta-sweep",
        action="store_true",
        help="also sweep config beta over DRSI_REPLAY_BETA_GRID and write beta_sweep.json",
    )
    args = parser.parse_args()

    method_file = Path(
        _required(args.method_file, "DRSI_REPLAY_METHOD_FILE", "method_file")
    ).resolve()
    result_dir = Path(
        _required(args.result_dir, "DRSI_REPLAY_RESULT_DIR", "--result-dir")
    ).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    target_spec = args.target or os.environ.get("DRSI_REPLAY_TARGET", "run_best")

    pool_dir = args.trace_pool
    if pool_dir is None:
        pool_env = os.environ.get("DRSI_REPLAY_TRACE_POOL", "").strip()
        if pool_env:
            pool_dir = Path(pool_env)
    trace_src: Path | None = None
    if pool_dir is None:
        trace_src = Path(
            _required(args.trace, "DRSI_REPLAY_TRACE", "a trace or --trace-pool")
        ).resolve()

    rc = 0
    try:
        if pool_dir is not None:
            trace_paths = list_pool_traces(Path(pool_dir))
            if not trace_paths:
                raise FileNotFoundError(f"no traces in pool: {pool_dir}")
            out = evaluate_multi_trace(method_file, trace_paths, target_spec=target_spec)
            if out.get("status") != "ok":
                rc = 1
        else:
            row = evaluate_on_trace(method_file, trace_src, target_spec=target_spec)
            out = {
                "method_file": str(method_file),
                "trace": str(_resolve_trace_path(trace_src)),
                "n_nodes": row["n_nodes"],
                "baseline_score": row["baseline_score"],
                "run_best": row["run_best"],
                "target": row["target"],
                "heuristic": row["heuristic"],
                "status": row["status"],
            }
            if row.get("note"):
                out["note"] = row["note"]
    except Exception as e:
        out = {
            "method_file": str(method_file),
            "status": "error",
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "heuristic": None,
        }
        rc = 1

    (result_dir / "heuristic_eval.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "heuristic_eval.txt").write_text(_render(out), encoding="utf-8")
    print(_render(out))

    sweep_on = args.beta_sweep or os.environ.get("DRSI_REPLAY_BETA_SWEEP", "").strip() not in (
        "",
        "0",
        "false",
        "False",
    )
    if sweep_on:
        try:
            sweep = run_beta_sweep(
                method_file,
                pool_dir=pool_dir,
                trace_src=trace_src,
                target_spec=target_spec,
                betas=_parse_beta_grid(),
            )
        except Exception as e:
            sweep = {"status": "error", "error": repr(e), "traceback": traceback.format_exc()}
        execution_traces = sweep.pop("execution_traces", [])
        trace_file = result_dir / "policy_execution_traces.jsonl"
        _write_execution_traces(trace_file, execution_traces)
        sweep["execution_trace_file"] = trace_file.name
        sweep["n_execution_episodes"] = len(execution_traces)
        (result_dir / "beta_sweep.json").write_text(
            json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pareto = sweep.get("pareto") or {}
        reward = pareto.get("reward")
        auc = pareto.get("auc")
        reward_s = f"{reward:.4g}" if isinstance(reward, (int, float)) else "-"
        auc_s = f"{auc:.4g}" if isinstance(auc, (int, float)) else "-"
        print(f"[beta-sweep] wrote {result_dir / 'beta_sweep.json'} "
              f"({len(sweep.get('grid', []))} beta points, "
              f"pareto.reward={reward_s}, auc={auc_s}, "
              f"execution_episodes={len(execution_traces)})")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
