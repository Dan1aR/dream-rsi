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

"""Offline learning helpers: trace export and ``TraceEnv`` replay of policies.

Replay uses :class:`dream_rsi.env.TraceEnv` in ``mode="replay"`` — the same surface
online explore uses in live mode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dream_rsi.policy import LLMDesignedMethod, ParallelRefineMethod, SimResult
from dream_rsi.policy.load import load_policy_class, promoted_policy_path
from dream_rsi.env.trace.export import default_out_path
from dream_rsi.env.trace.raw import (
    RawTrace,
    export_raw_scores,
    load_raw_scores,
    raw_trace_to_trace,
    write_raw_scores,
)
from dream_rsi.env.trace.types import Node, Trace

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TraceInfo:
    source: str
    n_nodes: int
    n_branches: int
    max_attempt: int
    baseline_score: float
    run_best: float
    best_node: str


@dataclass
class MethodReport:
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    budget: int = 0
    best: float = 0.0
    reach_run_best: int | None = None
    hit_run_best: bool = False
    gap: float = 0.0
    order_preview: list[str] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


@dataclass
class LearningResult:
    workspace: str
    trace_export: str | None
    trace: TraceInfo
    parallel_refine: MethodReport
    discovered: MethodReport | None


def infer_baseline_config(raw: RawTrace) -> dict[str, int]:
    """Infer replay grid limits from a frozen trace.

    ``max_refine`` is the deepest attempt index, not the number of depth levels.
    A 32 x 20 trace therefore resolves to ``max_width=32, max_refine=19``.
    """
    if not raw.nodes:
        return {}
    return {
        "max_width": max(node.branch for node in raw.nodes) + 1,
        "max_refine": max(node.attempt for node in raw.nodes),
    }


def resolve_baseline_config(
    overrides: dict[str, Any] | None = None,
    *,
    trace: RawTrace | None = None,
) -> dict[str, Any]:
    """Resolve the parallel-refine baseline's grid limits.

    Explicit overrides (the online ``--branches``/``--refine`` caps) win over
    the limits inferred from the frozen trace shape.
    """
    cfg: dict[str, Any] = {}
    if trace is not None:
        cfg.update(infer_baseline_config(trace))
    if overrides:
        cfg.update(overrides)
    return cfg


def resolve_method_file(
    method_file: Path | None = None,
    *,
    workspace: Path | None = None,
) -> Path | None:
    """Explicit ``method_file``, else the workspace's promoted policy, else ``None``."""
    if method_file is not None:
        p = method_file.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"method file not found: {p}")
        return p
    if workspace is not None:
        promoted = promoted_policy_path(workspace)
        if promoted.is_file():
            return promoted
    return None


def load_raw(source: str | Path) -> RawTrace:
    """Load the canonical raw trace (file → load, workspace dir → export)."""
    src = Path(source).resolve()
    if not src.exists():
        raise FileNotFoundError(f"trace source not found: {src}")
    return load_raw_scores(src) if src.is_file() else export_raw_scores(src)


def export_workspace_trace(workspace: str | Path, *, out: Path | None = None) -> Path:
    ws = Path(workspace).resolve()
    if not ws.is_dir():
        raise NotADirectoryError(f"workspace not found: {ws}")
    raw = export_raw_scores(ws)
    dest = out.resolve() if out else default_out_path(ws)
    write_raw_scores(raw, dest)
    return dest


def _best_node(trace: Trace) -> Node:
    return max(trace.nodes, key=lambda n: n.score)


def _order_preview(order: list[str], *, head: int = 8, tail: int = 4) -> list[str]:
    if len(order) <= head + tail:
        return list(order)
    return order[:head] + ["..."] + order[-tail:]


def _check_replay_invariants(result: SimResult, raw: RawTrace) -> None:
    if result.budget != len(result.order):
        raise AssertionError(f"budget {result.budget} != len(order) {len(result.order)}")
    if len(result.order) != len(set(result.order)):
        raise AssertionError("duplicate ids in reveal order")
    trace = raw_trace_to_trace(raw)
    if result.best > trace.best_score + 1e-9:
        raise AssertionError(f"best {result.best} > trace run-best {trace.best_score}")
    from dream_rsi.env.trace_env import TraceEnv
    from dream_rsi.env.trace_store import TraceStore

    env = TraceEnv(TraceStore.from_raw(raw), "replay")
    for nid in result.order:
        if not env.is_legal(nid):
            raise AssertionError(f"illegal action: {nid}")
        env.probe(nid)
    if abs(result.best - env.best_so_far) > 1e-9:
        raise AssertionError(f"best {result.best} != replay {env.best_so_far}")


def run_method(
    method: LLMDesignedMethod,
    raw: RawTrace,
    *,
    name: str,
    config: dict[str, Any] | None = None,
    budget: int | None = None,
) -> MethodReport:
    cfg = dict(config or {})
    try:
        # Lazy import keeps dream_rsi.learning ⇄ dream_rsi.env import-time acyclic.
        from dream_rsi.env.trace_env import TraceEnv
        from dream_rsi.env.trace_store import TraceStore

        trace = raw_trace_to_trace(raw)
        env = TraceEnv(TraceStore.from_raw(raw), "replay")
        result = method.solve(env, budget=budget)
        _check_replay_invariants(result, raw)
        run_best = trace.best_score
        gap = run_best - result.best
        return MethodReport(
            name=name,
            config=cfg,
            budget=result.budget,
            best=result.best,
            reach_run_best=result.budget_to_reach(run_best),
            hit_run_best=gap <= 1e-9,
            gap=gap,
            order_preview=_order_preview(result.order),
            ok=True,
        )
    except Exception as exc:
        return MethodReport(name=name, config=cfg, ok=False, error=f"{type(exc).__name__}: {exc}")


def run_parallel_refine(
    raw: RawTrace,
    *,
    config: dict[str, Any] | None = None,
    budget: int | None = None,
) -> MethodReport:
    cfg = dict(config or {})
    return run_method(
        ParallelRefineMethod(cfg),
        raw,
        name="parallel_refine",
        config=cfg,
        budget=budget,
    )


def run_discovered(
    raw: RawTrace,
    *,
    method_file: Path,
    budget: int | None = None,
) -> MethodReport:
    path = method_file.resolve()
    try:
        Policy = load_policy_class(path)
        return run_method(Policy(), raw, name="OptimalPolicy", config={"method_file": str(path)}, budget=budget)
    except Exception as exc:
        return MethodReport(
            name="OptimalPolicy",
            config={"method_file": str(path)},
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def trace_info(trace: Trace, *, source: str) -> TraceInfo:
    best = _best_node(trace)
    return TraceInfo(
        source=source,
        n_nodes=len(trace.nodes),
        n_branches=len(trace.branches),
        max_attempt=max((n.attempt for n in trace.nodes), default=0),
        baseline_score=trace.baseline_score,
        run_best=trace.best_score,
        best_node=best.id,
    )


def run_learning(
    workspace: str | Path | None = None,
    *,
    trace_source: str | Path | None = None,
    export: bool = True,
    method_file: Path | None = None,
    baseline_config: dict[str, Any] | None = None,
    budget: int | None = None,
) -> LearningResult:
    """Replay the baseline and a discovered policy on one workspace or trace.

    The discovered policy is ``method_file``, else the workspace's promoted
    policy; it is skipped when neither exists.
    """
    if trace_source is None and workspace is None:
        raise ValueError("pass workspace or trace_source")

    export_path: Path | None = None
    if trace_source is not None:
        src = Path(trace_source).resolve()
        ws_label = Path(workspace).resolve() if workspace else src.parent
    else:
        ws = Path(workspace).resolve()
        if not ws.is_dir():
            raise NotADirectoryError(f"workspace not found: {ws}")
        ws_label = ws
        if export:
            export_path = export_workspace_trace(ws)
            src = export_path
        else:
            src = ws

    raw = load_raw(src)
    trace = raw_trace_to_trace(raw)
    info = trace_info(trace, source=str(src))
    cfg = resolve_baseline_config(baseline_config, trace=raw)
    policy_file = resolve_method_file(
        method_file, workspace=Path(workspace) if workspace else None
    )

    return LearningResult(
        workspace=str(ws_label),
        trace_export=str(export_path) if export_path else None,
        trace=info,
        parallel_refine=run_parallel_refine(raw, config=cfg, budget=budget),
        discovered=(
            run_discovered(raw, method_file=policy_file, budget=budget)
            if policy_file is not None
            else None
        ),
    )


def format_learning_result(result: LearningResult) -> str:
    t = result.trace
    lines = [
        f"workspace   : {result.workspace}",
        f"trace       : {t.source}",
        f"nodes       : {t.n_nodes}  branches={t.n_branches}  max_attempt={t.max_attempt}",
        f"scores      : baseline={t.baseline_score:.6f}  run_best={t.run_best:.6f} ({t.best_node})",
    ]
    if result.trace_export:
        lines.append(f"exported    : {result.trace_export}")

    for label, rep in (
        ("parallel_refine", result.parallel_refine),
        ("OptimalPolicy", result.discovered),
    ):
        lines.append("")
        if rep is None:
            lines.append(f"[skip] {label}: no policy file (pass --method)")
            continue
        if not rep.ok:
            lines.append(f"[FAIL] {label}: {rep.error}")
            continue
        reach = str(rep.reach_run_best) if rep.reach_run_best is not None else "-"
        hit = "yes" if rep.hit_run_best else "no"
        cfg_s = rep.config if rep.config else "(defaults)"
        lines.append(f"[{label}] config={cfg_s}")
        lines.append(f"  budget={rep.budget}  best={rep.best:.6f}  reach_run_best={reach}  hit={hit}")
        if not rep.hit_run_best:
            lines.append(f"  gap={rep.gap:.6f}")
        lines.append(f"  order={rep.order_preview}")
    return "\n".join(lines) + "\n"


def learning_result_to_dict(result: LearningResult) -> dict[str, Any]:
    return asdict(result)
