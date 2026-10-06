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

"""Glue: pick TraceEnv mode, call ``policy.solve(env)``."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.runner import sync
from dream_rsi.exploration.store import NodeStore
from dream_rsi.env.trace_env import LiveExecutor, TraceEnv
from dream_rsi.env.trace_store import grid_limits
from dream_rsi.env.types import SearchEnv
from dream_rsi.learning.directions.context import resolve_direction_mode
from dream_rsi.policy.api import GridPlan, LLMDesignedMethod, SimResult
from dream_rsi.policy.grid_planning import (
    ResolvedGridPlan,
    completed_history,
    make_live_context,
    make_replay_context,
    resolve_grid_plan,
    trace_grid_capacity,
)


@dataclass
class ExploreRun:
    sim: SimResult
    node_ids: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    trace_path: str | None = None
    stopped_reason: str = "done"
    grid_plan: dict[str, Any] = field(default_factory=dict)
    direction_mode: str = "off"
    direction_summary_id: str | None = None
    direction_context_active: bool = False


def run_exploration(
    policy: LLMDesignedMethod,
    env: SearchEnv,
    *,
    budget: int | None = None,
) -> SimResult:
    return policy.solve(env, budget=budget)  # type: ignore[arg-type]


def run_replay(
    policy: LLMDesignedMethod,
    trace_path: Path | str,
    *,
    budget: int | None = None,
) -> SimResult:
    frozen = Path(trace_path).resolve()
    context = make_replay_context(frozen, max_workers=None)
    resolved = resolve_grid_plan(
        policy,
        context,
        support_branch_count=context.trace_branch_count,
        support_refine_count=context.trace_refine_count,
    )
    env = TraceEnv.replay(frozen, grid_plan=resolved.effective)
    return run_exploration(policy, env, budget=budget)


def run_live(
    workspace: Path,
    agent: Any,
    *,
    policy: LLMDesignedMethod | None = None,
    policy_source: str | Path | None = None,
    config: ExploreConfig | None = None,
    cycle: int = 0,
    budget: int | None = None,
    trace_path: Path | None = None,
    max_width: int | None = None,
    max_refine: int | None = None,
    live_executor: LiveExecutor | None = None,
    max_workers: int | None = None,
    prepare_workspace: bool = True,
    direction_mode: str | None = None,
) -> ExploreRun:
    """Online explore: ``TraceEnv`` live mode + ``policy.solve(env)``.

    Pass ``policy`` directly (tests) or ``policy_source`` (CLI / orchestrate).
    """
    if policy is None and policy_source is None:
        raise ValueError("pass policy or policy_source")

    ws = workspace.resolve()
    config = config or ExploreConfig.from_env()

    if prepare_workspace:
        ws.mkdir(parents=True, exist_ok=True)
        if config.problem is not None and config.problem.is_file():
            shutil.copy2(config.problem, ws / "problem.txt")
        if not (ws / "problem.txt").is_file():
            raise FileNotFoundError(f"Missing {ws / 'problem.txt'}")
        sync(ws)

    if policy is None:
        from dream_rsi.policy import load_policy

        policy_cfg: dict[str, Any] = {}
        if max_width is not None:
            policy_cfg["max_width"] = max_width
        if max_refine is not None:
            policy_cfg["max_refine"] = max_refine
        policy = load_policy(policy_source, config=policy_cfg)

    fallback_width, fallback_refine = grid_limits(
        max_width=max_width,
        max_refine=max_refine,
    )
    existing_trace = trace_path if trace_path is not None else ws / "raw_scores.json"
    if existing_trace.is_file():
        # A persisted raw trace is a resume, not a new cycle. Its structure is
        # immutable, so never claim a freshly generated plan changed it.
        width, refine = trace_grid_capacity(existing_trace)
        resolved = ResolvedGridPlan(
            requested=None,
            effective=GridPlan(
                branch_count=width,
                refine_count=refine,
                reason="existing frozen grid",
            ),
            source="existing_grid",
            fallback_reason="existing_raw_scores",
        )
    else:
        planning_context = make_live_context(
            ws,
            cycle=cycle,
            fallback_branch_count=fallback_width,
            fallback_refine_count=fallback_refine,
            hard_max_branch_count=fallback_width,
            hard_max_refine_count=fallback_refine,
            max_workers=max_workers,
        )
        if planning_context.history:
            resolved = resolve_grid_plan(policy, planning_context)
        else:
            resolved = ResolvedGridPlan(
                requested=None,
                effective=GridPlan(
                    branch_count=fallback_width,
                    refine_count=fallback_refine,
                    reason="first-cycle fallback grid",
                ),
                source="fallback",
                fallback_reason="no_completed_history",
            )

    from dream_rsi.exploration.history_seed import seed_history_to_branches

    seed_history_to_branches(
        ws,
        max_width=resolved.effective.branch_count,
        max_refine=resolved.effective.refine_count,
    )

    from dream_rsi.learning.directions.context import (
        GLOBAL_DIRECTION_MODE,
        load_global_direction_context,
    )

    resolved_direction_mode = resolve_direction_mode(direction_mode)
    direction_context: str | None = None
    direction_summary_id: str | None = None
    if resolved_direction_mode == GLOBAL_DIRECTION_MODE:
        try:
            direction_context, catalog = load_global_direction_context(ws)
        except FileNotFoundError:
            # E0 is explicitly the unguided bootstrap: its post-explore
            # summarization creates the first context for E1.  Every later
            # global cycle is strict so an intended guided experiment cannot
            # silently degrade to ordinary exploration.
            if cycle == 0 and not completed_history(ws):
                print(
                    "[directions] global bootstrap E0: no prior direction context; "
                    "running default explore before the first summary",
                    flush=True,
                )
            else:
                raise
        else:
            direction_summary_id = catalog.summary_id
            print(
                f"[directions] global context loaded: summary={catalog.summary_id}, "
                f"directions={len(catalog.directions)}, candidates={len(catalog.candidates)}",
                flush=True,
            )

    env = TraceEnv.live(
        ws,
        agent,
        config=config,
        cycle=cycle,
        trace_path=trace_path,
        max_width=resolved.effective.branch_count,
        max_refine=resolved.effective.refine_count,
        live_executor=live_executor,
        max_workers=max_workers,
        direction_context=direction_context,
        grid_plan=resolved.effective,
    )
    sim = run_exploration(policy, env, budget=budget)

    store = NodeStore(ws)
    node_ids: list[str] = []
    failed: list[str] = []
    for nid in sim.order:
        try:
            if store.read_meta(nid).status == "done":
                node_ids.append(nid)
            else:
                failed.append(nid)
        except (FileNotFoundError, OSError):
            failed.append(nid)

    tp = str(env.store.path) if env.store.path else None
    selected_meta = [env.meta(nid) for nid in sim.order]
    plan_record = resolved.to_dict()
    plan_record["actual"] = {
        "opened_branch_count": len({meta.branch for meta in selected_meta}),
        "max_attempt_selected": max((meta.attempt for meta in selected_meta), default=None),
        "probe_count": sim.budget,
        "decision_rounds": sim.decision_rounds,
        "effective_sequential_rounds": sim.effective_sequential_rounds,
    }
    return ExploreRun(
        sim=sim,
        node_ids=sorted(node_ids),
        failed=sorted(failed),
        trace_path=tp,
        stopped_reason="done",
        grid_plan=plan_record,
        direction_mode=resolved_direction_mode,
        direction_summary_id=direction_summary_id,
        direction_context_active=direction_context is not None,
    )
