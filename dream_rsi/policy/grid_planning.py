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

"""Resolve optional policy grid plans before a live/replay environment is built."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dream_rsi.env.trace.raw import trace_from_raw_file
from dream_rsi.policy.api import GridPlan, GridPlanningContext, LLMDesignedMethod


@dataclass(frozen=True)
class ResolvedGridPlan:
    """Validated policy request plus the grid shape the environment will use."""

    requested: GridPlan | None
    effective: GridPlan
    source: str
    fallback_reason: str | None = None
    clamp_reasons: tuple[str, ...] = ()
    out_of_support: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "requested": self.requested.to_dict() if self.requested is not None else None,
            "effective": self.effective.to_dict(),
            "fallback_reason": self.fallback_reason,
            "clamp_reasons": list(self.clamp_reasons),
            "out_of_support": self.out_of_support,
        }


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _coerce_plan(value: Any) -> GridPlan:
    if isinstance(value, GridPlan):
        plan = value
    elif isinstance(value, Mapping):
        plan = GridPlan(
            branch_count=value.get("branch_count"),
            refine_count=value.get("refine_count"),
            reason=value.get("reason", ""),
            confidence=value.get("confidence"),
        )
    else:
        raise TypeError("plan_grid() must return GridPlan, a mapping, or None")

    if not _is_int(plan.branch_count) or plan.branch_count < 1:
        raise ValueError("branch_count must be an integer >= 1")
    if not _is_int(plan.refine_count) or plan.refine_count < 0:
        raise ValueError("refine_count must be an integer >= 0")
    if not isinstance(plan.reason, str):
        raise ValueError("reason must be a string")
    if plan.confidence is not None and (
        isinstance(plan.confidence, bool) or not isinstance(plan.confidence, (int, float))
    ):
        raise ValueError("confidence must be numeric or None")
    return plan


def _fallback(context: GridPlanningContext) -> GridPlan:
    return GridPlan(
        branch_count=context.fallback_branch_count,
        refine_count=context.fallback_refine_count,
        reason="fallback grid",
    )


def resolve_grid_plan(
    policy: LLMDesignedMethod,
    context: GridPlanningContext,
    *,
    support_branch_count: int | None = None,
    support_refine_count: int | None = None,
) -> ResolvedGridPlan:
    """Call a policy planner and enforce fallback, hard caps, and replay support.

    A policy may request any integer shape. Runtime always enforces configured
    hard caps. Replay additionally marks a request larger than the frozen trace
    as ``out_of_support`` so that an unobservable expansion cannot earn a reward.
    """
    fallback = _fallback(context)
    try:
        raw = policy.plan_grid(context)
    except Exception as exc:
        return ResolvedGridPlan(
            requested=None,
            effective=fallback,
            source="fallback",
            fallback_reason=f"planner_error:{type(exc).__name__}",
        )
    if raw is None:
        return ResolvedGridPlan(
            requested=None,
            effective=fallback,
            source="fallback",
            fallback_reason="planner_returned_none",
        )

    try:
        requested = _coerce_plan(raw)
    except (TypeError, ValueError) as exc:
        return ResolvedGridPlan(
            requested=None,
            effective=fallback,
            source="fallback",
            fallback_reason=f"invalid_plan:{exc}",
        )

    branch_limit = context.hard_max_branch_count
    refine_limit = context.hard_max_refine_count
    out_of_support = False
    reasons: list[str] = []
    if support_branch_count is not None:
        if requested.branch_count > support_branch_count:
            out_of_support = True
            reasons.append("requested_branch_count_exceeds_trace_support")
        branch_limit = min(branch_limit, support_branch_count)
    if support_refine_count is not None:
        if requested.refine_count > support_refine_count:
            out_of_support = True
            reasons.append("requested_refine_count_exceeds_trace_support")
        refine_limit = min(refine_limit, support_refine_count)

    effective_width = min(requested.branch_count, branch_limit)
    effective_refine = min(requested.refine_count, refine_limit)
    if effective_width != requested.branch_count:
        reasons.append("branch_count_clamped")
    if effective_refine != requested.refine_count:
        reasons.append("refine_count_clamped")
    effective = GridPlan(
        branch_count=effective_width,
        refine_count=effective_refine,
        reason=requested.reason,
        confidence=requested.confidence,
    )
    return ResolvedGridPlan(
        requested=requested,
        effective=effective,
        source="policy",
        clamp_reasons=tuple(reasons),
        out_of_support=out_of_support,
    )


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def completed_history(workspace: Path) -> tuple[dict[str, Any], ...]:
    """Read only completed, archived cycle manifests in stable iteration order."""
    root = workspace.resolve() / "history"
    if not root.is_dir():
        return ()
    rows: list[dict[str, Any]] = []
    for iteration in sorted(p for p in root.iterdir() if p.is_dir()):
        manifest = _load_manifest(iteration / "live_cycle_manifest.json")
        if manifest is not None:
            rows.append(manifest)
    return tuple(rows)


def make_live_context(
    workspace: Path,
    *,
    cycle: int,
    fallback_branch_count: int,
    fallback_refine_count: int,
    hard_max_branch_count: int,
    hard_max_refine_count: int,
    max_workers: int | None,
) -> GridPlanningContext:
    return GridPlanningContext(
        cycle=cycle,
        mode="live",
        fallback_branch_count=fallback_branch_count,
        fallback_refine_count=fallback_refine_count,
        hard_max_branch_count=hard_max_branch_count,
        hard_max_refine_count=hard_max_refine_count,
        max_workers=max_workers,
        workspace=str(workspace.resolve()),
        history=completed_history(workspace),
    )


def trace_grid_capacity(trace_path: Path) -> tuple[int, int]:
    """Return frozen trace branch count and deepest attempt index."""
    trace = trace_from_raw_file(trace_path)
    if not trace.nodes:
        return 1, 0
    return (
        max(node.branch for node in trace.nodes) + 1,
        max(node.attempt for node in trace.nodes),
    )


def make_replay_context(
    trace_path: Path,
    *,
    max_workers: int | None,
    history: tuple[dict[str, Any], ...] = (),
) -> GridPlanningContext:
    branches, refine = trace_grid_capacity(trace_path)
    return GridPlanningContext(
        cycle=len(history),
        mode="replay",
        fallback_branch_count=branches,
        fallback_refine_count=refine,
        hard_max_branch_count=branches,
        hard_max_refine_count=refine,
        max_workers=max_workers,
        workspace=str(trace_path.resolve().parent),
        history=history,
        trace_branch_count=branches,
        trace_refine_count=refine,
    )
