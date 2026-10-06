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

"""Unified :class:`SearchEnv`: same trace tree, ``probe`` = reveal or execute+reveal."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from typing import Any, Callable, Literal

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.prompt import prepare_baseline
from dream_rsi.exploration.runner import _run_one_node, sync
from dream_rsi.exploration.store import NodeStore
from dream_rsi.env.trace_store import TraceStore
from dream_rsi.env.observation import observation_from_node
from dream_rsi.env.types import CellMeta, Observation
from dream_rsi.env.trace.raw import raw_trace_to_trace
from dream_rsi.env.trace.types import Trace
from dream_rsi.policy.api import GridPlan

ExperimentMode = Literal["replay", "live"]

# Optional test hook: ``(env, cell_id) -> None`` runs instead of real agent.
LiveExecutor = Callable[["TraceEnv", str], None]


def _sort_cells(env: TraceEnv, cell_ids: list[str]) -> list[str]:
    return sorted(cell_ids, key=lambda c: (env.meta(c).branch, env.meta(c).seq))


class TraceEnv:
    """One search surface over ``raw_scores.json`` — replay reveals, live executes."""

    def __init__(
        self,
        store: TraceStore,
        mode: ExperimentMode,
        *,
        workspace: Path | None = None,
        agent: Any = None,
        config: ExploreConfig | None = None,
        cycle: int = 0,
        baseline_bundle: Path | None = None,
        live_executor: LiveExecutor | None = None,
        max_workers: int | None = None,
        direction_context: str | None = None,
        grid_plan: GridPlan | None = None,
    ) -> None:
        self.store = store
        self.mode = mode
        self.workspace = workspace.resolve() if workspace is not None else None
        self.agent = agent
        self.config = config
        self.cycle = cycle
        self.baseline_bundle = baseline_bundle
        self.live_executor = live_executor
        self.max_workers = max_workers
        self.direction_context = direction_context
        self.grid_plan = grid_plan
        self.last_observation: Observation | None = None
        self._trace: Trace = raw_trace_to_trace(store.raw)
        self.reset()

    @classmethod
    def replay(
        cls,
        trace_path: Path,
        *,
        max_workers: int | None = None,
        grid_plan: GridPlan | None = None,
    ) -> TraceEnv:
        return cls(
            TraceStore.from_frozen(trace_path),
            "replay",
            max_workers=max_workers,
            grid_plan=grid_plan,
        )

    @classmethod
    def live(
        cls,
        workspace: Path,
        agent: Any,
        *,
        config: ExploreConfig | None = None,
        cycle: int = 0,
        trace_path: Path | None = None,
        max_width: int | None = None,
        max_refine: int | None = None,
        live_executor: LiveExecutor | None = None,
        max_workers: int | None = None,
        direction_context: str | None = None,
        grid_plan: GridPlan | None = None,
    ) -> TraceEnv:
        ws = workspace.resolve()
        config = config or ExploreConfig.from_env()
        store = TraceStore.for_live(
            ws,
            trace_path=trace_path,
            max_width=max_width,
            max_refine=max_refine,
        )
        return cls(
            store,
            "live",
            workspace=ws,
            agent=agent,
            config=config,
            cycle=cycle,
            baseline_bundle=prepare_baseline(ws, config),
            live_executor=live_executor,
            max_workers=max_workers,
            direction_context=direction_context,
            grid_plan=grid_plan,
        )

    def _reload_trace(self) -> None:
        self._trace = raw_trace_to_trace(self.store.raw)

    @property
    def baseline_score(self) -> float:
        return self._trace.baseline_score

    @property
    def max_parallelism(self) -> int:
        """Configured worker cap used to price a policy's batch choices."""
        if self.max_workers is not None:
            return max(1, int(self.max_workers))
        return max(1, len(self._trace.nodes))

    def reset(self) -> None:
        self._revealed: dict[str, Observation] = {}
        self._order: list[str] = []
        self._cost_per_branch: dict[int, int] = {}
        self.budget_spent = 0
        self.best_so_far = 0.0
        self.last_observation = None
        self.decision_rounds = 0
        self.effective_sequential_rounds = 0
        self.batch_sizes: list[int] = []
        self._execution_events: list[dict[str, Any]] = []

    @property
    def revealed_order(self) -> list[str]:
        return list(self._order)

    @property
    def done(self) -> bool:
        return not self.legal_actions()

    def legal_actions(self) -> list[str]:
        out: list[str] = []
        for n in self._trace.nodes:
            if n.id in self._revealed:
                continue
            if self.grid_plan is not None and (
                n.branch >= self.grid_plan.branch_count
                or n.attempt > self.grid_plan.refine_count
            ):
                continue
            if n.parent_id is None or n.parent_id in self._revealed:
                # Replay only reveals cells that were evaluated online; tail
                # scaffold (unevaluated) is outside the frozen action space.
                if self.mode == "replay" and not self.store.is_evaluated(n.id):
                    continue
                out.append(n.id)
        return out

    def is_legal(self, cell_id: str) -> bool:
        return cell_id in self.legal_actions()

    def meta(self, cell_id: str) -> CellMeta:
        n = self._trace.by_id[cell_id]
        return CellMeta(
            id=n.id,
            branch=n.branch,
            seq=n.seq,
            attempt=n.attempt,
            parent_id=n.parent_id,
            tags=list(n.tags),
        )

    def observed(self) -> dict[str, Observation]:
        return dict(self._revealed)

    def probe(self, cell_id: str) -> Observation:
        return self.step(cell_id)

    def step(self, cell_id: str) -> Observation:
        if cell_id in self._revealed:
            raise ValueError(f"{cell_id} already revealed")
        if cell_id not in self.legal_actions():
            raise ValueError(f"illegal probe: {cell_id}")

        event = self._begin_execution_event([cell_id])
        self._record_batch([cell_id])

        if self.mode == "live":
            self._execute_live_batch([cell_id])
        elif not self.store.is_evaluated(cell_id):
            raise ValueError(f"replay cannot reveal unevaluated cell: {cell_id}")

        obs = self._reveal(cell_id)
        self._finish_execution_event(event, [self._execution_outcome(obs)])
        return obs

    def probe_batch(
        self,
        cell_ids: list[str],
        *,
        max_workers: int | None = None,
        on_reveal: Callable[[Observation], None] | None = None,
    ) -> list[Observation]:
        """One dependency wave: live runs nodes in parallel, then reveals in stable order."""
        if not cell_ids:
            return []
        ordered = _sort_cells(self, cell_ids)
        for cid in ordered:
            if cid in self._revealed:
                raise ValueError(f"{cid} already revealed")
            if cid not in self.legal_actions():
                raise ValueError(f"illegal probe in batch: {cid}")

        event = self._begin_execution_event(ordered, max_workers=max_workers)
        self._record_batch(ordered, max_workers=max_workers)

        if self.mode == "live":
            self._execute_live_batch(ordered, max_workers=max_workers)
        else:
            for cid in ordered:
                if not self.store.is_evaluated(cid):
                    raise ValueError(f"replay cannot reveal unevaluated cell: {cid}")

        out: list[Observation] = []
        outcomes: list[dict[str, Any]] = []
        for cid in ordered:
            obs = self._reveal(cid)
            out.append(obs)
            outcomes.append(self._execution_outcome(obs))
            if on_reveal is not None:
                on_reveal(obs)
        self._finish_execution_event(event, outcomes)
        return out

    def _record_batch(self, cell_ids: list[str], *, max_workers: int | None = None) -> None:
        """Record logical and worker-capped sequential cost for one probe call."""
        size = len(cell_ids)
        if size <= 0:
            return
        workers = max(1, int(max_workers or self.max_parallelism))
        self.decision_rounds += 1
        self.effective_sequential_rounds += ceil(size / workers)
        self.batch_sizes.append(size)

    def _execution_snapshot(self) -> dict[str, Any]:
        """Prefix-derived state sufficient to diagnose a policy decision offline."""
        legal = self.legal_actions()
        roots = [cell_id for cell_id in legal if self.meta(cell_id).parent_id is None]
        return {
            "observed_count": len(self._revealed),
            "budget": self.budget_spent,
            "best": self.best_so_far,
            "legal_count": len(legal),
            "root_count": len(roots),
            "opened_branches": self.opened_branches(),
        }

    def _begin_execution_event(
        self, cell_ids: list[str], *, max_workers: int | None = None
    ) -> dict[str, Any]:
        """Capture the legal prefix immediately before one probe/probe_batch call."""
        workers = max(1, int(max_workers or self.max_parallelism))
        selected = []
        for cell_id in cell_ids:
            meta = self.meta(cell_id)
            selected.append(
                {
                    "branch": meta.branch,
                    "attempt": meta.attempt,
                    "seq": meta.seq,
                    "tags": list(meta.tags),
                }
            )
        return {
            "round": self.decision_rounds + 1,
            "before": self._execution_snapshot(),
            "batch": selected,
            "parallelism": {
                "worker_cap": workers,
                "batch_size": len(cell_ids),
                "effective_sequential_rounds": ceil(len(cell_ids) / workers),
            },
        }

    def _execution_outcome(self, obs: Observation) -> dict[str, Any]:
        """Record the observation delivered to a policy after one selected cell."""
        return {
            "branch": obs.branch,
            "attempt": obs.attempt,
            "score": obs.score,
            "valid": obs.valid,
            "evaluated": obs.evaluated,
            "fail_class": obs.fail_class,
            "delta_vs_baseline": obs.delta_vs_baseline,
            "delta_vs_parent": obs.delta_vs_parent,
            "n_valid": obs.n_valid,
            "n_total": obs.n_total,
            "budget": obs.budget_spent,
            "best_after": self.best_so_far,
        }

    def _finish_execution_event(
        self, event: dict[str, Any], outcomes: list[dict[str, Any]]
    ) -> None:
        event["outcomes"] = outcomes
        event["after"] = self._execution_snapshot()
        event["parallelism"]["decision_rounds_total"] = self.decision_rounds
        event["parallelism"]["effective_sequential_rounds_total"] = (
            self.effective_sequential_rounds
        )
        self._execution_events.append(event)

    def execution_trace(self) -> dict[str, Any]:
        """Return a replayable policy-decision episode without internal env state."""
        curve = [
            {"budget": outcome["budget"], "best": outcome["best_after"]}
            for event in self._execution_events
            for outcome in event["outcomes"]
        ]
        final = self._execution_snapshot()
        final.update(
            {
                "exhausted": self.done,
                "decision_rounds": self.decision_rounds,
                "effective_sequential_rounds": self.effective_sequential_rounds,
                "batch_sizes": list(self.batch_sizes),
                "curve": curve,
            }
        )
        out = {
            "schema_version": 1,
            "mode": self.mode,
            "max_parallelism": self.max_parallelism,
            "events": deepcopy(self._execution_events),
            "final": final,
        }
        if self.grid_plan is not None:
            out["grid_plan"] = self.grid_plan.to_dict()
        return out

    def _execute_live_batch(
        self,
        cell_ids: list[str],
        *,
        max_workers: int | None = None,
    ) -> None:
        if self.workspace is None:
            raise RuntimeError("live mode requires workspace")

        if self.live_executor is not None:
            for cid in cell_ids:
                self.live_executor(self, cid)
        else:
            if self.agent is None:
                raise RuntimeError("live mode requires agent or live_executor")
            store = NodeStore(self.workspace)
            workers = max(1, max_workers or self.max_workers or len(cell_ids))
            if self.mode == "live" and len(cell_ids) > 1:
                print(
                    f"[trace-env] live batch: {len(cell_ids)} cells, "
                    f"max_workers={workers}",
                    flush=True,
                )
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {
                    pool.submit(self._run_one_cell, store, cid): cid for cid in cell_ids
                }
                for fut in as_completed(futs):
                    cid = futs[fut]
                    try:
                        fut.result()
                    except Exception as exc:
                        try:
                            store.mark_failed(cid, str(exc))
                        except Exception:
                            pass

        sync(self.workspace)
        self.store.refresh_from_workspace(self.workspace)
        self.store.flush()
        self._reload_trace()

    def _run_one_cell(self, store: NodeStore, cell_id: str) -> None:
        n = self._trace.by_id[cell_id]
        _run_one_node(
            store,
            self.agent,
            branch=n.branch,
            seq=n.seq,
            parent_id=n.parent_id,
            intent=n.intent,
            tags=list(n.tags),
            ws=self.workspace,
            cycle=self.cycle,
            config=self.config,
            baseline_bundle=self.baseline_bundle,
            direction_context=self.direction_context,
        )

    def _reveal(self, cell_id: str) -> Observation:
        n = self._trace.by_id[cell_id]
        parent_score = self._revealed[n.parent_id].score if n.parent_id else None
        self.budget_spent += 1
        obs = observation_from_node(
            n,
            baseline=self.baseline_score,
            parent_score=parent_score,
            budget_spent=self.budget_spent,
        )
        self._revealed[cell_id] = obs
        self._order.append(cell_id)
        self._cost_per_branch[n.branch] = self._cost_per_branch.get(n.branch, 0) + 1
        self.best_so_far = max(self.best_so_far, n.score)
        self.last_observation = obs
        return obs

    def legal_roots(self) -> list[str]:
        return sorted(
            (a for a in self.legal_actions() if self.meta(a).parent_id is None),
            key=lambda a: self.meta(a).branch,
        )

    def opened_branches(self) -> list[int]:
        return sorted(
            {
                obs.branch
                for nid, obs in self._revealed.items()
                if self.meta(nid).parent_id is None
            }
        )

    def probe_new(self) -> tuple[Observation, int, bool]:
        roots = self.legal_roots()
        if not roots:
            raise ValueError("no new branch available")
        obs = self.probe_batch([roots[0]])[0]
        branch = obs.branch
        return obs, branch, self._branch_done(branch)

    def probe_more(self, branch: int) -> tuple[Observation, bool]:
        if branch not in self.opened_branches():
            raise ValueError(f"branch {branch} not opened yet")
        cand = sorted(
            (a for a in self.legal_actions() if self.meta(a).branch == branch),
            key=lambda a: (self.meta(a).attempt, self.meta(a).seq),
        )
        if not cand:
            raise ValueError(f"branch {branch} has no legal probe")
        obs = self.probe_batch([cand[0]])[0]
        return obs, self._branch_done(branch)

    def get_cost_per_branch_active(self) -> list[int]:
        return [self._cost_per_branch.get(b, 0) for b in self.opened_branches()]

    def _branch_done(self, branch: int) -> bool:
        return not any(self.meta(a).branch == branch for a in self.legal_actions())
