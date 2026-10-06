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

"""DREAM-RSI recursive loop: explore → learning → promote(method) → explore.

The selected policy is promoted into the workspace (``<ws>/policy/method.py``)
and picked up by the next cycle's online :func:`dream_rsi.env.runner.run_live`
(which drives ``dream_rsi.env.TraceEnv`` in live mode). There is no sim2real
porting stage — online and offline run the same ``policy.solve(env)``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.history_prune import maybe_prune_archived_iter
from dream_rsi.exploration.prompt import prepare_baseline, update_baseline_champion
from dream_rsi.exploration.runner import sync
from dream_rsi.learning.allocate import (
    allocate_learning_workspace,
    copy_direction_summary_to_job,
    promote_method_to_workspace,
    refresh_learning_baseline_preset,
    workspace_learning_root,
)
from dream_rsi.learning.discover import DiscoverConfig, run_discover_sync
from dream_rsi.learning.directions.context import OFF_DIRECTION_MODE, resolve_direction_mode
from dream_rsi.learning.trace_pool import (
    append_workspace_trace,
    list_pool_traces,
    sync_workspace_pool_to_job,
    write_pool_manifest,
    workspace_trace_pool,
)

Stage = Literal["explore", "learning", "promote"]
STAGES: tuple[Stage, ...] = ("explore", "learning", "promote")


@dataclass
class PipelineConfig:
    """Recursive self-improvement loop (repeat ``cycles`` times).

    One full cycle::

        explore → learning(discover) → promote(method) → explore
    """

    workspace: Path
    agent: str = "claude"
    cycles: int = 1
    # 0-based: skip cycles [0, start_cycle) and begin at ``start_cycle``.
    # Example: after deleting the sixth cycle's artifacts, set start_cycle=5 to
    # re-run ``history/iter005`` / ``learning/iter005`` with the current policy.
    start_cycle: int = 0
    from_stage: Stage = "explore"

    discover_rounds: int = 3
    # If the initial learning proposals all fail replay validation, discovery
    # adds candidates one at a time within this bounded retry budget.
    discover_retry_rounds: int = 3
    discover_max_retry_batches: int = 3
    # Re-evaluate the current policy as one learning candidate, so the phase
    # compares M = discover_rounds + 1 policies and may keep the incumbent.
    evaluate_incumbent: bool = True
    target: str = "run_best"
    eval_timeout_sec: float = 1800.0

    skip_promote_method: bool = False

    resume: bool = True
    stream: bool = True

    # At the end of each cycle, archive this cycle's explore output into
    # ``workspace/history/iter<NNN>/`` and clear it, so the next cycle re-explores
    # from scratch with the freshly promoted method (instead of skipping the
    # already-"done" nodes left in the workspace).
    archive_each_cycle: bool = True

    # Whether the best evaluated program of each archived cycle becomes the
    # next cycle's generative baseline.  Off by default: every cycle starts
    # from the task's initial program, as in the paper.
    update_baseline_champion: bool = False

    explore_max_steps: int = 0
    explore_max_workers: int | None = None
    # First-cycle/fallback grid and the hard ceiling for an optional policy plan.
    explore_max_width: int | None = None
    explore_max_refine: int | None = None
    explore_config: ExploreConfig = field(default_factory=ExploreConfig.from_env)
    # Shell access of the Gemini agent during exploration; None defers to
    # GEMINI_EXPLORE_SHELL (off unless set).
    explore_shell: bool | None = None
    # ``global`` injects one cumulative direction memory into every explore
    # prompt (the +Guidance ablation); ``off`` is plain DREAM-RSI.
    direction_mode: str | None = None

    def __post_init__(self) -> None:
        self.direction_mode = resolve_direction_mode(self.direction_mode)


def _stage_index(stage: Stage) -> int:
    return STAGES.index(stage)


def _discover_ok(summary: dict[str, Any]) -> bool:
    """Whether discovery produced a replay-qualified, selected policy.

    The final proposer attempt may fail while an earlier archived attempt is
    qualified and has already won by reward.  Promotion must follow that
    selected candidate, not the final attempt's return code.
    """
    best = summary.get("best_round")
    if not isinstance(best, dict):
        return False
    source = best.get("method_src")
    method = summary.get("method_file")
    if not (
        source
        and method
        and Path(source).is_file()
        and Path(method).is_file()
        and best.get("eval_returncode") == 0
    ):
        return False
    try:
        # Do not copy a stale/overwritten job-local method into the workspace.
        return Path(source).read_bytes() == Path(method).read_bytes()
    except OSError:
        return False


def _learning_job_path(ws: Path, *, cycle: int) -> Path:
    return workspace_learning_root(ws) / f"iter{cycle:03d}"


def _grid_caps(cfg: PipelineConfig) -> dict[str, int] | None:
    """The online grid caps, forwarded to the policy and the learning baseline."""
    overrides: dict[str, int] = {}
    if cfg.explore_max_width is not None:
        overrides["max_width"] = cfg.explore_max_width
    if cfg.explore_max_refine is not None:
        overrides["max_refine"] = cfg.explore_max_refine
    return overrides or None


def _archive_and_reset_workspace(ws: Path, *, cycle: int) -> str | None:
    """Move this cycle's explore output into ``ws/history/iter<NNN>/`` and clear it.

    Relocates ``explore_b*`` branch dirs, ``history.json``, and ``raw_scores.json``.
    Stable seeds (``problem.txt``, ``.baseline/``) stay in place so the next cycle's
    :func:`run_live` can scaffold a fresh grid and re-run every node with the
    newly promoted method.
    """
    branch_dirs = sorted(p for p in ws.glob("explore_b*") if p.is_dir())
    raw_scores = ws / "raw_scores.json"
    live_manifest = ws / "live_cycle_manifest.json"
    if not branch_dirs and not raw_scores.is_file() and not live_manifest.is_file():
        return None

    dest = ws / "history" / f"iter{cycle:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    for bdir in branch_dirs:
        # History seeds are only for prompt context during explore.
        # Never archive them into iterNNN, otherwise next-cycle seeding nests
        # history recursively inside explore_b*/history/... and blows up size.
        seeded_history = bdir / "history"
        if seeded_history.is_symlink():
            seeded_history.unlink()
        elif seeded_history.is_dir():
            shutil.rmtree(seeded_history, ignore_errors=True)
        target = dest / bdir.name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(bdir), str(target))

    hist = ws / "history.json"
    if hist.is_file():
        shutil.move(str(hist), str(dest / "history.json"))

    if raw_scores.is_file():
        target = dest / "raw_scores.json"
        if target.exists():
            target.unlink()
        shutil.move(str(raw_scores), str(target))
    if live_manifest.is_file():
        shutil.move(str(live_manifest), str(dest / live_manifest.name))

    if (dest / "raw_scores.json").is_file():
        append_workspace_trace(ws, f"iter{cycle:03d}", dest / "raw_scores.json")
        write_pool_manifest(workspace_trace_pool(ws))

    maybe_prune_archived_iter(ws, dest, cycle=cycle)

    print(f"[cycle] archived explore output → {dest} (workspace reset)", flush=True)
    return str(dest)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_source_path(policy_source: str) -> Path | None:
    if policy_source == "baseline":
        return Path(__file__).resolve().parent / "policy" / "baseline.py"
    path = Path(policy_source).expanduser()
    return path.resolve() if path.is_file() else None


def write_live_cycle_manifest(
    workspace: Path,
    *,
    cycle: int,
    policy_source: str,
    policy: Any,
    run: Any,
    max_workers: int | None,
    max_width: int | None,
    max_refine: int | None,
    max_steps: int,
) -> Path:
    """Record the exact policy/default-beta facts for one completed live cycle.

    ``run`` is the :class:`dream_rsi.env.runner.ExploreRun` of that cycle.  The
    manifest is archived with the cycle and read back by later grid planning
    and learning, so every live explore (``evolve`` or ``explore run``) writes it.
    """
    source_path = _policy_source_path(policy_source)
    raw_beta = getattr(policy, "beta", getattr(policy, "config", {}).get("beta"))
    try:
        beta_default: float | None = float(raw_beta) if raw_beta is not None else None
    except (TypeError, ValueError):
        beta_default = None
    policy_config = dict(getattr(policy, "config", {}))
    sim = run.sim
    manifest = {
        "schema_version": 2,
        "cycle": cycle,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace.resolve()),
        "policy": {
            "name": getattr(policy, "NAME", type(policy).__name__),
            "source": str(source_path) if source_path is not None else policy_source,
            "sha256": _file_sha256(source_path) if source_path is not None else None,
            "beta_default": beta_default,
            "config": policy_config,
        },
        "explore": {
            "max_workers": max_workers,
            "max_width": max_width,
            "max_refine": max_refine,
            "max_steps": max_steps,
            "direction_mode": run.direction_mode,
            "direction_context_active": run.direction_context_active,
            "direction_summary_id": run.direction_summary_id,
            "best_score": sim.best,
            "probe_budget": sim.budget,
            "decision_rounds": sim.decision_rounds,
            "effective_sequential_rounds": sim.effective_sequential_rounds,
            "batch_sizes": list(sim.batch_sizes),
        },
        "grid_plan": run.grid_plan or {},
    }
    out = workspace / "live_cycle_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _run_explore_stage(
    cfg: PipelineConfig, ws: Path, *, label: str, cycle: int
) -> dict[str, Any]:
    print(f"[cycle] {label} explore → {ws}", flush=True)
    from dream_rsi.agents import load_agent

    from dream_rsi.env.runner import run_live
    from dream_rsi.policy.load import default_policy_source, load_policy

    agent = load_agent(cfg.agent, explore_shell=cfg.explore_shell)
    policy_source = default_policy_source(ws)
    policy = load_policy(policy_source, config=dict(_grid_caps(cfg) or {}))
    run = run_live(
        ws,
        agent,
        policy=policy,
        config=cfg.explore_config,
        cycle=cycle,
        budget=cfg.explore_max_steps or None,
        max_workers=cfg.explore_max_workers,
        max_width=cfg.explore_max_width,
        max_refine=cfg.explore_max_refine,
        direction_mode=cfg.direction_mode,
    )
    manifest_path = write_live_cycle_manifest(
        ws,
        cycle=cycle,
        policy_source=policy_source,
        policy=policy,
        run=run,
        max_workers=cfg.explore_max_workers,
        max_width=cfg.explore_max_width,
        max_refine=cfg.explore_max_refine,
        max_steps=cfg.explore_max_steps,
    )
    return {
        "workspace": str(ws),
        "nodes": run.node_ids,
        "failed": run.failed,
        "stopped": run.stopped_reason or "ok",
        "best_score": run.sim.best,
        "probe_budget": run.sim.budget,
        "beta_default": getattr(policy, "beta", None),
        "grid_plan": run.grid_plan,
        "direction_mode": run.direction_mode,
        "direction_summary_id": run.direction_summary_id,
        "direction_context_active": run.direction_context_active,
        "live_cycle_manifest": str(manifest_path),
    }


def _run_learning_stage(
    cfg: PipelineConfig, ws: Path, *, cycle: int
) -> tuple[Path, dict[str, Any], dict[str, Any] | None]:
    job_path = _learning_job_path(ws, cycle=cycle)
    sync(ws)
    direction_summary: dict[str, Any] | None = None
    if cfg.direction_mode != OFF_DIRECTION_MODE:
        print("[cycle] summarize semantic directions …", flush=True)
        try:
            from dream_rsi.agents import load_agent
            from dream_rsi.learning.directions.summarize import run_direction_summary

            result = run_direction_summary(
                ws,
                load_agent(cfg.agent),
                cycle=cycle,
                task=ws.name,
            )
            direction_summary = result.to_dict()
            if result.ok:
                print(
                    f"[cycle] direction catalog → {result.catalog_path} "
                    f"and prompt context → {result.context_path} "
                    f"({result.n_attempts} attempts, {result.n_extracted} new)",
                    flush=True,
                )
            else:
                print(
                    f"[cycle] direction summary failed; retaining fallback: {result.error}",
                    flush=True,
                )
        except Exception as exc:
            direction_summary = {
                "ok": False,
                "summary_id": f"iter{cycle:03d}",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[cycle] direction summary skipped: {exc}", flush=True)
    if job_path.is_dir() and cfg.resume:
        learning_root = job_path.resolve()
        print(f"[cycle] reuse learning job → {learning_root}", flush=True)
        from dream_rsi.learning.pipeline import export_workspace_trace

        export_workspace_trace(ws, out=learning_root / "trace" / "raw_scores.json")
        copy_direction_summary_to_job(ws, learning_root)
        print(f"[cycle] refreshed trace from {ws}", flush=True)
    else:
        print(f"[cycle] learning allocate from {ws} …", flush=True)
        learning_root = allocate_learning_workspace(ws, dest=job_path)
        print(f"[cycle] learning job → {learning_root}", flush=True)
    baseline_preset = refresh_learning_baseline_preset(
        learning_root,
        overrides=_grid_caps(cfg),
    )
    print(
        "[cycle] learning replay grid → "
        f"max_width={baseline_preset.get('max_width')} "
        f"max_refine={baseline_preset.get('max_refine')}",
        flush=True,
    )
    pool = sync_workspace_pool_to_job(ws, learning_root)
    n_pool = len(list_pool_traces(pool))
    print(f"[cycle] trace pool → {pool} ({n_pool} trace(s))", flush=True)
    print("[cycle] discover …", flush=True)
    discover_summary = run_discover_sync(
        DiscoverConfig(
            learning_root=learning_root,
            agent=cfg.agent,
            rounds=cfg.discover_rounds,
            retry_rounds=cfg.discover_retry_rounds,
            max_retry_batches=cfg.discover_max_retry_batches,
            evaluate_incumbent=cfg.evaluate_incumbent,
            target=cfg.target,
            resume=cfg.resume,
            eval_timeout_sec=cfg.eval_timeout_sec,
            stream=cfg.stream,
            max_workers=cfg.explore_max_workers,
        )
    )
    return learning_root, discover_summary, direction_summary


def _load_learning_result(ws: Path, *, cycle: int) -> tuple[Path, dict[str, Any]]:
    """The finished learning job of ``cycle``, for a cycle that starts at promote."""
    job_path = _learning_job_path(ws, cycle=cycle)
    summary_path = job_path / "discover_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"{summary_path} not found — run the learning stage of cycle {cycle} first"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"[cycle] reuse finished learning job → {job_path}", flush=True)
    return job_path.resolve(), summary


def _run_promote_stage(learning_root: Path, ws: Path) -> str:
    dst = promote_method_to_workspace(learning_root, ws)
    print(f"[cycle] promote method → {dst}", flush=True)
    return str(dst)


def run_cycle(cfg: PipelineConfig, *, cycle: int, start_stage: Stage) -> dict[str, Any]:
    ws = cfg.workspace.resolve()
    start_i = _stage_index(start_stage)
    row: dict[str, Any] = {
        "cycle": cycle,
        "workspace": str(ws),
        "explore": None,
        "learning_root": None,
        "discover": None,
        "direction_summary": None,
        "promoted_method": None,
        "archived_to": None,
        "baseline_champion_updated": False,
    }

    if start_i <= _stage_index("explore"):
        row["explore"] = _run_explore_stage(cfg, ws, label="[1/3]", cycle=cycle)

    if start_i <= _stage_index("learning"):
        learning_root, discover, direction_summary = _run_learning_stage(
            cfg, ws, cycle=cycle
        )
        row["direction_summary"] = direction_summary
    else:
        learning_root, discover = _load_learning_result(ws, cycle=cycle)
    row["learning_root"] = str(learning_root)
    row["discover"] = discover

    if not cfg.skip_promote_method:
        if not _discover_ok(discover):
            print("[cycle] skip promote method — discover did not succeed", flush=True)
        elif discover["best_round"].get("incumbent"):
            print("[cycle] incumbent kept — the current policy stays in force", flush=True)
        else:
            row["promoted_method"] = _run_promote_stage(learning_root, ws)
    else:
        print("[cycle] skip promote method", flush=True)

    # Reset the board so the next cycle re-explores fresh with the new method.
    # This also archives a live round explored outside this run (e.g. a shared
    # E0 copied into the workspace before ``--from learning``).
    if cfg.archive_each_cycle:
        archived = _archive_and_reset_workspace(ws, cycle=cycle)
        row["archived_to"] = archived
        if cfg.update_baseline_champion:
            # Roll the generative baseline forward to this cycle's champion and
            # log it in .baseline/baseline_meta.json (init stays immutable).
            search_dir = Path(archived) if archived else ws
            try:
                champ = update_baseline_champion(
                    ws, search_dir, cycle=cycle, config=cfg.explore_config
                )
                if champ is not None:
                    row["baseline_champion_updated"] = True
                    print(
                        f"[cycle] baseline champion (cycle {cycle}): "
                        f"{champ['node_id']} score={champ['score']:.6f}",
                        flush=True,
                    )
            except Exception as exc:  # never let baseline bookkeeping abort a cycle
                print(f"[cycle] baseline update skipped: {exc}", flush=True)
        else:
            print("[cycle] baseline champion update disabled (fixed initial baseline)", flush=True)
    return row


def _bootstrap_workspace(ws: Path, explore_config: ExploreConfig) -> None:
    """Create workspace if missing and seed problem.txt + .baseline/."""
    created = not ws.is_dir()
    ws.mkdir(parents=True, exist_ok=True)
    if explore_config.problem is not None and explore_config.problem.is_file():
        shutil.copy2(explore_config.problem, ws / "problem.txt")
    if not (ws / "problem.txt").is_file():
        raise FileNotFoundError(
            f"Missing {ws / 'problem.txt'} — pass --problem or set DRSI_PROBLEM"
        )
    prepare_baseline(ws, explore_config)
    if created:
        print(f"[pipeline] created workspace → {ws}", flush=True)


def run_pipeline(cfg: PipelineConfig) -> dict[str, Any]:
    if cfg.cycles < 1:
        raise ValueError("cycles must be >= 1")
    if cfg.start_cycle < 0:
        raise ValueError("start_cycle must be >= 0")
    if cfg.start_cycle >= cfg.cycles:
        raise ValueError(
            f"start_cycle={cfg.start_cycle} must be < cycles={cfg.cycles}"
        )

    ws = cfg.workspace.resolve()
    _bootstrap_workspace(ws, cfg.explore_config)

    summary: dict[str, Any] = {
        "workspace": str(ws),
        "cycles": cfg.cycles,
        "start_cycle": cfg.start_cycle,
        "from_stage": cfg.from_stage,
        "update_baseline_champion": cfg.update_baseline_champion,
        "rounds": [],
    }

    for i in range(cfg.start_cycle, cfg.cycles):
        # Each cycle re-explores from a clean board with the latest promoted
        # method. Only the first *executed* cycle may honor a mid-pipeline --from.
        start: Stage = cfg.from_stage if i == cfg.start_cycle else "explore"
        print(
            f"\n{'=' * 60}\n"
            f"[cycle] {i + 1}/{cfg.cycles} (index={i})  start={start}\n"
            f"{'=' * 60}",
            flush=True,
        )
        row = run_cycle(cfg, cycle=i, start_stage=start)
        summary["rounds"].append(row)

    out = ws / "pipeline_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def format_pipeline_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"workspace : {summary.get('workspace')}",
        f"cycles    : {summary.get('cycles')}",
        f"start     : {summary.get('start_cycle', 0)}",
    ]
    for row in summary.get("rounds") or []:
        c = row.get("cycle", "?")
        lines.append(f"\n--- cycle {c} ---")
        ein = row.get("explore") or {}
        if ein:
            lines.append(
                f"  explore   : {len(ein.get('nodes') or [])} ok, "
                f"{len(ein.get('failed') or [])} failed (stopped={ein.get('stopped')})"
            )
            grid = (ein.get("grid_plan") or {}).get("effective") or {}
            if grid:
                lines.append(
                    f"  grid      : {grid.get('branch_count')} branches × "
                    f"{grid.get('refine_count')} refines "
                    f"({(ein.get('grid_plan') or {}).get('source')})"
                )
        lines.append(f"  learning  : {row.get('learning_root')}")
        if row.get("promoted_method"):
            lines.append(f"  method    : {row['promoted_method']}")
        d = row.get("discover") or {}
        if d:
            lines.append(
                f"  discover  : {d.get('rounds_run', '?')} round(s) run, "
                f"{d.get('rounds_planned', '?')} proposal(s) planned"
            )
            br = d.get("best_round") or {}
            if br:
                kept = " (incumbent kept)" if br.get("incumbent") else ""
                lines.append(f"  selected  : {br.get('round_id')}{kept}")
        ds = row.get("direction_summary") or {}
        if ds:
            lines.append(
                f"  directions: {ds.get('summary_id')} "
                f"({'ok' if ds.get('ok') else 'fallback'})"
            )
        if row.get("archived_to"):
            lines.append(f"  archived  : {row['archived_to']}")
    return "\n".join(lines) + "\n"
