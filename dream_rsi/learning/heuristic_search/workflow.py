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

"""Multi-round policy discovery: propose -> replay-evaluate -> archive -> select.

One learning job runs an optional incumbent round plus ``rounds`` agent
proposals.  Round ``r0000`` re-evaluates the incumbent policy (the job's
seeded ``method.py``, or the parallel-refine baseline before any promotion)
without calling the agent.  Every proposal round resets the editable
``discovered/method.py`` (the first one keeps a seeded incumbent as its
starting point), runs an ``agents.*`` backend to rewrite it, evaluates it with
``eval.py``, and archives it to ``<history_dir>/r####_<ts>_<uid>/``.  The best
replay-qualified candidate is finally copied back over ``method.py``.

Resume: with ``resume=True``, scan ``<history_dir>`` for
``rNNNN_YYYYMMDD_HHMMSS_<8hex>``, take the largest N of the latest session,
and continue from there.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dream_rsi.learning.heuristic_search.discovered_sandbox import (
    editable_paths,
    reset_from_templates,
)

# First line of the generated baseline incumbent.  Such a file is never a seed:
# proposals start from the template, exactly as before the first promotion.
BASELINE_INCUMBENT_HEADER = "# DREAM-RSI incumbent: parallel-refine baseline"

_BASELINE_INCUMBENT_SOURCE = '''{header}
"""Incumbent policy before any promotion: the parallel-refine baseline.

Replays the hand-designed wave-parallel explore + refine policy with the
online grid caps, so the first offline phase compares the proposals against
the policy that generated the trace.
"""

from dream_rsi.policy.baseline import ParallelRefineMethod

GRID_CAPS = {caps!r}


class OptimalPolicy(ParallelRefineMethod):
    NAME = "parallel_refine"

    def __init__(self, config=None):
        super().__init__({{**GRID_CAPS, **(config or {{}})}})
'''


def baseline_incumbent_source(grid_caps: dict[str, Any] | None = None) -> str:
    """Source of an ``OptimalPolicy`` that replays the parallel-refine baseline."""
    return _BASELINE_INCUMBENT_SOURCE.format(
        header=BASELINE_INCUMBENT_HEADER, caps=dict(grid_caps or {})
    )


def archive_round(
    *,
    workdir: Path,
    history_dir: str,
    round_id: str,
    method_file: str,
    method_src: Path,
    result_dir: str | Path | None,
    dest_allow_exists: bool = False,
    extra_files: tuple[str, ...] = (),
) -> Path:
    """Archive the current round under ``history/<round_id>/``."""
    dest = workdir / history_dir / round_id
    dest.mkdir(parents=True, exist_ok=dest_allow_exists)

    for rel in (method_file, *extra_files):
        src = (workdir / rel).resolve()
        if not src.is_file():
            continue
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

    if result_dir is None:
        return dest

    src = Path(result_dir).expanduser()
    if not src.is_absolute():
        src = (workdir / src).resolve()
    else:
        src = src.resolve()
    if not src.is_dir():
        (dest / "proposal_result_dir.txt").write_text(str(src), encoding="utf-8")
        return dest

    dest_pr = dest / "proposal_results"
    dest_pr.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        dst = dest_pr / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied_files.append(str(rel))
    if copied_files:
        (dest / "proposal_results_manifest.json").write_text(
            json.dumps(
                {"source_dir": str(src), "copied_files": copied_files},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    for child in src.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass

    return dest


_ROUND_ID_RE = re.compile(r"^r(\d{4})_(\d{8}_\d{6})_([0-9a-f]{8})$")


def parse_round_id(round_id: str) -> Optional[tuple[int, str, str]]:
    m = _ROUND_ID_RE.match(round_id.strip())
    if not m:
        return None
    return int(m.group(1), 10), m.group(2), m.group(3)


def scan_history_resume(
    workdir: Path, history_dir: str
) -> tuple[Optional[str], Optional[str], int]:
    base = workdir / history_dir
    if not base.is_dir():
        return None, None, 0
    candidates: list[tuple[Path, int, str, str, float]] = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_round_id(p.name)
        if parsed is None:
            continue
        idx, ts, uid = parsed
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        candidates.append((p, idx, ts, uid, mt))
    if not candidates:
        return None, None, 0
    _, _, run_ts, run_uid, _ = max(candidates, key=lambda c: c[4])
    mx = max(c[1] for c in candidates if c[2] == run_ts and c[3] == run_uid)
    return run_ts, run_uid, mx + 1


def append_workflow_index(
    *,
    workdir: Path,
    history_dir: str,
    row: dict[str, Any],
) -> None:
    index = workdir / history_dir / "workflow_index.jsonl"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval_subprocess(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_sec: float,
    stream: bool = False,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    full_env = {**os.environ, **(env or {})}
    try:
        if stream:
            print(f"[eval] $ {' '.join(cmd)}", flush=True)
            cp = subprocess.run(
                cmd,
                cwd=str(cwd.resolve()),
                timeout=timeout_sec,
                env=full_env,
            )
            return cp.returncode, "", ""
        cp = subprocess.run(
            cmd,
            cwd=str(cwd.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=full_env,
        )
        return cp.returncode, cp.stdout or "", cp.stderr or ""
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else str(e)
        return -1, out, err + "\n[workflow] eval timeout\n"
    except OSError as e:
        return -1, "", str(e)


@dataclass
class WorkflowConfig:
    workdir: Path
    method_file: str
    history_dir: str
    proposer_prompt_path: Path
    rounds: int
    log_parent: Path
    proposer_backend: str = "codex"
    resume: bool = False
    result_dir: Optional[Path] = None
    eval_cmd: tuple[str, ...] = ()
    eval_cwd: Optional[Path] = None
    eval_env: dict[str, str] | None = None
    eval_timeout_sec: float = 7200.0
    stream: bool = False
    prompt_vars: dict[str, str] | None = None
    # Run ``rounds`` initial proposals to compare their reward.  If none is
    # replay-valid, keep trying one proposal at a time, up to this bounded
    # retry budget.  The bound turns a systematic evaluator/prompt failure
    # into an explicit failure instead of an unbounded agent loop.
    retry_rounds: int = 3
    max_retry_batches: int = 3
    # Re-evaluate the incumbent as round r0000 so it competes for selection
    # (M = rounds + 1 candidates).  Without a seeded ``method.py`` the
    # incumbent is ``incumbent_fallback`` (see :func:`baseline_incumbent_source`).
    evaluate_incumbent: bool = False
    incumbent_fallback: str | None = None

    @property
    def incumbent_rounds(self) -> int:
        return 1 if self.evaluate_incumbent else 0

    @property
    def initial_rounds(self) -> int:
        """Rounds always run before early stopping is allowed."""
        return self.incumbent_rounds + self.rounds

    @property
    def max_rounds(self) -> int:
        return self.initial_rounds + self.retry_rounds * self.max_retry_batches


def _resolve_method_template(workdir: Path, method_file: str) -> Optional[Path]:
    template_path = (workdir / method_file).resolve().parent / "method.template.py"
    return template_path if template_path.is_file() else None


def _seeded_method_text(method_path: Path, template_path: Optional[Path]) -> Optional[str]:
    """Return the seeded ``method.py`` (a previously promoted policy), if any.

    A missing file, the empty template, and the generated baseline incumbent
    are not seeds.
    """
    if not method_path.is_file():
        return None
    text = method_path.read_text(encoding="utf-8")
    if text.startswith(BASELINE_INCUMBENT_HEADER):
        return None
    if template_path is not None and text == template_path.read_text(encoding="utf-8"):
        return None
    return text


def _render_proposer_prompt(
    proposer_prompt_path: Path,
    *,
    method_file: str,
    history_dir: str,
    prompt_vars: dict[str, str] | None = None,
) -> str:
    text = proposer_prompt_path.read_text(encoding="utf-8")
    replacements = {
        "method_file": method_file,
        "history_dir": history_dir,
        **(prompt_vars or {}),
    }
    for key, value in replacements.items():
        text = text.replace("{" + key + "}", value)
    return text


def _run_proposer_for_round(
    *,
    backend: str,
    workdir: Path,
    method_file: str,
    history_dir: str,
    proposer_prompt_path: Path,
    round_output_dir: Path,
    prompt_vars: dict[str, str] | None = None,
) -> dict[str, Any]:
    from dream_rsi.agents import load_agent

    prompt = _render_proposer_prompt(
        proposer_prompt_path,
        method_file=method_file,
        history_dir=history_dir,
        prompt_vars=prompt_vars,
    )
    print(f"[discover] agent={backend}  log_dir={round_output_dir}", flush=True)
    print(f"[discover] prompt saved → {round_output_dir / 'prompt.txt'}", flush=True)
    if os.environ.get("DRSI_AGENT_STREAM", "").strip().lower() in ("1", "true", "yes", "on"):
        print(f"[discover] streaming ON — tail -f {round_output_dir / 'events.jsonl'}  (gemini)", flush=True)
    agent = load_agent(backend)
    result = agent.run(
        prompt=prompt,
        workspace=workdir,
        node_dir=round_output_dir,
        log_dir=round_output_dir,
        guide={},
        run_cwd=workdir,
        stage="learning",
    )
    print(f"[discover] agent done: status={result.status}  {result.message}", flush=True)
    return {"backend": backend, "status": result.status, "message": result.message}


def _restore_method_from_history(
    workdir: Path, history_dir: str, method_file: str, round_index: int
) -> bool:
    """On resume, restore ``method.py`` from ``history/r{index-1}/``."""
    if round_index <= 0:
        return False
    hist = workdir / history_dir
    if not hist.is_dir():
        return False
    prefix = f"r{round_index - 1:04d}_"
    matches = sorted(p for p in hist.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        return False
    src = matches[-1] / method_file
    if not src.is_file():
        return False
    dst = (workdir / method_file).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _round_eval_returncodes(workdir: Path, history_dir: str) -> dict[str, int | None]:
    """Return the recorded evaluator return code for each archived round."""
    index = workdir / history_dir / "workflow_index.jsonl"
    if not index.is_file():
        return {}
    out: dict[str, int | None] = {}
    for line in index.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        round_id = row.get("round_id")
        if isinstance(round_id, str):
            rc = row.get("eval_returncode")
            out[round_id] = rc if isinstance(rc, int) else None
    return out


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _qualification_reason(
    round_dir: Path, *, eval_returncode: int | None
) -> tuple[bool, str, dict[str, Any] | None]:
    """Validate an archived candidate before it may compete for promotion.

    A policy is qualified only when its main replay evaluation succeeded and its
    beta sweep produced actual, error-free replay episodes with a finite current
    objective.  Merely finding a stale/partial ``beta_sweep.json`` is not enough.
    """
    if eval_returncode != 0:
        return False, f"eval_returncode={eval_returncode!r}", None

    results = round_dir / "proposal_results"
    eval_path = results / "heuristic_eval.json"
    sweep_path = results / "beta_sweep.json"
    if not eval_path.is_file() or not sweep_path.is_file():
        return False, "missing archived evaluation artifacts", None
    try:
        evaluation = json.loads(eval_path.read_text(encoding="utf-8", errors="replace"))
        sweep = json.loads(sweep_path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"unreadable evaluation artifact: {exc}", None

    if evaluation.get("status") != "ok":
        return False, f"replay status={evaluation.get('status')!r}", None
    traces = evaluation.get("traces")
    if isinstance(traces, list) and any(row.get("status") != "ok" for row in traces):
        return False, "one or more replay traces failed", None

    grid = sweep.get("grid")
    if not isinstance(grid, list) or not grid:
        return False, "beta sweep has no grid points", None
    if any(not isinstance(row, dict) or row.get("status") != "ok" for row in grid):
        return False, "one or more beta replay evaluations failed", None
    if not isinstance(sweep.get("n_execution_episodes"), int) or sweep["n_execution_episodes"] <= 0:
        return False, "beta sweep has no valid execution episodes", None

    pareto = sweep.get("pareto")
    if not isinstance(pareto, dict):
        return False, "missing pareto summary", None
    reward, auc = pareto.get("reward"), pareto.get("auc")
    if not _is_finite_number(reward) or not _is_finite_number(auc):
        return False, "non-finite pareto reward or auc", None
    if not isinstance(pareto.get("n_traces_scored"), int) or pareto["n_traces_scored"] <= 0:
        return False, "pareto summary has no scored traces", None
    return True, "ok", sweep


def _select_best_round(
    workdir: Path, history_dir: str, method_file: str
) -> Optional[dict[str, Any]]:
    """Return the highest-reward archived candidate that passed replay validation."""
    base = workdir / history_dir
    if not base.is_dir():
        return None
    returncodes = _round_eval_returncodes(workdir, history_dir)
    best: Optional[dict[str, Any]] = None
    for p in sorted(base.iterdir()):
        parsed = parse_round_id(p.name)
        if not p.is_dir() or parsed is None:
            continue
        method_src = p / method_file
        if not method_src.is_file():
            continue
        qualified, _reason, sweep = _qualification_reason(
            p, eval_returncode=returncodes.get(p.name)
        )
        if not qualified or sweep is None:
            continue
        pareto = sweep["pareto"]
        cand = {
            "round_id": p.name,
            "round_index": parsed[0],
            "reward": float(pareto["reward"]),
            "pareto_auc": float(pareto["auc"]),
            "parallel_penalty": pareto.get("parallel_penalty"),
            "eval_returncode": returncodes[p.name],
            "n_execution_episodes": sweep["n_execution_episodes"],
            "n_traces_scored": pareto["n_traces_scored"],
            "method_src": str(method_src),
        }
        # Highest reward wins; on ties prefer the later round (more history).
        if best is None or (cand["reward"], cand["round_index"]) > (
            best["reward"],
            best["round_index"],
        ):
            best = cand
    return best


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def promote_best_round(cfg: "WorkflowConfig") -> Optional[dict[str, Any]]:
    """Copy the best qualified, highest-reward method over live ``method_file``."""
    best = _select_best_round(cfg.workdir, cfg.history_dir, cfg.method_file)
    if best is None:
        (cfg.workdir / cfg.history_dir / "best_round.json").unlink(missing_ok=True)
        print("[heuristic_search] no replay-qualified candidate found; keeping last method")
        return None
    dst = (cfg.workdir / cfg.method_file).resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["method_src"], dst)
    source_digest = _sha256(best["method_src"])
    if _sha256(dst) != source_digest:
        raise RuntimeError(
            "promoted method verification failed: destination differs from selected round"
        )
    best["method_sha256"] = source_digest
    best["incumbent"] = cfg.evaluate_incumbent and best["round_index"] == 0
    (cfg.workdir / cfg.history_dir / "best_round.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[heuristic_search] promoted best round {best['round_id']} "
        f"(reward={best['reward']:.4g}, auc={best['pareto_auc']:.4g}, "
        f"parallel_penalty={best['parallel_penalty']}) -> {cfg.method_file}"
    )
    return best


async def run_workflow(cfg: WorkflowConfig) -> list[dict[str, Any]]:
    if cfg.rounds <= 0:
        raise ValueError("rounds must be positive")
    if cfg.retry_rounds < 0 or cfg.max_retry_batches < 0:
        raise ValueError("retry_rounds and max_retry_batches must be non-negative")

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / cfg.history_dir).mkdir(parents=True, exist_ok=True)
    cfg.log_parent.mkdir(parents=True, exist_ok=True)

    eval_cwd = (cfg.eval_cwd or cfg.workdir).resolve()
    method_path = (cfg.workdir / cfg.method_file).resolve()
    method_template_path = _resolve_method_template(cfg.workdir, cfg.method_file)

    start_i = 0
    resume_effective = False
    if cfg.resume:
        rts, ruid, nxt = scan_history_resume(cfg.workdir, cfg.history_dir)
        if rts and ruid and nxt > 0:
            run_ts, run_uid, start_i = rts, ruid, nxt
            resume_effective = True
            print(
                f"[heuristic_search] resume run={run_ts}_{run_uid}, next round r{start_i:04d}"
            )
            _restore_method_from_history(cfg.workdir, cfg.history_dir, cfg.method_file, start_i)
        else:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            run_uid = uuid.uuid4().hex[:8]
            start_i = 0
            print("[heuristic_search] resume requested but no archived rounds; fresh run")
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_uid = uuid.uuid4().hex[:8]

    # On a fresh run this is the promoted policy the job was seeded with; on a
    # resume right after r0000 it is restored from the incumbent's archive.
    seed_text = _seeded_method_text(method_path, method_template_path)
    if (
        cfg.evaluate_incumbent
        and start_i == 0
        and seed_text is None
        and cfg.incumbent_fallback is None
    ):
        raise ValueError("evaluate_incumbent needs a seeded method file or incumbent_fallback")

    results: list[dict[str, Any]] = []
    offset = cfg.incumbent_rounds
    max_rounds = cfg.max_rounds
    i = start_i
    # A resumed job may already have a qualified candidate, even when its final
    # proposal failed.  Restore/promote that candidate instead of treating the
    # last evaluator return code as the job's outcome.
    best_round = _select_best_round(cfg.workdir, cfg.history_dir, cfg.method_file)
    if i >= cfg.initial_rounds and best_round is not None:
        print("[heuristic_search] qualified archived candidate already exists")
        promote_best_round(cfg)
        return results
    if i >= max_rounds:
        (cfg.workdir / cfg.history_dir / "best_round.json").unlink(missing_ok=True)
        raise RuntimeError(
            "retry budget exhausted without a replay-qualified candidate"
        )

    while i < max_rounds:
        # The initial group is always completed so reward has several candidates
        # to compare.  Afterwards, stop as soon as a qualified retry is found.
        initial_group = i < cfg.initial_rounds
        incumbent = cfg.evaluate_incumbent and i == 0
        kind = "incumbent" if incumbent else "proposal"
        round_id = f"r{i:04d}_{run_ts}_{run_uid}"
        round_log = cfg.log_parent / round_id
        round_log.mkdir(parents=True, exist_ok=resume_effective)

        method_path.parent.mkdir(parents=True, exist_ok=True)
        if incumbent:
            source = seed_text if seed_text is not None else cfg.incumbent_fallback
            method_path.write_text(source, encoding="utf-8")
            origin = "seeded method.py" if seed_text is not None else "parallel-refine baseline"
            print(f"[discover] round r{i:04d}: evaluating the incumbent ({origin})")
        elif i == offset and seed_text is not None:
            method_path.write_text(seed_text, encoding="utf-8")
            print("[discover] first proposal starts from the seeded method.py")
        else:
            reset = reset_from_templates(method_path.parent)
            if reset:
                print(f"[heuristic_search] reset discovered: {', '.join(reset)}")
            elif method_template_path is not None:
                shutil.copy2(method_template_path, method_path)

        print(
            f"\n{'=' * 60}\n[discover] round {i + 1}/{max_rounds} ({kind})  id={round_id}\n{'=' * 60}",
            flush=True,
        )

        if not incumbent:
            await asyncio.to_thread(
                _run_proposer_for_round,
                backend=cfg.proposer_backend,
                workdir=cfg.workdir,
                method_file=cfg.method_file,
                history_dir=cfg.history_dir,
                proposer_prompt_path=cfg.proposer_prompt_path,
                round_output_dir=round_log,
                prompt_vars=cfg.prompt_vars,
            )

        eval_rc: Optional[int] = None
        if cfg.eval_cmd:
            print("[discover] running eval …", flush=True)
            eval_rc, eval_out, eval_err = await asyncio.to_thread(
                run_eval_subprocess,
                cmd=list(cfg.eval_cmd),
                cwd=eval_cwd,
                timeout_sec=cfg.eval_timeout_sec,
                stream=cfg.stream,
                env=cfg.eval_env,
            )
            if not cfg.stream and eval_out.strip():
                print(eval_out, end="" if eval_out.endswith("\n") else "\n", flush=True)
            if not cfg.stream and eval_err.strip():
                print(eval_err, file=sys.stderr, end="" if eval_err.endswith("\n") else "\n", flush=True)
            print(f"[discover] eval returncode={eval_rc}", flush=True)

        extra = tuple(p for p in editable_paths(cfg.method_file) if p != cfg.method_file)
        hist_path = archive_round(
            workdir=cfg.workdir,
            history_dir=cfg.history_dir,
            round_id=round_id,
            method_file=cfg.method_file,
            method_src=method_path,
            result_dir=cfg.result_dir,
            dest_allow_exists=resume_effective,
            extra_files=extra,
        )

        proposal_results_hist = hist_path / "proposal_results"
        proposal_results_archive = (
            str(proposal_results_hist) if proposal_results_hist.is_dir() else ""
        )
        result_dir_str = str(cfg.result_dir) if cfg.result_dir is not None else ""
        index_row = {
            "round_id": round_id,
            "kind": kind,
            "result_dir": result_dir_str,
            "eval_returncode": eval_rc,
            "dest": str(hist_path),
            "proposal_results_archive": proposal_results_archive,
        }
        append_workflow_index(
            workdir=cfg.workdir, history_dir=cfg.history_dir, row=index_row
        )

        row = {
            "round_index": i,
            "round_id": round_id,
            "kind": kind,
            "agent_log_dir": str(round_log),
            "history_archive": str(hist_path),
            "result_dir": result_dir_str,
            "eval_returncode": eval_rc,
            "eval_skipped": not bool(cfg.eval_cmd),
            "proposal_results_archive": proposal_results_archive,
        }
        results.append(row)
        i += 1

        # Do not end early in the initial comparison group.  Once it is over,
        # any qualified candidate is enough: select the highest reward among all
        # qualified archived rounds and stop.
        if not initial_group or i == cfg.initial_rounds:
            best_round = _select_best_round(cfg.workdir, cfg.history_dir, cfg.method_file)
            if best_round is not None:
                break

    best_round = promote_best_round(cfg)
    if best_round is None:
        raise RuntimeError(
            "retry budget exhausted without a replay-qualified candidate"
        )

    summary_path = cfg.log_parent / f"workflow_summary_{run_ts}_{run_uid}.json"
    prev_rounds: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            prev_obj = json.loads(
                summary_path.read_text(encoding="utf-8", errors="replace")
            )
            prev_rounds = list(prev_obj.get("rounds") or [])
        except (json.JSONDecodeError, OSError):
            prev_rounds = []
    summary_obj = {
        "workdir": str(cfg.workdir),
        "method_file": cfg.method_file,
        "history_dir": cfg.history_dir,
        "proposer_backend": cfg.proposer_backend,
        "rounds_planned": cfg.rounds,
        "evaluate_incumbent": cfg.evaluate_incumbent,
        "retry_rounds": cfg.retry_rounds,
        "max_retry_batches": cfg.max_retry_batches,
        "max_rounds": max_rounds,
        "eval_cmd": list(cfg.eval_cmd) if cfg.eval_cmd else [],
        "result_dir": str(cfg.result_dir) if cfg.result_dir is not None else "",
        "run_ts": run_ts,
        "run_uid": run_uid,
        "resumed": resume_effective,
        "resume_from_index": start_i if resume_effective else 0,
        "best_round": best_round,
        "rounds": prev_rounds + results,
    }
    summary_path.write_text(
        json.dumps(summary_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results
