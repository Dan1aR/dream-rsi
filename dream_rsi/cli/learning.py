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

"""Offline learning CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dream_rsi.cli._common import load_dotenv


def _baseline_overrides(args: argparse.Namespace) -> dict | None:
    cfg: dict = {}
    if args.max_width is not None:
        cfg["max_width"] = args.max_width
    if args.max_refine is not None:
        cfg["max_refine"] = args.max_refine
    return cfg or None


def cmd_replay(args: argparse.Namespace) -> int:
    from dream_rsi.learning.pipeline import (
        format_learning_result,
        learning_result_to_dict,
        run_learning,
    )

    if args.workspace is None and args.trace is None:
        print("ERROR: pass --workspace or --trace", file=sys.stderr)
        return 2
    try:
        result = run_learning(
            args.workspace,
            trace_source=args.trace,
            export=not args.no_export and args.trace is None,
            method_file=args.method,
            baseline_config=_baseline_overrides(args),
            budget=args.budget,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(learning_result_to_dict(result), indent=2))
    else:
        print(format_learning_result(result))

    if not result.parallel_refine.ok:
        return 1
    if result.discovered is not None and not result.discovered.ok:
        return 1
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from dream_rsi.learning.pipeline import export_workspace_trace

    if args.workspace is None:
        print("ERROR: pass --workspace", file=sys.stderr)
        return 2
    try:
        out = export_workspace_trace(args.workspace, out=args.out)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"workspace": str(args.workspace.resolve()), "output": str(out)}, indent=2))
    return 0


def cmd_allocate(args: argparse.Namespace) -> int:
    from dream_rsi.learning.allocate import allocate_learning_workspace

    if args.workspace is None and args.trace is None:
        print("ERROR: allocate requires --workspace or --trace", file=sys.stderr)
        return 2
    source = args.trace if args.trace is not None else args.workspace
    try:
        job_root = allocate_learning_workspace(source, dest=args.out)
    except (FileNotFoundError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"learning_root": str(job_root)}, indent=2))
    return 0


def _discover_job(args: argparse.Namespace) -> Path:
    """The ``--job`` to resume, else a new job allocated from a workspace or trace."""
    from dream_rsi.learning.allocate import allocate_learning_workspace
    from dream_rsi.learning.trace_pool import sync_workspace_pool_to_job

    if args.job is not None:
        return args.job
    source = args.trace if args.trace is not None else args.workspace
    job = allocate_learning_workspace(source, dest=args.out)
    if args.trace is None:
        # Replay on every archived round of the workspace, as evolve does.
        sync_workspace_pool_to_job(args.workspace, job)
    print(f"[discover] allocated learning job → {job}")
    return job


def cmd_discover(args: argparse.Namespace) -> int:
    from dream_rsi.learning.discover import (
        DiscoverConfig,
        format_discover_summary,
        preflight,
        run_discover_sync,
    )

    if args.job is None and args.workspace is None and args.trace is None:
        print("ERROR: discover requires --job, --workspace, or --trace", file=sys.stderr)
        return 2
    err = preflight(args.agent)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    try:
        cfg = DiscoverConfig(
            learning_root=_discover_job(args),
            agent=args.agent,
            rounds=args.rounds,
            retry_rounds=args.retry_rounds,
            max_retry_batches=args.max_retry_batches,
            evaluate_incumbent=args.eval_incumbent,
            target=args.target,
            resume=not args.no_resume,
            eval_timeout_sec=args.eval_timeout,
            stream=not args.no_stream,
            max_workers=args.max_workers,
        )
        summary = run_discover_sync(cfg)
    except (
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
        NotADirectoryError,
        OSError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_discover_summary(summary))

    # Success means a replay-qualified candidate was selected, even when the
    # final proposal itself failed.
    return 0 if summary.get("best_round") else 1


def cmd_directions(args: argparse.Namespace) -> int:
    """Summarize cumulative exploration directions for the next live round."""
    from dream_rsi.agents import load_agent
    from dream_rsi.learning.directions.summarize import (
        infer_next_cycle,
        run_direction_summary,
    )

    ws = args.workspace.resolve()
    if not ws.is_dir():
        print(f"ERROR: workspace not found: {ws}", file=sys.stderr)
        return 2
    cycle = args.cycle if args.cycle is not None else infer_next_cycle(ws)
    result = run_direction_summary(
        ws,
        load_agent(args.agent),
        cycle=cycle,
        task=args.task or ws.name,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


def _shared_learning_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workspace", "-w", type=Path, default=None)
    p.add_argument("--trace", "-t", type=Path, default=None)
    p.add_argument("--json", action="store_true")


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    from dream_rsi.agents import AGENTS

    p = argparse.ArgumentParser(
        prog=prog or "dream-rsi learning",
        description="Offline learning: export trace, replay eval, discover policy, or summarize directions",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("replay", help="replay the baseline and a policy on a frozen trace")
    _shared_learning_args(s)
    s.add_argument("--no-export", action="store_true")
    s.add_argument(
        "--method",
        "-m",
        type=Path,
        default=None,
        help="policy file (default: the workspace's promoted policy/method.py)",
    )
    s.add_argument("--max-width", type=int, default=None)
    s.add_argument("--max-refine", type=int, default=None)
    s.add_argument("--budget", type=int, default=None)
    s.set_defaults(func=cmd_replay)

    s = sub.add_parser("export", help="workspace → raw_scores.json")
    s.add_argument("--workspace", "-w", type=Path, required=True)
    s.add_argument(
        "--out", type=Path, default=None, help="default: <workspace>/export/raw_scores.json"
    )
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_export)

    out_help = (
        "learning job directory (default: <workspace>/learning/<UTC timestamp>; "
        "required with --trace)"
    )
    s = sub.add_parser("allocate", help="copy trace + sandbox into a learning job dir")
    _shared_learning_args(s)
    s.add_argument("--out", type=Path, default=None, help=out_help)
    s.set_defaults(func=cmd_allocate)

    s = sub.add_parser(
        "discover",
        help="agent rewrites OptimalPolicy; candidates are replayed and the best is kept",
    )
    _shared_learning_args(s)
    s.add_argument("--job", "-j", type=Path, default=None, help="existing learning job to run or resume")
    s.add_argument("--out", type=Path, default=None, help=out_help)
    s.add_argument("--agent", "-a", default="claude", choices=AGENTS)
    s.add_argument("--rounds", type=int, default=3, help="agent rewrites per learning phase")
    s.add_argument(
        "--retry-rounds",
        type=int,
        default=3,
        help="extra proposals tried after an all-invalid initial group",
    )
    s.add_argument(
        "--max-retry-batches",
        type=int,
        default=3,
        help="safety limit on invalid-candidate retry batches",
    )
    s.add_argument(
        "--eval-incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replay the job's seeded policy (or the parallel-refine baseline) "
        "as candidate r0000 (default: enabled)",
    )
    s.add_argument("--target", default="run_best")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--eval-timeout", type=float, default=1800.0)
    s.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="replay parallelism assumption; evolve forwards explore --max-workers",
    )
    s.add_argument("--no-stream", action="store_true")
    s.set_defaults(func=cmd_discover)

    s = sub.add_parser(
        "directions",
        help="summarize all attempt directions into workspace/directions/current.json",
    )
    s.add_argument("--workspace", "-w", type=Path, required=True)
    s.add_argument("--agent", "-a", default="claude", choices=AGENTS)
    s.add_argument("--cycle", type=int, default=None)
    s.add_argument("--task", default=None)
    s.set_defaults(func=cmd_directions)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
