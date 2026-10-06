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

"""CLI: the DREAM-RSI loop (explore → discover → promote)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dream_rsi.agents import AGENTS
from dream_rsi.cli._common import build_explore_config, load_dotenv
from dream_rsi.orchestrate import STAGES, PipelineConfig, format_pipeline_summary, run_pipeline


def cmd_evolve(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(
        workspace=args.workspace,
        agent=args.agent,
        cycles=args.cycles,
        start_cycle=args.start_cycle,
        from_stage=args.from_stage,
        discover_rounds=args.discover_rounds,
        discover_retry_rounds=args.discover_retry_rounds,
        discover_max_retry_batches=args.discover_max_retry_batches,
        evaluate_incumbent=args.eval_incumbent,
        target=args.target,
        eval_timeout_sec=args.eval_timeout,
        skip_promote_method=args.skip_promote_method,
        archive_each_cycle=not args.no_archive,
        update_baseline_champion=args.update_baseline_champion,
        resume=not args.no_resume,
        stream=not args.no_stream,
        explore_max_steps=args.max_steps,
        explore_max_workers=args.max_workers,
        explore_max_width=args.branches,
        explore_max_refine=args.refine,
        explore_config=build_explore_config(args),
        explore_shell=args.explore_shell,
        direction_mode=args.direction_mode,
    )

    try:
        summary = run_pipeline(cfg)
    except (RuntimeError, FileNotFoundError, FileExistsError, NotADirectoryError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(format_pipeline_summary(summary))

    last = (summary.get("rounds") or [{}])[-1]
    d = last.get("discover") or {}
    # A failed final proposal must not turn a successfully promoted earlier
    # candidate into a shell-level failure.
    return 0 if d.get("best_round") else 1


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "dream-rsi evolve",
        description="DREAM-RSI loop: explore → learning(discover) → promote(method)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--workspace", "-w", type=Path, required=True)
    p.add_argument("--agent", "-a", default="claude", choices=AGENTS)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument(
        "--start-cycle",
        type=int,
        default=0,
        help="0-based cycle index to start from (skip earlier cycles); "
        "e.g. --start-cycle 5 re-runs history/iter005 and learning/iter005",
    )
    p.add_argument("--from", dest="from_stage", default="explore", choices=STAGES)
    p.add_argument("--discover-rounds", type=int, default=3)
    p.add_argument(
        "--discover-retry-rounds",
        type=int,
        default=3,
        help="extra proposals tried after an all-invalid initial discovery group",
    )
    p.add_argument(
        "--discover-max-retry-batches",
        type=int,
        default=3,
        help="safety limit on invalid-candidate retry batches",
    )
    p.add_argument(
        "--eval-incumbent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="replay the current policy as one learning candidate, so the "
        "rewrites must beat it to be promoted (default: enabled)",
    )
    p.add_argument("--target", default="run_best")
    p.add_argument("--eval-timeout", type=float, default=1800.0, help="discover eval timeout (s)")
    p.add_argument(
        "--explore-eval-timeout",
        type=float,
        default=None,
        help="per-node explore eval timeout (s) (default: DRSI_EVAL_TIMEOUT_SEC, else 600)",
    )
    p.add_argument("--skip-promote-method", action="store_true")
    p.add_argument(
        "--update-baseline-champion",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="roll each cycle's best program into .baseline as the next cycle's "
        "initial program (default: disabled; every cycle starts from the task's "
        "initial program, as in the paper)",
    )
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument(
        "--branches",
        type=int,
        default=None,
        help="first-cycle/fallback grid width and learned-plan hard cap",
    )
    p.add_argument(
        "--refine",
        type=int,
        default=None,
        help="first-cycle/fallback refine count and learned-plan hard cap",
    )
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument(
        "--explore-shell",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="let the Gemini agent run shell commands while exploring "
        "(default: GEMINI_EXPLORE_SHELL, otherwise off)",
    )
    p.add_argument("--no-archive", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-stream", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--direction-mode",
        choices=("off", "global"),
        default=None,
        help="global = summarize directions after each explore and inject them into "
        "the next explore prompt (+Guidance ablation; default: DRSI_DIRECTION_MODE, otherwise off)",
    )
    p.add_argument("--problem", type=Path, default=None)
    p.add_argument("--evaluator", type=Path, default=None)
    p.add_argument("--init-program", type=Path, default=None)
    p.add_argument(
        "--explore-prompt",
        type=Path,
        default=None,
        help="agent prompt template (default: dream_rsi/prompts/explore.txt or DRSI_EXPLORE_PROMPT)",
    )
    p.set_defaults(func=cmd_evolve)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
