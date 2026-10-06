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

"""CLI: online explore (init workspace + run agents)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from dream_rsi.cli._common import (
    DEFAULT_EVALUATOR,
    DEFAULT_INIT_PROGRAM,
    DEFAULT_PROBLEM,
    build_explore_config,
    install_sigint_handler,
    load_dotenv,
)


def _add_workspace_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-w", "--workspace", type=Path, required=True)


def cmd_init(args: argparse.Namespace) -> int:
    from dream_rsi.exploration import sync

    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    if args.problem:
        shutil.copy2(args.problem, ws / "problem.txt")
    sync(ws)
    print(json.dumps({"workspace": str(ws)}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from dream_rsi.agents import load_agent
    from dream_rsi.env.runner import run_live
    from dream_rsi.orchestrate import write_live_cycle_manifest
    from dream_rsi.policy.load import default_policy_source, load_policy

    ws = Path(args.workspace).resolve()
    agent = load_agent(args.agent, explore_shell=args.explore_shell)
    config = build_explore_config(args)
    policy_source = args.policy_source or default_policy_source(ws)
    caps = {
        key: value
        for key, value in (("max_width", args.branches), ("max_refine", args.refine))
        if value is not None
    }
    policy = load_policy(policy_source, config=caps)
    run = run_live(
        ws,
        agent,
        policy=policy,
        config=config,
        cycle=args.cycle,
        budget=args.max_steps or None,
        max_workers=args.max_workers,
        max_width=args.branches,
        max_refine=args.refine,
        direction_mode=args.direction_mode,
    )
    manifest_path = write_live_cycle_manifest(
        ws,
        cycle=args.cycle,
        policy_source=policy_source,
        policy=policy,
        run=run,
        max_workers=args.max_workers,
        max_width=args.branches,
        max_refine=args.refine,
        max_steps=args.max_steps,
    )
    print(
        json.dumps(
            {
                "nodes": run.node_ids,
                "failed": run.failed,
                "stopped": run.stopped_reason or "ok",
                "best_score": run.sim.best,
                "grid_plan": run.grid_plan,
                "direction_mode": run.direction_mode,
                "direction_summary_id": run.direction_summary_id,
                "direction_context_active": run.direction_context_active,
                "live_cycle_manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    from dream_rsi.orchestrate import _archive_and_reset_workspace

    ws = Path(args.workspace).resolve()
    dest = ws / "history" / f"iter{args.cycle:03d}"
    if dest.exists():
        print(f"ERROR: {dest} already exists", file=sys.stderr)
        return 2
    if not (ws / "raw_scores.json").is_file():
        print(f"ERROR: no live round to archive in {ws}", file=sys.stderr)
        return 2
    archived = _archive_and_reset_workspace(ws, cycle=args.cycle)
    print(json.dumps({"archived": archived}, indent=2))
    return 0


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog or "dream-rsi explore", description="Online exploration")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create workspace and copy problem.txt")
    _add_workspace_arg(s)
    s.add_argument("--problem", type=Path, default=DEFAULT_PROBLEM)
    s.set_defaults(func=cmd_init)

    s = sub.add_parser("run", help="run explore with agent + policy")
    _add_workspace_arg(s)
    s.add_argument("-a", "--agent", default="gemini")
    s.add_argument("-n", "--max-steps", type=int, default=0)
    s.add_argument("--problem", type=Path, default=None)
    s.add_argument("--evaluator", type=Path, default=None, help=f"default: {DEFAULT_EVALUATOR.name}")
    s.add_argument("--init-program", type=Path, default=None, help=f"default: {DEFAULT_INIT_PROGRAM.name}")
    s.add_argument("--eval-program", default=None)
    s.add_argument(
        "--explore-prompt",
        type=Path,
        default=None,
        help="agent prompt template (default: dream_rsi/prompts/explore.txt or DRSI_EXPLORE_PROMPT)",
    )
    s.add_argument("--eval-timeout", type=float, default=None)
    s.add_argument("--max-workers", type=int, default=None)
    s.add_argument(
        "--explore-shell",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="let the Gemini agent run shell commands while exploring "
        "(default: GEMINI_EXPLORE_SHELL, otherwise off)",
    )
    s.add_argument(
        "--direction-mode",
        choices=("off", "global"),
        default=None,
        help="global = inject the cumulative direction memory into every explore prompt "
        "(default: DRSI_DIRECTION_MODE, otherwise off)",
    )
    s.add_argument(
        "--branches",
        type=int,
        default=None,
        help="first-cycle/fallback grid width and learned-plan hard cap",
    )
    s.add_argument(
        "--refine",
        type=int,
        default=None,
        help="first-cycle/fallback refine count and learned-plan hard cap",
    )
    s.add_argument(
        "--policy-source",
        default=None,
        help="policy module path or 'baseline' "
        "(default: the workspace's promoted policy/method.py if present, else baseline)",
    )
    s.add_argument(
        "--cycle",
        type=int,
        default=0,
        help="cycle index of this round, recorded in live_cycle_manifest.json (default: 0 = E0)",
    )
    s.set_defaults(func=cmd_run)

    s = sub.add_parser(
        "archive",
        help="move the live round to history/iter<CYCLE>/ so the next run starts a new round",
    )
    _add_workspace_arg(s)
    s.add_argument("--cycle", type=int, required=True, help="cycle index of the live round")
    s.set_defaults(func=cmd_archive)

    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    install_sigint_handler()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        from dream_rsi.agents.base import shutdown_agents

        shutdown_agents()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
