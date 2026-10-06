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

"""Shared CLI helpers."""

from __future__ import annotations

import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = ROOT / "tasks/lasso_path"
DEFAULT_PROBLEM = DEFAULT_TASK / "lasso_path.txt"
DEFAULT_EVALUATOR = DEFAULT_TASK / "evaluator.py"
DEFAULT_INIT_PROGRAM = DEFAULT_TASK / "init_program.py"


def ensure_root_on_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def load_dotenv() -> None:
    ensure_root_on_path()
    from dream_rsi.dotenv import load_env

    load_env()


def install_sigint_handler() -> None:
    from dream_rsi.agents.base import shutdown_agents

    def _handler(signum, frame):
        killed = shutdown_agents()
        print(
            f"\ninterrupted: killed {killed} agent subprocess(es), shutting down...",
            file=sys.stderr,
        )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


def build_explore_config(args) -> "ExploreConfig":
    from dream_rsi.exploration.config import ExploreConfig

    cfg = ExploreConfig.from_env()
    if getattr(args, "problem", None) is not None:
        cfg.problem = args.problem
    elif DEFAULT_PROBLEM.is_file() and cfg.problem is None:
        cfg.problem = DEFAULT_PROBLEM
    if getattr(args, "evaluator", None) is not None:
        cfg.evaluator = args.evaluator
    elif DEFAULT_EVALUATOR.is_file() and cfg.evaluator is None:
        cfg.evaluator = DEFAULT_EVALUATOR
    if getattr(args, "init_program", None) is not None:
        cfg.init_program = args.init_program
    elif DEFAULT_INIT_PROGRAM.is_file() and cfg.init_program is None:
        cfg.init_program = DEFAULT_INIT_PROGRAM
    if getattr(args, "eval_program", None) is not None:
        cfg.eval_program = args.eval_program
    if getattr(args, "explore_prompt", None) is not None:
        cfg.explore_prompt = args.explore_prompt
    # evolve's --eval-timeout limits policy replays, not explore evaluations.
    if hasattr(args, "explore_eval_timeout"):
        eval_timeout = args.explore_eval_timeout
    else:
        eval_timeout = getattr(args, "eval_timeout", None)
    if eval_timeout is not None:
        cfg.eval_timeout_sec = eval_timeout
    return cfg
