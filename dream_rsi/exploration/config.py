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

"""Run configuration. Pass paths explicitly; DRSI_* env vars are a fallback."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExploreConfig:
    """All run inputs in one place.

    problem        source problem spec; copied to <workspace>/problem.txt at start
    evaluator      evaluator.py exposing evaluate(program_path) -> dict (None disables eval)
    init_program   baseline seed program (None disables baseline seeding)
    eval_program   where the agent writes / evaluator reads the candidate (rel to a node dir)
    explore_prompt path to explore agent prompt template (default: dream_rsi/prompts/explore.txt)
    """

    problem: Path | None = None
    evaluator: Path | None = None
    init_program: Path | None = None
    eval_program: str = "artifact/program.py"
    explore_prompt: Path | None = None
    eval_timeout_sec: float = 600.0
    eval_python: str | None = None

    def eval_enabled(self) -> bool:
        return self.evaluator is not None

    @classmethod
    def from_env(cls) -> "ExploreConfig":
        """Build from DRSI_* environment variables (backward-compatible fallback)."""

        def _path(name: str) -> Path | None:
            v = os.environ.get(name, "").strip()
            return Path(v) if v else None

        return cls(
            problem=_path("DRSI_PROBLEM"),
            evaluator=_path("DRSI_EVALUATOR"),
            init_program=_path("DRSI_INIT_PROGRAM"),
            eval_program=os.environ.get("DRSI_EVAL_PROGRAM", "").strip()
            or "artifact/program.py",
            explore_prompt=_path("DRSI_EXPLORE_PROMPT"),
            eval_timeout_sec=float(os.environ.get("DRSI_EVAL_TIMEOUT_SEC", "600")),
            eval_python=os.environ.get("DRSI_EVAL_PYTHON", "").strip() or None,
        )
