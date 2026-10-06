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

"""Load an exploration policy: the baseline or a file defining ``OptimalPolicy``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Literal

from dream_rsi.policy.api import LLMDesignedMethod
from dream_rsi.policy.baseline import ParallelRefineMethod
from dream_rsi.policy.constants import PROMOTED_POLICY_REL


def promoted_policy_path(workspace: Path) -> Path:
    """Where the learning phase promotes the next online policy of ``workspace``."""
    return Path(workspace).resolve() / PROMOTED_POLICY_REL


def default_policy_source(workspace: Path | None = None) -> str:
    """The workspace's promoted policy if present, else ``"baseline"``."""
    if workspace is not None:
        path = promoted_policy_path(workspace)
        if path.is_file():
            return str(path)
    return "baseline"


def load_policy_class(path: str | Path) -> type[LLMDesignedMethod]:
    """Import ``path`` and return its ``OptimalPolicy`` class."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"policy file not found: {path}")
    spec = importlib.util.spec_from_file_location("dream_rsi_policy", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "OptimalPolicy"):
        raise AttributeError(f"{path} has no OptimalPolicy class")
    return module.OptimalPolicy


def load_policy(
    source: str | Path | Literal["baseline"] = "baseline",
    *,
    config: dict[str, Any] | None = None,
) -> LLMDesignedMethod:
    """Instantiate ``source``: ``"baseline"`` or a policy file path."""
    if source == "baseline":
        return ParallelRefineMethod(config)
    return load_policy_class(source)(config)
