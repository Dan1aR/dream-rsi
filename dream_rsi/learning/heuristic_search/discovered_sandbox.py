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

"""Agent-editable sandbox: ``discovered/method.py`` only.

Framework modules (``api``, ``grid_planning``, ``observation_signal``,
``baseline``) are read-only. Each proposal round the workflow resets
``method.py`` from ``method.template.py``, except that the first proposal
starts from a seeded incumbent policy.
"""

from __future__ import annotations

import shutil
from pathlib import Path

EDITABLE_FILES: tuple[str, ...] = ("method.py",)


def template_path(discovered_dir: Path, filename: str) -> Path:
    return discovered_dir / f"{Path(filename).stem}.template.py"


def editable_paths(method_file: str) -> list[str]:
    parent = Path(method_file).parent
    return [f"{parent.as_posix()}/{name}" for name in EDITABLE_FILES]


def reset_from_templates(discovered_dir: Path) -> list[str]:
    """Copy ``method.template.py`` → ``method.py``."""
    discovered_dir.mkdir(parents=True, exist_ok=True)
    reset: list[str] = []
    for name in EDITABLE_FILES:
        tpl = template_path(discovered_dir, name)
        dst = discovered_dir / name
        if tpl.is_file():
            shutil.copy2(tpl, dst)
            reset.append(name)
    return reset


def discovered_dir_for_method(method_file: Path) -> Path:
    return method_file.resolve().parent
