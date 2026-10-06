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

"""Copy workspace explore archives into each branch for agent conditioning."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dream_rsi.env.trace_store import grid_limits

HISTORY_DIRNAME = "history"


def list_workspace_archives(workspace: Path) -> list[Path]:
    """Sorted archive dirs under ``workspace/history/`` (iter*, manual*, pre_evolve*, …)."""
    root = workspace.resolve() / "history"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _seed_mode() -> str:
    """``copy`` (default, as in the paper runs) or ``symlink`` to save disk space."""
    return os.environ.get("DRSI_HISTORY_SEED_MODE", "copy").strip().lower()


def seed_history_to_branches(
    workspace: Path,
    *,
    max_width: int | None = None,
    max_refine: int | None = None,
) -> list[str]:
    """Link or copy each ``workspace/history/*`` archive into ``explore_bN/history/``.

    Called before live explore so agents read prior turns via ``./history/...``.
    Returns branch dirs that received history seeds.
    """
    archives = list_workspace_archives(workspace)
    if not archives:
        return []

    ws = workspace.resolve()
    m, _ = grid_limits(max_width=max_width, max_refine=max_refine)
    mode = _seed_mode()
    use_symlink = mode in ("symlink", "link")

    touched: list[str] = []
    for branch in range(m):
        branch_dir = ws / f"explore_b{branch}"
        branch_dir.mkdir(parents=True, exist_ok=True)
        dst_root = branch_dir / HISTORY_DIRNAME
        dst_root.mkdir(exist_ok=True)

        for arch in archives:
            dst = dst_root / arch.name
            if dst.exists() or dst.is_symlink():
                continue
            if use_symlink:
                dst.symlink_to(arch.resolve(), target_is_directory=True)
            else:
                shutil.copytree(arch, dst, dirs_exist_ok=True, symlinks=True)
        touched.append(str(branch_dir))

    names = ", ".join(a.name for a in archives)
    print(
        f"[history_seed] {len(archives)} archive(s) → {len(touched)} branch(es) "
        f"({mode}): {names}",
        flush=True,
    )
    return touched
