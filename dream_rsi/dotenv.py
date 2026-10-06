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

"""Load DREAM-RSI environment variables from a .env file."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# First existing file wins: DRSI_ENV_FILE > project/.env > cwd/.env
ENV_CANDIDATES = (
    lambda: Path(os.environ["DRSI_ENV_FILE"]).expanduser()
    if os.environ.get("DRSI_ENV_FILE")
    else None,
    lambda: ROOT / ".env",
    lambda: Path.cwd() / ".env",
)


def resolve_env_file() -> Path | None:
    for getter in ENV_CANDIDATES:
        path = getter()
        if path is not None and path.is_file():
            return path.resolve()
    return None


def load_env(*, override: bool = False) -> Path | None:
    """
    Load KEY=VALUE pairs from .env into os.environ.

    By default, variables already set in the shell are kept.
    Pass override=True to let .env win.
    """
    path = resolve_env_file()
    if path is None:
        return None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]

        if not override and key in os.environ:
            continue
        os.environ[key] = value

    return path


def _gemini_key_count() -> int:
    multi = os.environ.get("GEMINI_API_KEYS", "").strip()
    if multi:
        return len([k for k in multi.replace("\n", ",").split(",") if k.strip()])
    return 1 if (
        os.environ.get("GEMINI_API_KEY", "").strip()
        or os.environ.get("GOOGLE_API_KEY", "").strip()
    ) else 0


def gemini_ready() -> bool:
    return _gemini_key_count() > 0
