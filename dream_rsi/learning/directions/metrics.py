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

"""Root-direction diversity metrics for one exploration cycle."""

from __future__ import annotations

from typing import Any

from dream_rsi.learning.directions.schema import AttemptExtraction


def _duplicate_stats(values: list[str]) -> dict[str, Any]:
    unique = len(set(values))
    duplicates = max(0, len(values) - unique)
    return {
        "count": len(values),
        "unique": unique,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / len(values) if values else 0.0,
    }


def compute_direction_metrics(
    *,
    cycle: str,
    extractions: list[AttemptExtraction] | None = None,
) -> dict[str, Any]:
    """Compute how diverse the mechanisms of this cycle's new roots were."""
    roots = [
        item
        for item in (extractions or [])
        if item.cycle == cycle and item.attempt == 0
    ]
    actual = [
        item.actual_direction_id or item.primary_mechanism
        for item in roots
        if item.primary_mechanism and item.primary_mechanism != "unknown_no_mechanism"
    ]
    return {
        "cycle": cycle,
        "n_roots": len(roots),
        "actual": _duplicate_stats(actual),
    }
