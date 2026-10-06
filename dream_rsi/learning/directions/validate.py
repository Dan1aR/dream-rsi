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

"""Strict structural validation for direction-learning artifacts."""

from __future__ import annotations

import math

from dream_rsi.learning.directions.schema import (
    CANDIDATE_ORIGINS,
    CATALOG_SCHEMA_VERSION,
    DIRECTION_STATUSES,
    OUTCOME_CLASSES,
    AttemptExtraction,
    DirectionCatalog,
)

INTENSITY_FIELDS = frozenset(
    {
        "n_roots",
        "n_refinements",
        "n_rounds",
        "n_successes",
        "n_failures",
        "n_improvements",
    }
)


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes


def validate_catalog(
    catalog: DirectionCatalog, *, known_attempt_ids: set[str] | None = None
) -> None:
    if catalog.schema_version != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported direction catalog schema {catalog.schema_version}; "
            f"expected {CATALOG_SCHEMA_VERSION}"
        )
    if not catalog.summary_id:
        raise ValueError("direction catalog summary_id is required")
    if not catalog.task:
        raise ValueError("direction catalog task is required")
    cycles = catalog.generated_from.get("cycles")
    if (
        not isinstance(cycles, list)
        or any(not isinstance(item, str) or not item.strip() for item in cycles)
        or len(cycles) != len(set(cycles))
    ):
        raise ValueError(
            "direction catalog generated_from.cycles must be unique nonempty strings"
        )
    attempt_count = catalog.generated_from.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 0
    ):
        raise ValueError("direction catalog attempt_count must be a nonnegative integer")
    ids = [item.id for item in catalog.directions] + [item.id for item in catalog.candidates]
    if any(not item_id for item_id in ids):
        raise ValueError("direction and candidate ids must be nonempty")
    dupes = _duplicates(ids)
    if dupes:
        raise ValueError(f"duplicate direction ids: {', '.join(sorted(dupes))}")

    direction_ids = {item.id for item in catalog.directions}
    for item in catalog.directions:
        if (
            not item.canonical_name
            or not item.title
            or not item.description
            or not item.target_bottleneck
        ):
            raise ValueError(
                f"direction {item.id!r} is missing "
                "name/title/description/target_bottleneck"
            )
        if item.status not in DIRECTION_STATUSES:
            raise ValueError(f"direction {item.id!r} has invalid status {item.status!r}")
        missing_intensity = INTENSITY_FIELDS - set(item.intensity)
        if missing_intensity:
            raise ValueError(
                f"direction {item.id!r} is missing intensity fields: "
                f"{', '.join(sorted(missing_intensity))}"
            )
        if any(value < 0 for value in item.intensity.values()):
            raise ValueError(f"direction {item.id!r} has a negative intensity count")
        missing_outcomes = {"best_score", "latest_score"} - set(item.outcomes)
        if missing_outcomes:
            raise ValueError(
                f"direction {item.id!r} is missing outcome fields: "
                f"{', '.join(sorted(missing_outcomes))}"
            )
        if known_attempt_ids is not None:
            unknown_evidence = set(item.evidence_attempts) - known_attempt_ids
            if unknown_evidence:
                raise ValueError(
                    f"direction {item.id!r} references unknown attempts: "
                    f"{', '.join(sorted(unknown_evidence))}"
                )

    for item in catalog.candidates:
        if (
            not item.title
            or not item.instruction
            or not item.reason
            or not item.target_bottleneck
        ):
            raise ValueError(
                f"candidate {item.id!r} is missing "
                "title/instruction/reason/target_bottleneck"
            )
        if item.origin not in CANDIDATE_ORIGINS:
            raise ValueError(f"candidate {item.id!r} has invalid origin {item.origin!r}")
        if not math.isfinite(item.priority) or not 0.0 <= item.priority <= 1.0:
            raise ValueError(f"candidate {item.id!r} priority must be in [0, 1]")
        missing = set(item.related_direction_ids) - direction_ids
        if missing:
            raise ValueError(
                f"candidate {item.id!r} references unknown directions: "
                f"{', '.join(sorted(missing))}"
            )
        if known_attempt_ids is not None:
            unknown_evidence = set(item.evidence_attempts) - known_attempt_ids
            if unknown_evidence:
                raise ValueError(
                    f"candidate {item.id!r} references unknown attempts: "
                    f"{', '.join(sorted(unknown_evidence))}"
                )


def validate_extractions(
    extractions: list[AttemptExtraction], *, expected_ids: set[str] | None = None
) -> None:
    ids = [item.attempt_id for item in extractions]
    if any(not item_id for item_id in ids):
        raise ValueError("attempt extraction id is required")
    dupes = _duplicates(ids)
    if dupes:
        raise ValueError(f"duplicate extraction ids: {', '.join(sorted(dupes))}")
    if expected_ids is not None:
        unknown = set(ids) - expected_ids
        if unknown:
            raise ValueError(f"extractions contain unknown attempts: {', '.join(sorted(unknown))}")
    for item in extractions:
        if item.branch < 0 or item.attempt < 0:
            raise ValueError(f"invalid branch/attempt for {item.attempt_id}")
        if not item.cycle:
            raise ValueError(f"extraction {item.attempt_id} has no cycle")
        if not item.primary_mechanism:
            raise ValueError(f"extraction {item.attempt_id} has no primary_mechanism")
        if item.outcome_class not in OUTCOME_CLASSES:
            raise ValueError(
                f"extraction {item.attempt_id} has invalid outcome_class "
                f"{item.outcome_class!r}"
            )
        if not item.evidence:
            raise ValueError(f"extraction {item.attempt_id} has no evidence")
        if item.score is not None and not math.isfinite(item.score):
            raise ValueError(f"extraction {item.attempt_id} has a non-finite score")
        if item.compliance_confidence is not None and not 0 <= item.compliance_confidence <= 1:
            raise ValueError(f"invalid compliance confidence for {item.attempt_id}")
