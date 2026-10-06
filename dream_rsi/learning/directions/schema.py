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

"""Versioned JSON contracts for direction extraction and catalogs."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CATALOG_SCHEMA_VERSION = 1

DIRECTION_STATUSES = frozenset(
    {
        "unexplored",
        "underexplored",
        "promising",
        "promising_but_fragile",
        "promising_but_underexplored",
        "well_explored",
        "saturated",
        "failed_but_recoverable",
        "unpromising",
    }
)
CANDIDATE_ORIGINS = frozenset({"new", "underexplored", "combination", "repair"})
OUTCOME_CLASSES = frozenset(
    {
        "success",
        "conceptual_failure",
        "implementation_failure",
        "compile_failure",
        "runtime_failure",
        "unknown",
    }
)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [str(item) for item in value if str(item).strip()]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a numeric score")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric values must be finite")
    return result


def _dict_list(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{field_name} must be a list of objects")
    return [dict(item) for item in value]


@dataclass(frozen=True)
class DirectionCandidate:
    id: str
    title: str
    instruction: str
    priority: float
    reason: str
    origin: str = "new"
    target_bottleneck: str = ""
    related_direction_ids: list[str] = field(default_factory=list)
    avoid_repeating: list[str] = field(default_factory=list)
    evidence_attempts: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionCandidate":
        priority = data.get("priority", 0.0)
        if isinstance(priority, bool) or not isinstance(priority, (int, float)):
            raise ValueError("candidate priority must be numeric")
        return cls(
            id=str(data.get("id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            instruction=str(data.get("instruction") or "").strip(),
            priority=float(priority),
            reason=str(data.get("reason") or "").strip(),
            origin=str(data.get("origin") or "new").strip(),
            target_bottleneck=str(data.get("target_bottleneck") or "").strip(),
            related_direction_ids=_string_list(data.get("related_direction_ids")),
            avoid_repeating=_string_list(data.get("avoid_repeating")),
            evidence_attempts=_string_list(data.get("evidence_attempts")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionRecord:
    id: str
    canonical_name: str
    title: str
    description: str
    status: str
    aliases: list[str] = field(default_factory=list)
    target_bottleneck: str = ""
    intensity: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, float | None] = field(default_factory=dict)
    failure_analysis: list[dict[str, Any]] = field(default_factory=list)
    evidence_attempts: list[str] = field(default_factory=list)
    future_suggestions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionRecord":
        raw_intensity = data.get("intensity")
        raw_outcomes = data.get("outcomes")
        if not isinstance(raw_intensity, dict):
            raise ValueError("direction intensity must be an object")
        if not isinstance(raw_outcomes, dict):
            raise ValueError("direction outcomes must be an object")
        intensity: dict[str, int] = {}
        for key, value in raw_intensity.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"direction intensity {key!r} must be an integer")
            intensity[str(key)] = value
        outcomes = {
            str(key): _optional_float(value)
            for key, value in raw_outcomes.items()
        }
        return cls(
            id=str(data.get("id") or "").strip(),
            canonical_name=str(data.get("canonical_name") or "").strip(),
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            status=str(data.get("status") or "").strip(),
            aliases=_string_list(data.get("aliases")),
            target_bottleneck=str(data.get("target_bottleneck") or "").strip(),
            intensity=intensity,
            outcomes=outcomes,
            failure_analysis=_dict_list(
                data.get("failure_analysis"), field_name="failure_analysis"
            ),
            evidence_attempts=_string_list(data.get("evidence_attempts")),
            future_suggestions=_string_list(data.get("future_suggestions")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptExtraction:
    attempt_id: str
    cycle: str
    branch: int
    attempt: int
    parent_attempt_id: str | None
    assigned_direction_id: str | None
    primary_mechanism: str
    supporting_mechanisms: list[str]
    outcome_class: str
    score: float | None
    improved_over_parent: bool | None
    evidence: str
    actual_direction_id: str | None = None
    agent_claimed_compliance: bool | None = None
    summarizer_assessed_compliance: bool | None = None
    compliance_confidence: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttemptExtraction":
        def optional_bool(value: Any) -> bool | None:
            return value if isinstance(value, bool) else None

        return cls(
            attempt_id=str(data.get("attempt_id") or "").strip(),
            cycle=str(data.get("cycle") or "").strip(),
            branch=int(data.get("branch", 0)),
            attempt=int(data.get("attempt", 0)),
            parent_attempt_id=(
                str(data["parent_attempt_id"]).strip()
                if data.get("parent_attempt_id")
                else None
            ),
            assigned_direction_id=(
                str(data["assigned_direction_id"]).strip()
                if data.get("assigned_direction_id")
                else None
            ),
            primary_mechanism=str(data.get("primary_mechanism") or "").strip(),
            supporting_mechanisms=_string_list(data.get("supporting_mechanisms")),
            outcome_class=str(data.get("outcome_class") or "unknown").strip(),
            score=_optional_float(data.get("score")),
            improved_over_parent=optional_bool(data.get("improved_over_parent")),
            evidence=str(data.get("evidence") or "").strip(),
            actual_direction_id=(
                str(data["actual_direction_id"]).strip()
                if data.get("actual_direction_id")
                else None
            ),
            agent_claimed_compliance=optional_bool(data.get("agent_claimed_compliance")),
            summarizer_assessed_compliance=optional_bool(
                data.get("summarizer_assessed_compliance")
            ),
            compliance_confidence=_optional_float(data.get("compliance_confidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionCatalog:
    schema_version: int
    summary_id: str
    task: str
    generated_from: dict[str, Any]
    directions: list[DirectionRecord]
    candidates: list[DirectionCandidate]

    @classmethod
    def empty(cls, *, summary_id: str = "empty", task: str = "unknown") -> "DirectionCatalog":
        return cls(
            schema_version=CATALOG_SCHEMA_VERSION,
            summary_id=summary_id,
            task=task,
            generated_from={"cycles": [], "attempt_count": 0},
            directions=[],
            candidates=[],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DirectionCatalog":
        generated_from = data.get("generated_from")
        directions = data.get("directions")
        candidates = data.get("candidates")
        if not isinstance(generated_from, dict):
            raise ValueError("direction catalog generated_from must be an object")
        if not isinstance(directions, list) or any(
            not isinstance(item, dict) for item in directions
        ):
            raise ValueError("direction catalog directions must be a list of objects")
        if not isinstance(candidates, list) or any(
            not isinstance(item, dict) for item in candidates
        ):
            raise ValueError("direction catalog candidates must be a list of objects")
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            summary_id=str(data.get("summary_id") or "").strip(),
            task=str(data.get("task") or "unknown").strip(),
            generated_from=dict(generated_from),
            directions=[
                DirectionRecord.from_dict(item) for item in directions
            ],
            candidates=[
                DirectionCandidate.from_dict(item) for item in candidates
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "summary_id": self.summary_id,
            "task": self.task,
            "generated_from": dict(self.generated_from),
            "directions": [item.to_dict() for item in self.directions],
            "candidates": [item.to_dict() for item in self.candidates],
        }


def load_catalog(path: str | Path) -> DirectionCatalog:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"direction catalog must contain a JSON object: {path}")
    from dream_rsi.learning.directions.validate import validate_catalog

    catalog = DirectionCatalog.from_dict(data)
    validate_catalog(catalog)
    return catalog
