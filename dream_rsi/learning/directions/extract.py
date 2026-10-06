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

"""Agent-backed, batched attempt-level semantic extraction."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from string import Template
from typing import Any

from dream_rsi.learning.directions.collect import AttemptSource
from dream_rsi.learning.directions.schema import AttemptExtraction
from dream_rsi.learning.directions.validate import validate_extractions

PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "direction_summarizer.txt"


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _run_agent(agent: Any, *, prompt: str, workspace: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result = agent.run(
        prompt=prompt,
        workspace=workspace,
        node_dir=run_dir,
        log_dir=run_dir / "agent_log" / agent.name,
        guide={},
        run_cwd=workspace,
        stage="learning",
    )
    if result.status != "ok":
        raise RuntimeError(f"direction summarizer failed: {result.message}")


def extract_batch(
    agent: Any,
    *,
    workspace: Path,
    attempts: list[AttemptSource],
    existing_catalog_path: Path | None,
    run_dir: Path,
) -> list[AttemptExtraction]:
    input_path = run_dir / "attempts.json"
    output_path = run_dir / "extractions.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_text(
        json.dumps([item.to_dict() for item in attempts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    previous = str(existing_catalog_path) if existing_catalog_path else "(none)"
    prompt = Template(_load_prompt()).safe_substitute(
        mode="extract",
        input_path=str(input_path),
        previous_catalog_path=previous,
        output_path=str(output_path),
        summary_id="not-applicable",
        task=workspace.name,
    )
    _run_agent(agent, prompt=prompt, workspace=workspace, run_dir=run_dir)
    if not output_path.is_file():
        raise FileNotFoundError(f"direction extraction output missing: {output_path}")
    data = json.loads(output_path.read_text(encoding="utf-8"))
    rows = data.get("extractions") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("direction extraction output must be a list or {extractions: [...]}")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("direction extraction rows must be JSON objects")
    extractions = [AttemptExtraction.from_dict(row) for row in rows]
    expected = {item.attempt_id for item in attempts}
    validate_extractions(extractions, expected_ids=expected)

    # A model may omit one otherwise valid row from a batch.  Keep the rows it did
    # produce: the summarizer records the coverage gap and retries those attempt
    # ids on a later summary.  Dropping every other valid batch would turn a small
    # omission into an entirely unguided next exploration cycle.

    # Identity, lineage, and measured score come from the collector, never from
    # the semantic agent.  The agent supplies only interpretation.  Directions
    # are never assigned to branches, so ``assigned_direction_id`` stays empty.
    source_by_id = {item.attempt_id: item for item in attempts}
    normalized: list[AttemptExtraction] = []
    for item in extractions:
        source = source_by_id[item.attempt_id]
        normalized.append(
            replace(
                item,
                cycle=source.cycle,
                branch=source.branch,
                attempt=source.attempt,
                parent_attempt_id=source.parent_attempt_id,
                assigned_direction_id=None,
                score=source.score,
                improved_over_parent=(
                    source.score > source.parent_score
                    if source.score is not None and source.parent_score is not None
                    else None
                ),
            )
        )
    validate_extractions(normalized, expected_ids=expected)
    return normalized
