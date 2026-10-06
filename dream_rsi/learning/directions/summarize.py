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

"""Cumulative direction summarization and safe catalog promotion."""

from __future__ import annotations

import json
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Template
from typing import Any

from dream_rsi.learning.directions.collect import collect_attempts
from dream_rsi.learning.directions.context import write_direction_context
from dream_rsi.learning.directions.extract import PROMPT_PATH, _run_agent
from dream_rsi.learning.directions.schema import AttemptExtraction, DirectionCatalog, load_catalog
from dream_rsi.learning.directions.validate import validate_catalog, validate_extractions


@dataclass(frozen=True)
class DirectionSummaryResult:
    ok: bool
    summary_id: str
    catalog_path: str | None = None
    snapshot_path: str | None = None
    context_path: str | None = None
    n_attempts: int = 0
    n_extracted: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary_id": self.summary_id,
            "catalog_path": self.catalog_path,
            "snapshot_path": self.snapshot_path,
            "context_path": self.context_path,
            "n_attempts": self.n_attempts,
            "n_extracted": self.n_extracted,
            "metrics": dict(self.metrics),
            "error": self.error,
        }


def infer_next_cycle(workspace: Path) -> int:
    """Infer a stable manual-workflow cycle from the latest catalog snapshot."""
    history = workspace.resolve() / "directions" / "history"
    latest = -1
    if history.is_dir():
        for path in history.glob("iter*.json"):
            match = re.match(r"^iter(\d+)(?:_r\d+)?$", path.stem)
            if match:
                latest = max(latest, int(match.group(1)))
    return latest + 1


def _summary_id(history_root: Path, cycle: int) -> str:
    """Return iterNNN, then immutable iterNNN_r001 revisions on same-cycle resume."""
    base = f"iter{cycle:03d}"
    if not (history_root / f"{base}.json").exists():
        return base
    revision = 1
    while (history_root / f"{base}_r{revision:03d}.json").exists():
        revision += 1
    return f"{base}_r{revision:03d}"


def _read_extractions(root: Path) -> dict[str, AttemptExtraction]:
    out: dict[str, AttemptExtraction] = {}
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                item = AttemptExtraction.from_dict(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if item.attempt_id:
                out[item.attempt_id] = item
    return out


def _append_extractions(root: Path, items: list[AttemptExtraction]) -> None:
    by_cycle: dict[str, list[AttemptExtraction]] = {}
    for item in items:
        by_cycle.setdefault(item.cycle, []).append(item)
    root.mkdir(parents=True, exist_ok=True)
    for cycle, cycle_items in by_cycle.items():
        path = root / f"{cycle}.jsonl"
        existing = _read_extractions(root)
        with path.open("a", encoding="utf-8") as fh:
            for item in cycle_items:
                if item.attempt_id in existing:
                    continue
                fh.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def _batch_size() -> int:
    try:
        return max(1, int(os.environ.get("DRSI_DIRECTION_EXTRACT_BATCH", "8")))
    except ValueError:
        return 8


def _extract_workers() -> int:
    """Return the independent concurrency limit for semantic extraction.

    ``DRSI_DIRECTION_EXTRACT_BATCH`` limits prompt size; it deliberately does
    not control concurrency.  This separate limit keeps extraction tunable
    without changing the amount of history each agent sees.
    """
    try:
        return max(1, int(os.environ.get("DRSI_DIRECTION_EXTRACT_WORKERS", "4")))
    except ValueError:
        return 4


def _aggregate_retries() -> int:
    """Return extra aggregate attempts after an agent-level transient failure.

    Gemini already retries each invocation, but an aggregate failure occurs only
    after all extraction batches have completed.  One fresh invocation prevents
    a rare empty stream from throwing away that completed work.  Keep this
    independently configurable because aggregate prompts can be large.
    """
    try:
        return max(0, int(os.environ.get("DRSI_DIRECTION_AGGREGATE_RETRIES", "1")))
    except ValueError:
        return 1


def _extract_missing_batches(
    agent: Any,
    *,
    workspace: Path,
    missing: list[Any],
    batch_size: int,
    existing_catalog_path: Path | None,
    run_root: Path,
) -> tuple[list[AttemptExtraction], list[str]]:
    """Extract independent batches concurrently, retaining valid partial work.

    Each batch only reads immutable attempt data and the prior catalog, and
    writes inside its own ``extract_NNNN`` directory.  A failed or incomplete
    batch does not invalidate valid rows from other batches: the caller records
    partial coverage, publishes the usable catalog, and leaves missing attempt
    ids eligible for the next summary.
    """
    from dream_rsi.learning.directions.extract import extract_batch

    batches = [
        missing[start : start + batch_size]
        for start in range(0, len(missing), batch_size)
    ]
    if not batches:
        return [], []

    workers = min(_extract_workers(), len(batches))

    def run_one(index: int) -> tuple[int, list[AttemptExtraction], str | None]:
        try:
            items = extract_batch(
                agent,
                workspace=workspace,
                attempts=batches[index],
                existing_catalog_path=existing_catalog_path,
                run_dir=run_root / f"extract_{index:04d}",
            )
            return index, items, None
        except Exception as exc:
            return index, [], f"extract_{index:04d}: {type(exc).__name__}: {exc}"

    # Preserve the exact former behavior when explicitly configured with one
    # worker, while using the same deterministic output ordering in both modes.
    if workers == 1:
        results: list[list[AttemptExtraction]] = []
        errors: list[str] = []
        for index in range(len(batches)):
            _, items, error = run_one(index)
            results.append(items)
            if error:
                errors.append(error)
        return [item for batch in results for item in batch], errors

    results: list[list[AttemptExtraction] | None] = [None] * len(batches)
    errors: list[str | None] = [None] * len(batches)
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="direction-extract",
    ) as executor:
        futures = {
            executor.submit(run_one, index): index
            for index in range(len(batches))
        }
        for future in as_completed(futures):
            index, items, error = future.result()
            results[index] = items
            errors[index] = error

    return (
        [item for batch in results if batch is not None for item in batch],
        [error for error in errors if error],
    )


def _aggregate_catalog(
    agent: Any,
    *,
    workspace: Path,
    summary_id: str,
    task: str,
    extraction_path: Path,
    known_attempt_ids: set[str],
    previous_catalog_path: Path | None,
    run_dir: Path,
) -> DirectionCatalog:
    output_path = run_dir / "catalog.proposed.json"
    previous = str(previous_catalog_path) if previous_catalog_path else "(none)"
    prompt = Template(PROMPT_PATH.read_text(encoding="utf-8")).safe_substitute(
        mode="aggregate",
        input_path=str(extraction_path),
        previous_catalog_path=previous,
        output_path=str(output_path),
        summary_id=summary_id,
        task=task,
    )
    _run_agent(agent, prompt=prompt, workspace=workspace, run_dir=run_dir)
    if not output_path.is_file():
        raise FileNotFoundError(f"direction catalog output missing: {output_path}")
    data = json.loads(output_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("direction catalog output must be a JSON object")
    catalog = DirectionCatalog.from_dict(data)
    validate_catalog(catalog, known_attempt_ids=known_attempt_ids)
    if catalog.summary_id != summary_id:
        raise ValueError(
            f"direction catalog summary_id {catalog.summary_id!r} != {summary_id!r}"
        )
    if previous_catalog_path is not None and previous_catalog_path.is_file():
        previous_catalog = load_catalog(previous_catalog_path)
        if catalog.task != previous_catalog.task:
            raise ValueError(
                f"direction catalog task changed from {previous_catalog.task!r} "
                f"to {catalog.task!r}"
            )
        previous_by_id = {item.id: item for item in previous_catalog.directions}
        current_by_id = {item.id: item for item in catalog.directions}
        previous_ids = set(previous_by_id)
        current_ids = set(current_by_id)
        missing = previous_ids - current_ids
        if missing:
            warnings.warn(
                "direction catalog dropped canonical ids: "
                + ", ".join(sorted(missing)),
                RuntimeWarning,
                stacklevel=2,
            )
        # A missing canonical ID is now non-fatal.  Compare immutable fields
        # only for directions that remain in the proposed catalog.
        for direction_id in sorted(previous_ids & current_ids):
            previous = previous_by_id[direction_id]
            current = current_by_id[direction_id]
            if current.canonical_name != previous.canonical_name:
                raise ValueError(
                    f"direction {direction_id!r} changed canonical_name from "
                    f"{previous.canonical_name!r} to {current.canonical_name!r}"
                )
            dropped_aliases = set(previous.aliases) - set(current.aliases)
            if dropped_aliases:
                raise ValueError(
                    f"direction {direction_id!r} dropped aliases: "
                    f"{', '.join(sorted(dropped_aliases))}"
                )
    return catalog


def _aggregate_catalog_with_retries(
    agent: Any,
    *,
    workspace: Path,
    summary_id: str,
    task: str,
    extraction_path: Path,
    known_attempt_ids: set[str],
    previous_catalog_path: Path | None,
    run_dir: Path,
) -> tuple[DirectionCatalog, int]:
    """Aggregate with a fresh agent invocation after transient agent failures.

    ``GeminiAgent`` handles retries within an invocation.  Retrying here is
    deliberately narrower: only the ``RuntimeError`` raised by ``_run_agent``
    is retried, so invalid catalogs and local filesystem/schema errors still
    fail immediately.  Retry logs live in separate directories for diagnosis.
    """
    retries = _aggregate_retries()
    for attempt in range(retries + 1):
        attempt_dir = run_dir if attempt == 0 else run_dir / f"retry_{attempt:03d}"
        try:
            return (
                _aggregate_catalog(
                    agent,
                    workspace=workspace,
                    summary_id=summary_id,
                    task=task,
                    extraction_path=extraction_path,
                    known_attempt_ids=known_attempt_ids,
                    previous_catalog_path=previous_catalog_path,
                    run_dir=attempt_dir,
                ),
                attempt + 1,
            )
        except RuntimeError:
            if attempt >= retries:
                raise

    raise AssertionError("aggregate retry loop exhausted unexpectedly")


def run_direction_summary(
    workspace: Path,
    agent: Any,
    *,
    cycle: int,
    task: str | None = None,
) -> DirectionSummaryResult:
    """Extract unseen attempts and atomically promote one cumulative catalog."""
    ws = workspace.resolve()
    directions = ws / "directions"
    extraction_root = directions / "extractions"
    history_root = directions / "history"
    summary_id = _summary_id(history_root, cycle)
    cycle_label = f"iter{cycle:03d}"
    input_root = directions / "inputs"
    run_root = directions / "runs" / summary_id
    current_path = directions / "current.json"
    previous_path = current_path if current_path.is_file() else None

    try:
        attempts = collect_attempts(ws, current_cycle=cycle)
        if not attempts:
            raise ValueError("no exploration attempts available for direction summarization")
        input_root.mkdir(parents=True, exist_ok=True)
        cumulative_input = input_root / f"{summary_id}.json"
        cumulative_input.write_text(
            json.dumps([item.to_dict() for item in attempts], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        existing = _read_extractions(extraction_root)
        missing = [item for item in attempts if item.attempt_id not in existing]
        size = _batch_size()
        new_items, extraction_errors = _extract_missing_batches(
            agent,
            workspace=ws,
            missing=missing,
            batch_size=size,
            existing_catalog_path=previous_path,
            run_root=run_root,
        )
        if new_items:
            _append_extractions(extraction_root, new_items)

        all_extractions = list(_read_extractions(extraction_root).values())
        validate_extractions(all_extractions)
        available_ids = {item.attempt_id for item in attempts}
        extracted_ids = {item.attempt_id for item in all_extractions}
        covered_ids = available_ids & extracted_ids
        missing_ids = sorted(available_ids - extracted_ids)
        if not covered_ids:
            raise ValueError("direction summarization produced no valid extractions")

        from dream_rsi.learning.directions.metrics import compute_direction_metrics

        metrics = compute_direction_metrics(
            cycle=cycle_label,
            extractions=all_extractions,
        )
        metrics["coverage"] = {
            "available_attempt_count": len(available_ids),
            "extracted_attempt_count": len(covered_ids),
            "missing_attempt_count": len(missing_ids),
            "complete": not missing_ids,
            "failed_batch_count": len(extraction_errors),
        }
        if extraction_errors:
            metrics["extraction_errors"] = extraction_errors
        aggregation_input = run_root / "all_extractions.json"
        aggregation_input.parent.mkdir(parents=True, exist_ok=True)
        aggregation_input.write_text(
            json.dumps(
                [item.to_dict() for item in sorted(all_extractions, key=lambda x: x.attempt_id)],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        catalog, aggregate_attempt_count = _aggregate_catalog_with_retries(
            agent,
            workspace=ws,
            summary_id=summary_id,
            task=task or ws.name,
            extraction_path=aggregation_input,
            known_attempt_ids={item.attempt_id for item in all_extractions},
            previous_catalog_path=previous_path,
            run_dir=run_root / "aggregate",
        )
        metrics["aggregation"] = {
            "attempt_count": aggregate_attempt_count,
            "retry_count": aggregate_attempt_count - 1,
        }
        catalog = replace(
            catalog,
            generated_from={
                "cycles": sorted({item.cycle for item in all_extractions}),
                "attempt_count": len(all_extractions),
                "available_attempt_count": len(available_ids),
                "missing_attempt_count": len(missing_ids),
                "coverage_complete": not missing_ids,
            },
        )
        validate_catalog(
            catalog,
            known_attempt_ids={item.attempt_id for item in all_extractions},
        )

        directions.mkdir(parents=True, exist_ok=True)
        history_root.mkdir(parents=True, exist_ok=True)
        snapshot = history_root / f"{summary_id}.json"
        if snapshot.exists():
            prior = load_catalog(snapshot)
            if prior.to_dict() != catalog.to_dict():
                raise FileExistsError(f"immutable direction snapshot already differs: {snapshot}")
        else:
            snapshot.write_text(
                json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        proposed = directions / "current.proposed.json"
        proposed.write_text(
            json.dumps(catalog.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        load_catalog(proposed)
        proposed.replace(current_path)
        context = write_direction_context(catalog, directions / "current.md")
        context_snapshot = history_root / f"{summary_id}.md"
        rendered = context.read_text(encoding="utf-8")
        if context_snapshot.exists():
            if context_snapshot.read_text(encoding="utf-8") != rendered:
                raise FileExistsError(
                    f"immutable direction context snapshot already differs: {context_snapshot}"
                )
        else:
            context_snapshot.write_text(rendered, encoding="utf-8")
        return DirectionSummaryResult(
            ok=True,
            summary_id=summary_id,
            catalog_path=str(current_path),
            snapshot_path=str(snapshot),
            context_path=str(context),
            n_attempts=len(attempts),
            n_extracted=len(new_items),
            metrics=metrics,
        )
    except Exception as exc:
        return DirectionSummaryResult(
            ok=False,
            summary_id=summary_id,
            catalog_path=str(current_path) if current_path.is_file() else None,
            n_attempts=len(locals().get("attempts", [])),
            n_extracted=len(locals().get("new_items", [])),
            metrics=dict(locals().get("metrics", {})),
            error=f"{type(exc).__name__}: {exc}",
        )
