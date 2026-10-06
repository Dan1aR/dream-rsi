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

"""Render and load the global direction memory injected into explore prompts.

The JSON catalog remains the authoritative, validated machine artifact.  This
module derives a concise Markdown view for agents.  It intentionally does not
assign directions to branches: every live agent receives the same cumulative
memory and chooses its concrete mechanism independently.
"""

from __future__ import annotations

import os
from pathlib import Path

from dream_rsi.learning.directions.schema import DirectionCatalog, DirectionRecord, load_catalog

GLOBAL_DIRECTION_MODE = "global"
OFF_DIRECTION_MODE = "off"
DIRECTION_MODES = frozenset({OFF_DIRECTION_MODE, GLOBAL_DIRECTION_MODE})
CONTEXT_REL = Path("directions/current.md")
CATALOG_REL = Path("directions/current.json")
MAX_CONTEXT_CHARS = 12_000


def resolve_direction_mode(value: str | None = None) -> str:
    """Resolve the direction mode: ``off`` (default) or ``global``.

    ``value`` wins; otherwise ``DRSI_DIRECTION_MODE`` is consulted.
    """
    raw = value if value is not None else os.environ.get("DRSI_DIRECTION_MODE", "")
    mode = raw.strip().lower() or OFF_DIRECTION_MODE
    if mode not in DIRECTION_MODES:
        choices = ", ".join(sorted(DIRECTION_MODES))
        raise ValueError(f"invalid direction mode {raw!r}; choose one of: {choices}")
    return mode


def context_path(workspace: Path) -> Path:
    return workspace.resolve() / CONTEXT_REL


def catalog_path(workspace: Path) -> Path:
    return workspace.resolve() / CATALOG_REL


def _clip(text: str, limit: int = 600) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _intensity(record: DirectionRecord) -> str:
    data = record.intensity
    return (
        f"roots={data.get('n_roots', 0)}, refines={data.get('n_refinements', 0)}, "
        f"rounds={data.get('n_rounds', 0)}, improvements={data.get('n_improvements', 0)}, "
        f"failures={data.get('n_failures', 0)}"
    )


def _outcomes(record: DirectionRecord) -> str:
    data = record.outcomes
    best = data.get("best_score")
    latest = data.get("latest_score")
    return f"best={best if best is not None else '-'}, latest={latest if latest is not None else '-'}"


def _record_line(record: DirectionRecord, *, include_failure: bool) -> list[str]:
    lines = [
        f"- **{record.title}** (`{record.status}`; {_intensity(record)}; {_outcomes(record)})",
        f"  Mechanism: {_clip(record.description, 500)}",
    ]
    if include_failure and record.failure_analysis:
        failure = record.failure_analysis[0]
        lines.append(
            f"  Evidence/caution: {_clip(str(failure.get('summary') or failure.get('class') or ''), 420)}"
        )
    return lines


def render_direction_context(catalog: DirectionCatalog) -> str:
    """Create a bounded, deterministic global prompt context from a catalog."""
    avoid_statuses = {"well_explored", "saturated", "unpromising"}
    promising_statuses = {
        "unexplored",
        "underexplored",
        "promising",
        "promising_but_fragile",
        "promising_but_underexplored",
        "failed_but_recoverable",
    }
    avoid = sorted(
        (item for item in catalog.directions if item.status in avoid_statuses),
        key=lambda item: (-item.intensity.get("n_rounds", 0), item.title, item.id),
    )[:8]
    promising = sorted(
        (item for item in catalog.directions if item.status in promising_statuses),
        key=lambda item: (
            item.status not in {"promising", "promising_but_underexplored"},
            -item.intensity.get("n_improvements", 0),
            item.title,
            item.id,
        ),
    )[:8]
    candidates = sorted(catalog.candidates, key=lambda item: (-item.priority, item.id))[:12]

    generated = catalog.generated_from
    available = generated.get("available_attempt_count")
    extracted = generated.get("attempt_count")
    missing = generated.get("missing_attempt_count")
    coverage_lines: list[str] = []
    if isinstance(available, int) and isinstance(extracted, int) and isinstance(missing, int):
        if missing > 0:
            coverage_lines = [
                f"Coverage: partial ({extracted}/{available} extracted; {missing} pending). "
                "Treat absent history as unknown, not negative evidence.",
            ]
        else:
            coverage_lines = [f"Coverage: complete ({extracted}/{available} attempts extracted)."]

    lines = [
        "# Cumulative exploration direction memory",
        "",
        f"Summary ID: {catalog.summary_id}",
        f"Task: {catalog.task}",
        "This is cumulative evidence from completed earlier exploration cycles. "
        "Use it to avoid blind repetition and choose a concrete mechanism for this attempt.",
        *coverage_lines,
        "",
        "## Well explored or saturated directions — do not repeat without a concrete repair hypothesis",
    ]
    if avoid:
        for record in avoid:
            lines.extend(_record_line(record, include_failure=True))
    else:
        lines.append("- No direction is yet classified as saturated.")

    lines.extend(["", "## Promising or underexplored directions"])
    if promising:
        for record in promising:
            lines.extend(_record_line(record, include_failure=True))
    else:
        lines.append("- No prior canonical direction is available; seek a concrete novel mechanism.")

    lines.extend(["", "## Candidate mechanisms for future exploration"])
    if candidates:
        for candidate in candidates:
            related = ", ".join(candidate.related_direction_ids) or "none"
            avoid_repeat = "; ".join(candidate.avoid_repeating) or "none specified"
            lines.extend(
                [
                    f"- **{candidate.title}** (priority={candidate.priority:.2f}, origin={candidate.origin})",
                    f"  Try: {_clip(candidate.instruction, 560)}",
                    f"  Why: {_clip(candidate.reason, 420)}",
                    f"  Related directions: {related}. Avoid repeating: {_clip(avoid_repeat, 300)}",
                ]
            )
    else:
        lines.append("- No explicit candidate is available; propose a genuinely distinct mechanism and explain why.")

    lines.extend(
        [
            "",
            "## Required use in this attempt",
            "- Choose one concrete mechanism from the promising/candidate evidence, or justify a genuinely new one.",
            "- Do not merely restate a saturated direction or repeat its known failure mode.",
            "- In proposal.md, state which prior direction you extend, avoid, repair, or differ from.",
            "- For a refinement, preserve the parent mechanism unless you diagnose a specific reason for a controlled pivot.",
        ]
    )
    text = "\n".join(lines).strip() + "\n"
    if len(text) <= MAX_CONTEXT_CHARS:
        return text
    return text[: MAX_CONTEXT_CHARS - 64].rstrip() + "\n\n[direction memory truncated]\n"


def write_direction_context(catalog: DirectionCatalog, path: Path) -> Path:
    """Atomically write the Markdown rendering paired with ``current.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(render_direction_context(catalog), encoding="utf-8")
    tmp.replace(path)
    return path


def load_global_direction_context(workspace: Path) -> tuple[str, DirectionCatalog]:
    """Load a validated catalog and its matching Markdown prompt artifact.

    Global mode is intentionally strict: a requested global context must exist
    and match the current catalog rather than silently degrading to unguided
    prompts.
    """
    catalog_file = catalog_path(workspace)
    if not catalog_file.is_file():
        raise FileNotFoundError(
            f"global direction context missing (catalog): {catalog_file}"
        )
    catalog = load_catalog(catalog_file)
    path = context_path(workspace)
    if not path.is_file():
        raise FileNotFoundError(f"global direction context missing: {path}")
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        raise ValueError(f"global direction context is empty: {path}")
    marker = f"Summary ID: {catalog.summary_id}"
    if marker not in text:
        raise ValueError(
            f"global direction context does not match catalog {catalog.summary_id}: {path}"
        )
    return text, catalog
