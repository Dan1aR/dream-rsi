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

"""Build the agent prompt and seed each branch with its readable context
(problem spec + previous-SOTA baseline)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from string import Template
from typing import Any

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.evaluate import evaluate_node
from dream_rsi.exploration.store import NodeStore

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_EXPLORE_PROMPT = PROMPTS_DIR / "explore.txt"
BASELINE_DIRNAME = "baseline"
INIT_DIRNAME = "init"
BASELINE_META = "baseline_meta.json"


def _resolve_explore_prompt_path(config: ExploreConfig) -> Path:
    if config.explore_prompt is not None:
        path = config.explore_prompt.expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Explore prompt not found: {path}")
        return path
    return DEFAULT_EXPLORE_PROMPT


def _load_template(config: ExploreConfig | None = None) -> str:
    config = config or ExploreConfig.from_env()
    return _resolve_explore_prompt_path(config).read_text(encoding="utf-8")


def _render(template: str, /, **values: Any) -> str:
    """Fill $placeholders. safe_substitute never re-scans the injected values,
    so code blocks containing $ or {} are left intact."""
    return Template(template).safe_substitute(**values)


def _global_direction_guidance(direction_context: str | None) -> str:
    """Return the shared cumulative memory used by global direction mode."""
    text = (direction_context or "").strip()
    if not text:
        return ""
    return f"""
## Cumulative exploration guidance

{text}
""".strip()


def explore_prompt(
    node_dir: Path,
    config: ExploreConfig | None = None,
    *,
    direction_context: str | None = None,
) -> str:
    config = config or ExploreConfig.from_env()
    template = _load_template(config)
    guidance = _global_direction_guidance(direction_context)
    has_guidance_slot = (
        "$direction_guidance" in template or "${direction_guidance}" in template
    )
    rendered = _render(
        template,
        problem_file="./problem.txt",
        baseline_dir="./baseline/",
        history_glob="./attempt_*/",
        history_dir="./history/",
        node_dir=f"./{node_dir.name}",
        eval_program=config.eval_program,
        direction_guidance=guidance,
    )
    if guidance and not has_guidance_slot:
        return guidance + "\n\n" + rendered
    return rendered


def _read_score(score_json: Path) -> tuple[float, float | None]:
    """Return (combined_score, latency_ms) from an eval/score.json (0.0 if absent)."""
    try:
        data = json.loads(score_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0, None
    raw = data.get("combined_score", data.get("score"))
    try:
        score = float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        score = 0.0
    lat = (data.get("metadata") or {}).get("latency_ms")
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    return score, lat


def _load_baseline_meta(bundle: Path) -> dict[str, Any]:
    meta_path = bundle / BASELINE_META
    if not meta_path.is_file():
        return {"init": None, "current": None, "champions": []}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"init": None, "current": None, "champions": []}
    data.setdefault("init", None)
    data.setdefault("current", None)
    data.setdefault("champions", [])
    return data


def _write_baseline_meta(bundle: Path, meta: dict[str, Any]) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / BASELINE_META).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def prepare_baseline(ws: Path, config: ExploreConfig | None = None) -> Path | None:
    """Build workspace/.baseline/ once from config.init_program.

    Layout:
      .baseline/<eval_program>  current generative baseline (read by export + agent);
                                rolled forward to the best champion each cycle.
      .baseline/eval/score.json its score
      .baseline/init/           immutable archive of the original init_program
      .baseline/baseline_meta.json  {init, current, champions[]} stats

    Returns the bundle dir, or None when no init_program is configured (cold
    start: baseline stays empty until the first champion is recorded). The bundle
    is hidden (dot-prefixed) so list_history's explore_b* glob skips it.
    """
    config = config or ExploreConfig.from_env()
    bundle = ws / ".baseline"
    if config.init_program is None or not config.init_program.is_file():
        # Cold start: no seed program — baseline is empty until first champion.
        return bundle if bundle.exists() else None

    code = config.init_program.read_text(encoding="utf-8")
    prog = bundle / config.eval_program
    prog.parent.mkdir(parents=True, exist_ok=True)
    if not prog.is_file():
        prog.write_text(code, encoding="utf-8")
    (bundle / "proposal.md").write_text(
        "# Baseline\n\nProvided starting solution (seed for every branch).\n",
        encoding="utf-8",
    )

    if config.eval_enabled() and not (bundle / "eval" / "score.json").is_file():
        evaluate_node(bundle, config)

    score, lat = _read_score(bundle / "eval" / "score.json")

    # Immutable init archive (written once).
    init_dir = bundle / INIT_DIRNAME
    if not init_dir.exists():
        init_prog = init_dir / config.eval_program
        init_prog.parent.mkdir(parents=True, exist_ok=True)
        init_prog.write_text(code, encoding="utf-8")
        src_score = bundle / "eval" / "score.json"
        if src_score.is_file():
            (init_dir / "eval").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_score, init_dir / "eval" / "score.json")

    meta = _load_baseline_meta(bundle)
    init_entry = {"score": score, "latency_ms": lat, "source": "init_program"}
    meta["init"] = init_entry
    if meta.get("current") is None:
        meta["current"] = {"cycle": -1, "score": score, "latency_ms": lat, "node_id": "init"}
    _write_baseline_meta(bundle, meta)
    return bundle


def find_champion(
    search_dir: Path, config: ExploreConfig | None = None
) -> tuple[float, float | None, Path, str] | None:
    """Best (highest combined_score) evaluated node under ``search_dir``.

    Scans ``explore_b*/attempt_*/eval/score.json``. Returns
    ``(score, latency_ms, node_dir, node_id)`` or None when nothing scored > 0.
    """
    config = config or ExploreConfig.from_env()
    best: tuple[float, float | None, Path, str] | None = None
    for sj in search_dir.glob("explore_b*/attempt_*/eval/score.json"):
        score, lat = _read_score(sj)
        if score <= 0.0:
            continue
        node_dir = sj.parent.parent
        if not (node_dir / config.eval_program).is_file():
            continue
        node_id = node_dir.name
        meta_file = node_dir / "meta.json"
        if meta_file.is_file():
            try:
                node_id = json.loads(meta_file.read_text(encoding="utf-8")).get("id", node_id)
            except (OSError, json.JSONDecodeError):
                pass
        if best is None or score > best[0]:
            best = (score, lat, node_dir, node_id)
    return best


def update_baseline_champion(
    ws: Path,
    search_dir: Path,
    *,
    cycle: int,
    config: ExploreConfig | None = None,
) -> dict[str, Any] | None:
    """Record this cycle's champion and roll the current baseline forward.

    Appends a champions[] entry every cycle. Overwrites the generative baseline
    (``.baseline/<eval_program>`` + eval/score.json) only when the champion beats
    the existing current score, so ``current`` always tracks the global best.
    Returns the appended champion entry, or None when nothing scored this cycle.
    """
    config = config or ExploreConfig.from_env()
    champ = find_champion(search_dir, config)
    if champ is None:
        return None
    score, lat, node_dir, node_id = champ

    bundle = ws / ".baseline"
    meta = _load_baseline_meta(bundle)

    try:
        source_path = str(node_dir.relative_to(ws))
    except ValueError:
        source_path = str(node_dir)
    entry = {
        "cycle": cycle,
        "score": score,
        "latency_ms": lat,
        "node_id": node_id,
        "source_path": source_path,
    }
    meta.setdefault("champions", []).append(entry)

    cur = meta.get("current") or {}
    cur_score = float(cur.get("score") or 0.0)
    if score > cur_score:
        prog = bundle / config.eval_program
        prog.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(node_dir / config.eval_program, prog)
        src_score = node_dir / "eval" / "score.json"
        if src_score.is_file():
            (bundle / "eval").mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_score, bundle / "eval" / "score.json")
        src_prop = node_dir / "proposal.md"
        if src_prop.is_file():
            shutil.copy2(src_prop, bundle / "proposal.md")
        meta["current"] = {
            "cycle": cycle,
            "score": score,
            "latency_ms": lat,
            "node_id": node_id,
            "source_path": source_path,
        }

    _write_baseline_meta(bundle, meta)
    return entry


def seed_branch_baseline(store: NodeStore, branch: int, bundle: Path | None) -> None:
    """Copy the baseline bundle into explore_b{branch}/baseline/ (idempotent).

    The copy is a plain sibling folder next to attempt_*, with no meta.json, so it
    never enters node numbering or history scans — it is purely readable context.
    """
    if bundle is None:
        return
    dst = store.branch_dir(branch) / BASELINE_DIRNAME
    if dst.exists():
        return
    store.branch_dir(branch).mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, dst)


def seed_branch_problem(store: NodeStore, branch: int, problem_file: Path) -> None:
    """Copy the canonical problem.txt into explore_b{branch}/ (idempotent), so the
    agent's cwd holds the spec locally and the prompt can point at ./problem.txt."""
    if not problem_file.is_file():
        return
    dst = store.branch_dir(branch) / "problem.txt"
    if dst.exists():
        return
    store.branch_dir(branch).mkdir(parents=True, exist_ok=True)
    shutil.copy2(problem_file, dst)
