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

"""Online explore: per-node agent/eval primitives for :class:`dream_rsi.env.TraceEnv`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dream_rsi.exploration.config import ExploreConfig
from dream_rsi.exploration.evaluate import evaluate_node
from dream_rsi.exploration.prompt import (
    explore_prompt,
    seed_branch_baseline,
    seed_branch_problem,
)
from dream_rsi.exploration.store import (
    META_FILE,
    NodeMeta,
    NodeStore,
    attempt_name,
    node_id as make_node_id,
)


def sync(workspace: Path) -> None:
    """Aggregate every node's meta into <workspace>/history.json."""
    store = NodeStore(workspace)
    attempts = [
        {
            "id": m.id,
            "branch": m.branch,
            "seq": m.seq,
            "path": str(store.get_node_dir(m.id)),
            "parent_id": m.parent_id,
            "intent": m.intent,
            "cycle": m.cycle,
            "tags": m.tags,
        }
        for m in store.list_history()
    ]
    ws = workspace.resolve()
    (ws / "history.json").write_text(
        json.dumps({"count": len(attempts), "attempts": attempts}, indent=2),
        encoding="utf-8",
    )


def _resolve_node(
    store: NodeStore,
    *,
    branch: int,
    seq: int,
    parent_id: str | None,
    intent: str,
    tags: list[str],
    cycle: int,
) -> tuple[Path, str, str]:
    """Return (node_dir, node_id, mode) where mode is new | retry | skip."""
    nid = make_node_id(branch, seq)
    node_dir = store.branch_dir(branch) / attempt_name(seq)

    if node_dir.is_dir() and (node_dir / "meta.json").is_file():
        if store.read_meta(nid).status == "done":
            return node_dir, nid, "skip"
        return node_dir, nid, "retry"

    node_dir = store.create_node(
        branch=branch,
        seq=seq,
        parent_id=parent_id,
        cycle=cycle,
        intent=intent,
        label=intent or f"b{branch}_a{seq - 1}",
        tags=tags,
    )
    return node_dir, nid, "new"


def _rogue_branch_paths(branch_dir: Path, node_dir: Path) -> list[str]:
    rogues: list[str] = []
    for p in branch_dir.iterdir():
        if p == node_dir or p.name in ("baseline", "problem.txt", "history"):
            continue
        if not p.name.startswith("attempt_"):
            continue
        if p.is_dir() and (p / "meta.json").is_file():
            continue
        rogues.append(p.name + ("/" if p.is_dir() else ""))
    return sorted(rogues)


def _restore_meta_if_tampered(node_dir: Path, nid: str, good_meta: bytes) -> None:
    """Restore meta.json if the agent overwrote/corrupted it during its run.

    The prompt tells the agent to write only proposal.md + the eval program, but a
    misbehaving agent can overwrite this node's meta.json (e.g. with a title/summary
    blob). That drops the structured fields the workflow relies on, so a later
    ``read_meta`` (mark_failed/mark_done, scans, retries) would raise KeyError. We
    snapshot the known-good meta before the agent runs and roll it back here, before
    any ``mark_*`` call reads it.
    """
    mp = node_dir / META_FILE
    try:
        meta = NodeMeta.from_dict(json.loads(mp.read_text(encoding="utf-8")))
        if meta.id == nid:
            return
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    print(f"[runner] agent tampered with meta.json for {nid}; restoring", flush=True)
    mp.write_bytes(good_meta)


def _validate_agent_outputs(node_dir: Path, config: ExploreConfig) -> str | None:
    proposal = node_dir / "proposal.md"
    if not proposal.is_file():
        return "missing proposal.md"
    if not proposal.read_text(encoding="utf-8").strip():
        return "proposal.md is empty"

    program = node_dir / config.eval_program
    if not program.is_file():
        return f"missing {config.eval_program}"
    return None


def _run_one_node(
    store: NodeStore,
    agent: Any,
    *,
    branch: int,
    seq: int,
    parent_id: str | None,
    intent: str,
    tags: list[str],
    ws: Path,
    cycle: int,
    config: ExploreConfig,
    baseline_bundle: Path | None,
    direction_context: str | None = None,
) -> tuple[str, bool]:
    """Run one node end-to-end. Thread-safe: only writes this node's directory."""
    seed_branch_baseline(store, branch, baseline_bundle)
    seed_branch_problem(store, branch, ws / "problem.txt")

    node_dir, nid, mode = _resolve_node(
        store,
        branch=branch,
        seq=seq,
        parent_id=parent_id,
        intent=intent,
        tags=tags,
        cycle=cycle,
    )
    if mode == "skip":
        return nid, True
    if mode == "retry":
        node_dir = store.reset_for_retry(nid)

    good_meta = (node_dir / META_FILE).read_bytes()

    branch_dir = store.branch_dir(branch)
    ar = agent.run(
        prompt=explore_prompt(node_dir, config, direction_context=direction_context),
        workspace=ws,
        node_dir=node_dir,
        log_dir=node_dir / "agent_log" / agent.name,
        guide={},
        run_cwd=branch_dir,
        stage="explore",
    )
    _restore_meta_if_tampered(node_dir, nid, good_meta)
    if ar.status != "ok":
        store.mark_failed(nid, ar.message)
        return nid, False

    rogues = _rogue_branch_paths(branch_dir, node_dir)
    if rogues:
        store.mark_failed(nid, f"agent created paths outside node: {', '.join(rogues)}")
        return nid, False

    output_err = _validate_agent_outputs(node_dir, config)
    if output_err:
        store.mark_failed(nid, output_err)
        return nid, False

    metrics = evaluate_node(node_dir, config)
    if metrics and metrics.get("error"):
        store.mark_failed(nid, str(metrics["error"]))
        return nid, False

    try:
        store.mark_proposed(nid, program_relpath=config.eval_program)
    except FileNotFoundError as exc:
        store.mark_failed(nid, str(exc))
        return nid, False

    store.mark_done(nid)
    return nid, True
