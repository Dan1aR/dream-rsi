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

"""``raw_scores.json`` IO: workspace export, load/write, conversion to :class:`Trace`."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAW_TRACE_VERSION = 1


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


@dataclass
class CellRaw:
    """Raw outcome for one grid cell (one attempt node or baseline)."""

    evaluated: bool = False
    combined_score: float | None = None
    n_valid: int | None = None
    n_total: int | None = None
    sol_times_ms: list[float | None] = field(default_factory=list)
    valids: list[bool | None] = field(default_factory=list)
    max_gaps: list[float | None] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CellRaw:
        return cls(
            evaluated=bool(data.get("evaluated", False)),
            combined_score=data.get("combined_score"),
            n_valid=data.get("n_valid"),
            n_total=data.get("n_total"),
            sol_times_ms=list(data.get("sol_times_ms") or []),
            valids=list(data.get("valids") or []),
            max_gaps=list(data.get("max_gaps") or []),
            error=data.get("error"),
        )


@dataclass
class RawNode:
    id: str
    branch: int
    attempt: int
    parent_id: str | None
    intent: str
    tags: list[str]
    path: str
    raw: CellRaw
    created_at: str | None = None
    status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "branch": self.branch,
            "attempt": self.attempt,
            "parent_id": self.parent_id,
            "intent": self.intent,
            "tags": list(self.tags),
            "path": self.path,
            "created_at": self.created_at,
            "status": self.status,
            "raw": self.raw.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawNode:
        return cls(
            id=str(data["id"]),
            branch=int(data["branch"]),
            attempt=int(data["attempt"]),
            parent_id=data.get("parent_id"),
            intent=str(data.get("intent") or ""),
            tags=list(data.get("tags") or []),
            path=str(data["path"]),
            created_at=data.get("created_at"),
            status=data.get("status"),
            raw=CellRaw.from_dict(data.get("raw") or {}),
        )


@dataclass
class RawTrace:
    version: int
    workspace: str
    exported_at: str
    n_problems: int
    baseline: CellRaw
    nodes: list[RawNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "workspace": self.workspace,
            "exported_at": self.exported_at,
            "n_problems": self.n_problems,
            "baseline": self.baseline.to_dict(),
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawTrace:
        return cls(
            version=int(data.get("version", RAW_TRACE_VERSION)),
            workspace=str(data.get("workspace", "")),
            exported_at=str(data.get("exported_at", "")),
            n_problems=int(data.get("n_problems", 0)),
            baseline=CellRaw.from_dict(data.get("baseline") or {}),
            nodes=[RawNode.from_dict(n) for n in data.get("nodes") or []],
        )


def parse_score_json(
    score_path: Path | None,
    *,
    n_problems: int | None = None,
) -> CellRaw:
    """Parse one ``eval/score.json`` into a :class:`CellRaw`."""
    if score_path is None or not score_path.is_file():
        n = n_problems or 0
        return CellRaw(
            evaluated=False,
            sol_times_ms=[None] * n,
            valids=[None] * n,
            max_gaps=[None] * n,
        )

    data = _load_json(score_path) or {}
    problems = [p for p in (data.get("problems") or []) if isinstance(p, dict)]
    max_idx = max((int(p["problem_idx"]) for p in problems if "problem_idx" in p), default=-1)
    n_total = int(data.get("n_total") or max(len(problems), max_idx + 1) or (n_problems or 0))

    times: list[float | None] = [None] * n_total
    valids: list[bool | None] = [None] * n_total
    gaps: list[float | None] = [None] * n_total
    for prob in problems:
        idx = prob.get("problem_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= n_total:
            continue
        t = prob.get("sol_time_ms")
        times[idx] = float(t) if isinstance(t, (int, float)) else None
        v = prob.get("valid")
        valids[idx] = bool(v) if v is not None else None
        g = prob.get("max_gap")
        gaps[idx] = float(g) if isinstance(g, (int, float)) else None

    raw_score = data.get("combined_score", data.get("score"))
    try:
        combined = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError):
        combined = None

    err = data.get("error")
    n_valid_raw = data.get("n_valid")

    return CellRaw(
        evaluated=True,
        combined_score=combined,
        n_valid=int(n_valid_raw) if n_valid_raw is not None else None,
        n_total=n_total,
        sol_times_ms=times,
        valids=valids,
        max_gaps=gaps,
        error=str(err) if err else None,
    )


def _has_agent_output(node_dir: Path) -> bool:
    """Heuristic: did the agent leave any artifact under this node?

    A freshly created ``draft`` scaffold has only empty subdirs. Any of these
    means the agent actually produced something (so live spent budget):
    a recorded ``error.txt``, a ``proposal.md``, or a non-empty ``artifact/``.
    """
    if (node_dir / "error.txt").is_file():
        return True
    if (node_dir / "proposal.md").is_file():
        return True
    artifact = node_dir / "artifact"
    return artifact.is_dir() and any(artifact.iterdir())


def _attempt_outcome(node_dir: Path, status: str | None) -> str | None:
    """Classify a node that has no ``eval/score.json`` (defense in depth).

    Stage 1 (authoritative): the runner explicitly recorded ``status="failed"``.
    Stage 2 (heuristic fallback): not marked failed, but the agent clearly ran
    and left artifacts — the process was likely killed (OOM/timeout/kill) before
    eval or ``mark_failed`` could run.

    Returns ``"failed"``, ``"zombie"``, or ``None`` (pure scaffold: never run).
    """
    if status == "failed":
        return "failed"
    if _has_agent_output(node_dir):
        return "zombie"
    return None


def _attempted_zero_cell(
    node_dir: Path, kind: str, *, n_problems: int | None = None
) -> CellRaw:
    """Materialize a 0-score *evaluated* cell for an attempted-but-unscored node.

    Live mode spent budget on it and recorded a 0 score. Replay must observe the
    same outcome, otherwise ``legal_actions`` would treat it like an unexplored
    ``draft`` scaffold and silently skip it, making replay appear to save budget
    that live actually spent. ``kind`` records which check matched.
    """
    n = n_problems or 0
    err: str | None = None
    err_file = node_dir / "error.txt"
    if err_file.is_file():
        try:
            err = err_file.read_text(encoding="utf-8").strip() or None
        except OSError:
            err = None
    if err is None:
        err = (
            "failed: no eval/score.json produced"
            if kind == "failed"
            else "attempted (artifacts present) but no eval/score.json; likely killed mid-run"
        )
    return CellRaw(
        evaluated=True,
        combined_score=0.0,
        n_valid=0,
        n_total=n or None,
        sol_times_ms=[None] * n,
        valids=[None] * n,
        max_gaps=[None] * n,
        error=err,
    )


def _pad_unevaluated(cells: list[CellRaw], n: int) -> None:
    for c in cells:
        if not c.sol_times_ms:
            c.sol_times_ms = [None] * n
        if not c.valids:
            c.valids = [None] * n
        if not c.max_gaps:
            c.max_gaps = [None] * n


def _history_records(ws: Path) -> list[dict[str, Any]]:
    history = _load_json(ws / "history.json")
    if history is not None:
        attempts = [rec for rec in (history.get("attempts") or []) if isinstance(rec, dict)]
        if attempts:
            return attempts

    from dream_rsi.exploration.store import NodeStore

    store = NodeStore(ws)
    return [
        {
            "id": m.id,
            "branch": m.branch,
            "seq": m.seq,
            "path": str(store.get_node_dir(m.id)),
            "parent_id": m.parent_id,
            "intent": m.intent,
            "cycle": m.cycle,
            "tags": list(m.tags),
        }
        for m in store.list_history()
    ]


def export_raw_scores(workspace: str | Path, *, n_problems: int | None = None) -> RawTrace:
    """Build a :class:`RawTrace` from an exploration workspace on disk."""
    ws = Path(workspace).resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"workspace not found: {ws}")

    baseline = parse_score_json(ws / ".baseline" / "eval" / "score.json", n_problems=n_problems)

    nodes: list[RawNode] = []
    seen_ids: set[str] = set()
    for rec in _history_records(ws):
        nid = str(rec["id"])
        if nid in seen_ids:
            continue
        seen_ids.add(nid)
        node_dir = Path(rec["path"])
        seq = int(rec.get("seq", 0))
        meta = _load_json(node_dir / "meta.json") or {}
        status = meta.get("status")
        cell = parse_score_json(node_dir / "eval" / "score.json", n_problems=n_problems)
        if not cell.evaluated:
            kind = _attempt_outcome(node_dir, status)
            if kind is not None:
                cell = _attempted_zero_cell(node_dir, kind, n_problems=n_problems)
        nodes.append(
            RawNode(
                id=str(rec["id"]),
                branch=int(rec["branch"]),
                attempt=max(0, seq - 1),
                parent_id=rec.get("parent_id"),
                intent=str(rec.get("intent") or ""),
                tags=list(rec.get("tags") or []),
                path=str(node_dir),
                created_at=meta.get("created_at"),
                status=status,
                raw=cell,
            )
        )

    all_cells = [baseline, *(n.raw for n in nodes)]
    n_eff = n_problems or max((len(c.sol_times_ms) for c in all_cells), default=0)
    _pad_unevaluated(all_cells, n_eff)

    return RawTrace(
        version=RAW_TRACE_VERSION,
        workspace=str(ws),
        exported_at=datetime.now(timezone.utc).isoformat(),
        n_problems=n_eff,
        baseline=baseline,
        nodes=nodes,
    )


def write_raw_scores(trace: RawTrace, out_path: str | Path) -> Path:
    out = Path(out_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(trace.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_raw_scores(path: str | Path) -> RawTrace:
    p = Path(path).resolve()
    data = _load_json(p)
    if data is None:
        raise FileNotFoundError(f"raw trace not found or invalid: {p}")
    return RawTrace.from_dict(data)


def _fail_class_from_raw(raw: CellRaw) -> str:
    if not raw.evaluated:
        return "no_score"
    if raw.error is None and raw.n_valid is not None and raw.n_total is not None:
        if raw.n_valid == raw.n_total and raw.n_valid > 0:
            return "ok"
    err = raw.error or ""
    if "compilation failed" in err or "compile" in err.lower():
        if "unsupported option" in err or "fopenmp" in err:
            return "compile_env"
        return "compile_other"
    if raw.error is None:
        return "ok"
    return "runtime"


def raw_trace_to_trace(raw: RawTrace) -> "Trace":
    from dream_rsi.env.trace.types import Node, Trace

    baseline_score = raw.baseline.combined_score if raw.baseline.combined_score is not None else 0.0

    nodes: list[Node] = []
    seen_ids: set[str] = set()
    for rn in raw.nodes:
        if rn.id in seen_ids:
            continue
        seen_ids.add(rn.id)
        cell = rn.raw
        score = cell.combined_score if cell.combined_score is not None else 0.0
        valid = (
            cell.evaluated
            and cell.error is None
            and cell.n_valid is not None
            and cell.n_total is not None
            and cell.n_valid == cell.n_total
            and cell.n_valid > 0
        )
        nodes.append(
            Node(
                id=rn.id,
                branch=rn.branch,
                seq=rn.attempt + 1,
                parent_id=rn.parent_id,
                intent=rn.intent,
                tags=list(rn.tags),
                path=Path(rn.path),
                score=score,
                valid=valid,
                fail_class=_fail_class_from_raw(cell),
                error=cell.error,
                evaluated=cell.evaluated,
                n_valid=cell.n_valid,
                n_total=cell.n_total,
                sol_times_ms=list(cell.sol_times_ms),
                valids=list(cell.valids),
                max_gaps=list(cell.max_gaps),
                created_at=rn.created_at,
                status=rn.status,
            )
        )

    by_id = {n.id: n for n in nodes}
    children: dict[str | None, list[str]] = {}
    for n in nodes:
        children.setdefault(n.parent_id, []).append(n.id)

    for n in nodes:
        n.delta_vs_baseline = n.score - baseline_score
        if n.parent_id and n.parent_id in by_id:
            n.parent_score = by_id[n.parent_id].score
            n.delta_vs_parent = n.score - n.parent_score

    return Trace(
        nodes=nodes,
        baseline_score=baseline_score,
        by_id=by_id,
        children=children,
    )


def trace_from_workspace(workspace: str | Path, *, n_problems: int | None = None) -> "Trace":
    return raw_trace_to_trace(export_raw_scores(workspace, n_problems=n_problems))


def trace_from_raw_file(path: str | Path) -> "Trace":
    return raw_trace_to_trace(load_raw_scores(path))
