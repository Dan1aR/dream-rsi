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

"""Layout: explore_b0/attempt_00, attempt_01, ...  explore_b1/attempt_00, ..."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

NODE_DIRS = ("artifact", "logs", "eval", "agent_log")
ARTIFACT_DEFAULT = "artifact/main.md"
PROPOSAL_FILE = "proposal.md"
META_FILE = "meta.json"


@dataclass
class NodeMeta:
    id: str
    parent_id: Optional[str] = None
    branch: int = 0
    seq: int = 0
    cycle: int = 0
    intent: str = ""
    status: str = "draft"
    created_at: str = ""
    tags: list[str] = field(default_factory=list)
    artifact_relpath: str = ARTIFACT_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeMeta":
        return cls(
            id=data["id"],
            parent_id=data.get("parent_id"),
            branch=int(data.get("branch", 0)),
            seq=int(data.get("seq", 0)),
            cycle=int(data.get("cycle", 0)),
            intent=str(data.get("intent", "")),
            status=str(data.get("status", "draft")),
            created_at=str(data.get("created_at", "")),
            tags=list(data.get("tags") or []),
            artifact_relpath=str(data.get("artifact_relpath", ARTIFACT_DEFAULT)),
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def branch_name(branch: int) -> str:
    return f"explore_b{branch}"


def attempt_name(seq: int) -> str:
    """seq 1 → attempt_00 (explore), seq 2 → attempt_01 (refine), ...

    Zero-padded so lexical order matches execution order (attempt_10 after attempt_09).
    """
    return f"attempt_{seq - 1:02d}"


def node_id(branch: int, seq: int) -> str:
    return f"b{branch}_a{seq - 1}"


class NodeStore:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace.resolve()

    def branch_dir(self, branch: int) -> Path:
        return self.workspace / branch_name(branch)

    def create_node(
        self,
        *,
        branch: int,
        seq: int,
        parent_id: Optional[str] = None,
        cycle: int = 0,
        intent: str = "",
        label: str = "",
        tags: Optional[list[str]] = None,
        artifact_relpath: str = ARTIFACT_DEFAULT,
    ) -> Path:
        if parent_id is not None and not self.get_node_dir(parent_id).is_dir():
            raise FileNotFoundError(f"Parent not found: {parent_id}")

        nid = node_id(branch, seq)
        node_dir = self.branch_dir(branch) / attempt_name(seq)
        if node_dir.exists():
            raise FileExistsError(f"Already exists: {node_dir}")
        node_dir.mkdir(parents=True)
        for sub in NODE_DIRS:
            (node_dir / sub).mkdir()

        meta = NodeMeta(
            id=nid,
            parent_id=parent_id,
            branch=branch,
            seq=seq,
            cycle=cycle,
            intent=intent or label,
            status="draft",
            created_at=_utc_now(),
            tags=list(tags or []),
            artifact_relpath=artifact_relpath,
        )
        self._write_meta(node_dir, meta)
        return node_dir

    def get_node_dir(self, nid: str) -> Path:
        m = re.match(r"b(\d+)_a(\d+)", nid)
        if m:
            b, a = int(m.group(1)), int(m.group(2))
            for name in (attempt_name(a + 1), f"attempt_{a}"):
                p = self.branch_dir(b) / name
                if p.is_dir():
                    return p
        for mp in self.workspace.glob("explore_b*/attempt_*/meta.json"):
            meta = NodeMeta.from_dict(json.loads(mp.read_text(encoding="utf-8")))
            if meta.id == nid:
                return mp.parent
        return self.workspace / nid

    def read_meta(self, nid: str) -> NodeMeta:
        return NodeMeta.from_dict(
            json.loads((self.get_node_dir(nid) / META_FILE).read_text(encoding="utf-8"))
        )

    def list_branch(self, branch: int, *, cycle: Optional[int] = None) -> list[NodeMeta]:
        return self._scan_dir(self.branch_dir(branch), cycle=cycle)

    def list_history(self, *, cycle: Optional[int] = None) -> list[NodeMeta]:
        out: list[NodeMeta] = []
        for bdir in sorted(self.workspace.glob("explore_b*")):
            if bdir.is_dir():
                out.extend(self._scan_dir(bdir, cycle=cycle))
        return sorted(out, key=lambda m: (m.branch, m.seq))

    def list_nodes(self, *, cycle: Optional[int] = None) -> list[NodeMeta]:
        return self.list_history(cycle=cycle)

    def _scan_dir(self, root: Path, *, cycle: Optional[int] = None) -> list[NodeMeta]:
        if not root.is_dir():
            return []
        out: list[NodeMeta] = []
        for p in sorted(root.iterdir()):
            mp = p / META_FILE
            if p.is_dir() and mp.is_file():
                meta = NodeMeta.from_dict(json.loads(mp.read_text(encoding="utf-8")))
                if cycle is None or meta.cycle == cycle:
                    out.append(meta)
        return out

    def mark_proposed(self, nid: str, *, program_relpath: str) -> NodeMeta:
        node_dir = self.get_node_dir(nid)
        if not (node_dir / PROPOSAL_FILE).is_file():
            raise FileNotFoundError(f"Missing proposal.md in {nid}")
        if not (node_dir / program_relpath).is_file():
            raise FileNotFoundError(f"Missing {program_relpath} in {nid}")
        meta = self.read_meta(nid)
        meta.status = "proposed"
        self._write_meta(node_dir, meta)
        return meta

    def mark_done(self, nid: str) -> Path:
        node_dir = self.get_node_dir(nid)
        meta = self.read_meta(nid)
        meta.status = "done"
        self._write_meta(node_dir, meta)
        return node_dir

    def mark_failed(self, nid: str, reason: str = "") -> Path:
        node_dir = self.get_node_dir(nid)
        meta = self.read_meta(nid)
        meta.status = "failed"
        self._write_meta(node_dir, meta)
        if reason:
            (node_dir / "error.txt").write_text(reason, encoding="utf-8")
        return node_dir

    def reset_for_retry(self, nid: str) -> Path:
        """Drop prior artifacts but keep meta.json for a stable retry.

        Called before re-running the agent on a failed/draft/proposed node so
        stale proposal.md, program, eval scores, and logs do not confuse the
        next attempt.
        """
        node_dir = self.get_node_dir(nid)
        meta = self.read_meta(nid)
        meta.status = "draft"
        for child in node_dir.iterdir():
            if child.name == META_FILE:
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        for sub in NODE_DIRS:
            (node_dir / sub).mkdir(exist_ok=True)
        self._write_meta(node_dir, meta)
        return node_dir

    def export_graph(self) -> dict[str, Any]:
        nodes = self.list_history()
        return {
            "nodes": [n.to_dict() for n in nodes],
            "edges": [{"from": n.parent_id, "to": n.id} for n in nodes if n.parent_id],
        }

    def _write_meta(self, node_dir: Path, meta: NodeMeta) -> None:
        (node_dir / META_FILE).write_text(
            json.dumps(meta.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
