#!/usr/bin/env python3
"""Dream-RSI study tree: measured attempts (input -> one value) stored as a tree.

The tool only stores and summarizes. It never runs or measures anything.

Study directory layout:

  study.json            metric, goal, input name, workers, budget, patience
  problem.md            problem statement and the fixed measurement procedure
  guidance.md           optional cumulative direction guidance kept by the lead
  attempt_NNNN/
    meta.json           id, parent, status, closed, handed-off run, input hash, evidence
    proposal.md         the attempt's proposal, written by its worker
    <input>             the attempt's input: a parameters file, a program, a directory
    eval/score.json     {"metric", "value", "evidence"} after a successful measurement
    error.txt           what failed, after a failed measurement

attempt_0000 is the baseline. Each child of the baseline opens a direction; a
direction with all its descendants is a branch.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from string import Template

BASELINE = "0000"
GOALS = ("max", "min")
RESERVED = {"meta.json", "proposal.md", "eval", "error.txt"}
SKILL_MD = Path(__file__).resolve().parent.parent / "SKILL.md"
WORKER_START = "<!-- worker-prompt:start -->"
WORKER_END = "<!-- worker-prompt:end -->"


class StudyError(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def norm_id(raw) -> str:
    s = str(raw).strip()
    if s.startswith("attempt_"):
        s = s[len("attempt_"):]
    if not s.isdigit():
        raise StudyError(f"bad node id: {raw!r}")
    return f"{int(s):04d}"


def _hash_input(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_file():
        h.update(path.read_bytes())
        return h.hexdigest()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _title(proposal: Path) -> str:
    if not proposal.is_file():
        return ""
    for line in proposal.read_text(encoding="utf-8").splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:70]
    return ""


def _first_line(text: str | None, limit: int = 80) -> str:
    return (text or "").strip().splitlines()[0][:limit] if (text or "").strip() else ""


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.6g}"


def _delta(x: float | None, ref: float | None) -> str:
    return "" if x is None or ref is None else f"{x - ref:+.4g}"


class Study:
    def __init__(self, root) -> None:
        self.root = Path(root).resolve()
        if not (self.root / "study.json").is_file():
            raise StudyError(f"not a study directory: {self.root}")
        self.config = _read_json(self.root / "study.json")

    # ---- creation -------------------------------------------------------

    @classmethod
    def init(cls, root, *, problem, metric, goal, input_name, workers, budget, patience) -> "Study":
        root = Path(root).resolve()
        problem = Path(problem)
        if goal not in GOALS:
            raise StudyError(f"goal must be one of {GOALS}")
        if min(workers, budget, patience) < 1:
            raise StudyError("workers, budget and patience must be >= 1")
        rel = Path(input_name)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] in RESERVED:
            raise StudyError(f"input must be a relative path inside the node, not {input_name!r}")
        if not problem.is_file():
            raise StudyError(f"problem file not found: {problem}")
        if root.exists() and any(root.iterdir()):
            raise StudyError(f"directory is not empty: {root}")
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(problem, root / "problem.md")
        _write_json(root / "study.json", {
            "metric": metric,
            "goal": goal,
            "input": input_name,
            "workers": workers,
            "budget": budget,
            "patience": patience,
            "created_at": _now(),
        })
        study = cls(root)
        study._create_node(BASELINE, None, None)
        return study

    def node_dir(self, nid: str) -> Path:
        return self.root / f"attempt_{nid}"

    def _create_node(self, nid: str, parent: str | None, src_input: Path | None) -> Path:
        d = self.node_dir(nid)
        d.mkdir()
        if src_input is not None and src_input.exists():
            dst = d / self.config["input"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src_input.is_dir():
                shutil.copytree(src_input, dst)
            else:
                shutil.copy2(src_input, dst)
        _write_json(d / "meta.json", {
            "id": nid,
            "parent": parent,
            "status": "open",
            "created_at": _now(),
            "recorded_at": None,
            "record_seq": None,
            "evidence": None,
            "input_sha256": None,
            "run": None,
            "closed": None,
        })
        return d

    @contextmanager
    def _locked(self):
        with open(self.root / ".lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    # ---- reading --------------------------------------------------------

    def _meta(self, nid: str) -> dict:
        path = self.node_dir(nid) / "meta.json"
        if not path.is_file():
            raise StudyError(f"no such node: {nid}")
        return _read_json(path)

    def nodes(self) -> dict[str, dict]:
        """All nodes by id, with value, error and title loaded from their files."""
        out: dict[str, dict] = {}
        for d in sorted(self.root.glob("attempt_*")):
            if not (d / "meta.json").is_file():
                continue
            n = _read_json(d / "meta.json")
            n["dir"] = d
            n["value"] = None
            n["error"] = None
            if n["status"] == "done":
                n["value"] = _read_json(d / "eval" / "score.json")["value"]
            elif n["status"] == "failed":
                n["error"] = (d / "error.txt").read_text(encoding="utf-8").strip()
            n["title"] = _title(d / "proposal.md")
            out[n["id"]] = n
        return out

    def better(self, a: float, b: float) -> bool:
        return a > b if self.config["goal"] == "max" else a < b

    def best(self, nodes: list[dict]) -> dict | None:
        best = None
        for n in nodes:
            if n["value"] is not None and (best is None or self.better(n["value"], best["value"])):
                best = n
        return best

    @staticmethod
    def branch_of(nid: str, nodes: dict[str, dict]) -> str | None:
        """The direction (child of the baseline) whose branch contains ``nid``."""
        if nid == BASELINE:
            return None
        while nodes[nid]["parent"] != BASELINE:
            nid = nodes[nid]["parent"]
        return nid

    @staticmethod
    def closed_on_path(nid: str, nodes: dict[str, dict]) -> dict | None:
        while nid is not None:
            if nodes[nid]["closed"]:
                return nodes[nid]
            nid = nodes[nid]["parent"]
        return None

    # ---- writing --------------------------------------------------------

    def add(self, parent) -> tuple[str, Path]:
        parent = norm_id(parent)
        cfg = self.config
        with self._locked():
            nodes = self.nodes()
            if parent not in nodes:
                raise StudyError(f"no such node: {parent}")
            if nodes[parent]["status"] == "open":
                raise StudyError(f"parent {parent} is not measured yet")
            closed = self.closed_on_path(parent, nodes)
            if closed:
                raise StudyError(f"{closed['id']} is closed ({closed['closed']}); reopen it first")
            open_ids = [k for k, n in nodes.items() if n["status"] == "open"]
            if len(open_ids) >= cfg["workers"]:
                raise StudyError(f"all {cfg['workers']} worker slots are busy: {', '.join(open_ids)}")
            if len(nodes) - 1 >= cfg["budget"]:
                raise StudyError(f"budget of {cfg['budget']} attempts is used up")
            if parent != BASELINE:
                branch = self.branch_of(parent, nodes)
                busy = [k for k in open_ids if self.branch_of(k, nodes) == branch]
                if busy:
                    raise StudyError(f"branch {branch} already has an open attempt: {busy[0]}")
            last = max(int(p.name[len("attempt_"):]) for p in self.root.glob("attempt_*"))
            nid = f"{last + 1:04d}"
            d = self._create_node(nid, parent, nodes[parent]["dir"] / cfg["input"])
        return nid, d

    def record(self, nid, *, value=None, error=None, evidence: str, force: bool = False) -> None:
        nid = norm_id(nid)
        if (value is None) == (error is None):
            raise StudyError("record exactly one of --value or --error")
        if not evidence.strip():
            raise StudyError("--evidence is required: the command, log path or run id of the measurement")
        cfg = self.config
        with self._locked():
            nodes = self.nodes()
            if nid not in nodes:
                raise StudyError(f"no such node: {nid}")
            meta = self._meta(nid)
            if meta["status"] != "open" and not force:
                raise StudyError(f"{nid} is already recorded ({meta['status']})")
            d = self.node_dir(nid)
            inp = d / cfg["input"]
            score = d / "eval" / "score.json"
            if value is not None:
                if not math.isfinite(value):
                    raise StudyError("value must be a finite number")
                proposal = d / "proposal.md"
                if not proposal.is_file() or not proposal.read_text(encoding="utf-8").strip():
                    raise StudyError(f"{nid} has no proposal.md")
                if not inp.exists():
                    raise StudyError(f"{nid} has no input {cfg['input']}")
                digest = _hash_input(inp)
                for other in nodes.values():
                    if other["id"] != nid and other["status"] == "done" and other["input_sha256"] == digest:
                        raise StudyError(
                            f"input is identical to {other['id']}, already measured at "
                            f"{_fmt(other['value'])}; an attempt must not repeat a measured input"
                        )
                score.parent.mkdir(exist_ok=True)
                _write_json(score, {"metric": cfg["metric"], "value": value, "evidence": evidence})
                stale = d / "error.txt"
                meta["status"] = "done"
                meta["input_sha256"] = digest
            else:
                (d / "error.txt").write_text(f"{error.strip()}\n\nEvidence: {evidence}\n", encoding="utf-8")
                stale = score
                meta["status"] = "failed"
                meta["input_sha256"] = _hash_input(inp) if inp.exists() else None
            meta["evidence"] = evidence
            meta["recorded_at"] = _now()
            if meta["record_seq"] is None:
                meta["record_seq"] = 1 + max((n["record_seq"] or 0) for n in nodes.values())
            _write_json(d / "meta.json", meta)
            # Removed only after meta.json points at the new result, so readers never miss a file.
            stale.unlink(missing_ok=True)

    def handoff(self, nid, run: str) -> None:
        """Mark an open attempt's measurement as started and left to the lead to collect."""
        nid = norm_id(nid)
        if not run.strip():
            raise StudyError("--run is required: the run id and where its result will appear")
        with self._locked():
            meta = self._meta(nid)
            if meta["status"] != "open":
                raise StudyError(f"{nid} is already recorded ({meta['status']})")
            d = self.node_dir(nid)
            proposal = d / "proposal.md"
            if not proposal.is_file() or not proposal.read_text(encoding="utf-8").strip():
                raise StudyError(f"{nid} has no proposal.md")
            if not (d / self.config["input"]).exists():
                raise StudyError(f"{nid} has no input {self.config['input']}")
            meta["run"] = run
            _write_json(d / "meta.json", meta)

    def close(self, nid, reason: str) -> None:
        nid = norm_id(nid)
        if nid == BASELINE:
            raise StudyError("the baseline cannot be closed")
        if not reason.strip():
            raise StudyError("--reason is required")
        with self._locked():
            meta = self._meta(nid)
            if meta["status"] == "open":
                raise StudyError(f"{nid} is open; record its result first")
            meta["closed"] = reason
            _write_json(self.node_dir(nid) / "meta.json", meta)

    def reopen(self, nid) -> None:
        nid = norm_id(nid)
        with self._locked():
            meta = self._meta(nid)
            if not meta["closed"]:
                raise StudyError(f"{nid} is not closed")
            meta["closed"] = None
            _write_json(self.node_dir(nid) / "meta.json", meta)

    # ---- views ----------------------------------------------------------

    def _line(self, n: dict, nodes: dict[str, dict], best: dict | None) -> str:
        parent = nodes.get(n["parent"]) if n["parent"] else None
        if n["status"] == "open":
            result = f"… running: {_first_line(n['run'], 60)}" if n["run"] else "… open"
        elif n["status"] == "failed":
            result = f"✗ {_first_line(n['error'], 60)}"
        else:
            result = _fmt(n["value"])
            d = _delta(n["value"], parent["value"] if parent else None)
            if d:
                result += f" ({d})"
        parts = [n["id"], result]
        if n["title"]:
            parts.append(f"— {n['title']}")
        if best is not None and n["id"] == best["id"]:
            parts.append("★")
        if n["closed"]:
            parts.append(f"[closed: {n['closed']}]")
        return "  ".join(parts)

    def show(self) -> str:
        nodes = self.nodes()
        best = self.best(list(nodes.values()))
        children: dict[str | None, list[str]] = {}
        for k, n in nodes.items():
            children.setdefault(n["parent"], []).append(k)
        lines: list[str] = []

        def walk(nid: str, prefix: str, tail: bool, top: bool) -> None:
            head = "" if top else prefix + ("└─ " if tail else "├─ ")
            lines.append(head + self._line(nodes[nid], nodes, best))
            kids = sorted(children.get(nid, []))
            sub = "" if top else prefix + ("   " if tail else "│  ")
            for i, k in enumerate(kids):
                walk(k, sub, i == len(kids) - 1, False)

        walk(BASELINE, "", True, True)
        return "\n".join(lines)

    def summary(self) -> str:
        cfg = self.config
        nodes = self.nodes()
        base = nodes[BASELINE]
        best = self.best(list(nodes.values()))
        open_ids = [k for k, n in nodes.items() if n["status"] == "open"]
        out = [
            f"study: {self.root}",
            f"metric: {cfg['metric']} ({cfg['goal']}) | input: {cfg['input']} | "
            f"workers busy {len(open_ids)}/{cfg['workers']} | "
            f"budget used {len(nodes) - 1}/{cfg['budget']} | patience {cfg['patience']}",
            f"baseline {BASELINE}: " + (
                _fmt(base["value"]) if base["status"] == "done"
                else "open" if base["status"] == "open" else f"✗ {_first_line(base['error'])}"
            ),
        ]
        if best is not None:
            out.append(f"best: {best['id']} {_fmt(best['value'])} "
                       f"({_delta(best['value'], base['value']) or 'n/a'} vs baseline) — {best['title']}")
        branches: dict[str, list[dict]] = {}
        for k, n in nodes.items():
            if k != BASELINE:
                branches.setdefault(self.branch_of(k, nodes), []).append(n)
        out.append("")
        out.append("branches (trajectory in record order):" if branches else "branches: none yet")
        has_child = {n["parent"] for n in nodes.values()}
        extendable = []
        for root_id in sorted(branches):
            members = branches[root_id]
            measured = sorted((n for n in members if n["record_seq"] is not None),
                              key=lambda n: n["record_seq"])
            anchor = self.best(members)
            stale = len([n for n in measured if anchor is None or n["record_seq"] > anchor["record_seq"]])
            failed = [n for n in members if n["status"] == "failed"]
            busy = [n["id"] for n in members if n["status"] == "open"]
            unresolved = [n for n in failed if n["id"] not in has_child and not n["closed"]]
            closed = [n for n in members if n["closed"]]
            head = f"  {root_id} \"{nodes[root_id]['title']}\": "
            head += (f"best {anchor['id']} {_fmt(anchor['value'])} "
                     f"({_delta(anchor['value'], base['value']) or 'n/a'} vs baseline)"
                     if anchor else "no successful measurement")
            head += f" | {len(measured)} measured, {len(failed)} failed | {stale} since best"
            if stale >= cfg["patience"]:
                head += " [patience reached]"
            if busy:
                head += f" | open: {', '.join(busy)}"
            out.append(head)
            traj = ", ".join(f"{n['id']} {_fmt(n['value']) if n['status'] == 'done' else '✗'}"
                             for n in measured)
            out.append(f"     trajectory: {traj or '-'}")
            for n in unresolved:
                out.append(f"     unresolved failure: {n['id']} — {_first_line(n['error'])}")
            for n in closed:
                out.append(f"     closed: {n['id']} — {n['closed']}")
            if not busy and not nodes[root_id]["closed"]:
                extendable.append(root_id)
        out.append("")
        out.append("open attempts: " + ("" if open_ids else "none"))
        for k in open_ids:
            who = f"handed off: {_first_line(nodes[k]['run'])}" if nodes[k]["run"] else "with its worker"
            out.append(f"  {k} (parent {nodes[k]['parent']}) — {who}")
        out.append("can extend: baseline (new direction)"
                   + (f"; branches {', '.join(extendable)} (any measured, unclosed node in them)"
                      if extendable else ""))
        return "\n".join(out)

    def export(self, stream) -> None:
        nodes = self.nodes()
        base = nodes[BASELINE]["value"]
        w = csv.writer(stream)
        w.writerow(["id", "parent", "branch", "status", "value", "delta_vs_parent",
                    "delta_vs_baseline", "record_seq", "closed", "title", "error", "evidence"])
        for k, n in nodes.items():
            parent = nodes.get(n["parent"]) if n["parent"] else None
            w.writerow([
                k, n["parent"] or "", self.branch_of(k, nodes) or "", n["status"],
                "" if n["value"] is None else n["value"],
                _delta(n["value"], parent["value"] if parent else None),
                _delta(n["value"], base) if k != BASELINE else "",
                n["record_seq"] or "", n["closed"] or "", n["title"],
                _first_line(n["error"], 200), n["evidence"] or "",
            ])

    def brief(self, nid) -> str:
        nid = norm_id(nid)
        nodes = self.nodes()
        if nid not in nodes:
            raise StudyError(f"no such node: {nid}")
        n = nodes[nid]
        if n["status"] != "open":
            raise StudyError(f"{nid} is already recorded; brief only open attempts")
        if nid == BASELINE:
            raise StudyError("the baseline is measured by the lead, not briefed to a worker")
        text = SKILL_MD.read_text(encoding="utf-8")
        start, end = text.index(WORKER_START) + len(WORKER_START), text.index(WORKER_END)
        parent_dir = self.node_dir(n["parent"])
        if n["parent"] == BASELINE:
            assignment = (f"Your parent is the baseline `{parent_dir}`: your attempt opens a new "
                          f"direction.")
        else:
            root_dir = self.node_dir(self.branch_of(nid, nodes))
            assignment = (f"Your parent is `{parent_dir}`, in the branch opened by `{root_dir}`: "
                          f"your attempt continues that branch.")
        guidance = self.root / "guidance.md"
        guidance = guidance.read_text(encoding="utf-8").strip() if guidance.is_file() else ""
        return Template(text[start:end].strip()).safe_substitute(
            node_dir=str(self.node_dir(nid)),
            node_id=nid,
            study_dir=str(self.root),
            baseline_dir=str(self.node_dir(BASELINE)),
            problem_file=str(self.root / "problem.md"),
            input=self.config["input"],
            cli=f"python3 {Path(__file__).resolve()}",
            assignment=assignment,
            direction_guidance=f"## Cumulative exploration guidance\n\n{guidance}" if guidance else "",
        ) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsi_tree.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", help="study directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("init", help="create a study and its baseline node 0000")
    s.add_argument("--problem", required=True, help="problem statement with the measurement procedure")
    s.add_argument("--metric", required=True, help="name of the one measured value")
    s.add_argument("--goal", required=True, choices=GOALS)
    s.add_argument("--input", required=True, help="input path inside each node, e.g. params.json")
    s.add_argument("--workers", required=True, type=int, help="max attempts open at once (W)")
    s.add_argument("--budget", required=True, type=int, help="max attempts besides the baseline")
    s.add_argument("--patience", required=True, type=int,
                   help="attempts in a branch without beating its best before it is flagged")
    s = sub.add_parser("add", help="open an attempt under a measured parent; prints id and dir")
    s.add_argument("--parent", required=True)
    s = sub.add_parser("brief", help="print the worker prompt for an open attempt")
    s.add_argument("id")
    s = sub.add_parser("record", help="record a measured value or a failure")
    s.add_argument("id")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--value", type=float)
    g.add_argument("--error")
    s.add_argument("--evidence", required=True, help="command, log path or run id of the measurement")
    s.add_argument("--force", action="store_true", help="overwrite an existing record (lead only)")
    s = sub.add_parser("handoff", help="register a started measurement for the lead to collect")
    s.add_argument("id")
    s.add_argument("--run", required=True, help="run id and where its result will appear")
    s = sub.add_parser("close", help="stop extending a node and its subtree")
    s.add_argument("id")
    s.add_argument("--reason", required=True)
    s = sub.add_parser("reopen", help="undo close")
    s.add_argument("id")
    sub.add_parser("show", help="print the tree")
    sub.add_parser("summary", help="print branch trajectories and what can be extended")
    sub.add_parser("export", help="print all nodes as CSV")
    a = p.parse_args(argv)

    try:
        if a.cmd == "init":
            st = Study.init(a.study, problem=a.problem, metric=a.metric, goal=a.goal,
                            input_name=a.input, workers=a.workers, budget=a.budget,
                            patience=a.patience)
            print(f"{BASELINE} {st.node_dir(BASELINE)}")
            return 0
        st = Study(a.study)
        if a.cmd == "add":
            nid, d = st.add(a.parent)
            print(f"{nid} {d}")
        elif a.cmd == "brief":
            sys.stdout.write(st.brief(a.id))
        elif a.cmd == "record":
            st.record(a.id, value=a.value, error=a.error, evidence=a.evidence, force=a.force)
        elif a.cmd == "handoff":
            st.handoff(a.id, a.run)
        elif a.cmd == "close":
            st.close(a.id, a.reason)
        elif a.cmd == "reopen":
            st.reopen(a.id)
        elif a.cmd == "show":
            print(st.show())
        elif a.cmd == "summary":
            print(st.summary())
        elif a.cmd == "export":
            st.export(sys.stdout)
    except StudyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
