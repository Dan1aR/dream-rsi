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

"""raw_scores.json export and roundtrip."""

from __future__ import annotations

import json

import pytest

from dream_rsi.env.trace.raw import (
    export_raw_scores,
    load_raw_scores,
    parse_score_json,
    raw_trace_to_trace,
    trace_from_raw_file,
    trace_from_workspace,
    write_raw_scores,
)


def test_parse_score_json(tmp_path):
    missing = parse_score_json(tmp_path / "missing.json", n_problems=2)
    assert not missing.evaluated and len(missing.sol_times_ms) == 2

    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps({"combined_score": 0.88, "n_valid": 3, "n_total": 3}))
    parsed = parse_score_json(score_path)
    assert parsed.evaluated and parsed.combined_score == pytest.approx(0.88)


def test_raw_scores_roundtrip(mini_raw_trace_file, tmp_path):
    loaded = load_raw_scores(mini_raw_trace_file)
    out = tmp_path / "out.json"
    write_raw_scores(loaded, out)
    again = load_raw_scores(out)
    assert again.baseline.combined_score == loaded.baseline.combined_score
    assert len(again.nodes) == len(loaded.nodes)


def test_trace_from_workspace(mini_workspace):
    trace = trace_from_workspace(mini_workspace)
    assert len(trace.nodes) == 2
    assert trace.best_score == pytest.approx(0.5)


def test_export_dedupes_history(mini_workspace):
    history = json.loads((mini_workspace / "history.json").read_text(encoding="utf-8"))
    history["attempts"].append(dict(history["attempts"][0]))
    (mini_workspace / "history.json").write_text(json.dumps(history))
    trace = raw_trace_to_trace(export_raw_scores(mini_workspace))
    assert len(trace.nodes) == 2
