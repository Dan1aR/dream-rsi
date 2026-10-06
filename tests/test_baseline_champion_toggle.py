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

"""Regression tests for fixed-baseline self-evolution ablations."""

from __future__ import annotations

from pathlib import Path

import pytest

import dream_rsi.orchestrate as orchestrate
from dream_rsi.cli.evolve import build_parser


def test_cli_baseline_champion_toggle_defaults_off_and_can_enable() -> None:
    parser = build_parser()
    # The paper starts every cycle from the task's initial program.
    assert (
        parser.parse_args(["-w", "workspace/demo"]).update_baseline_champion is False
    )
    assert (
        parser.parse_args(
            ["-w", "workspace/demo", "--update-baseline-champion"]
        ).update_baseline_champion
        is True
    )
    assert orchestrate.PipelineConfig(workspace=Path("demo")).update_baseline_champion is False


@pytest.mark.parametrize("enabled", [False, True])
def test_run_cycle_honors_baseline_champion_toggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    ws = tmp_path / "workspace"
    archived = tmp_path / "archive"
    calls: list[tuple[Path, Path, int]] = []

    monkeypatch.setattr(
        orchestrate, "_run_explore_stage", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        orchestrate,
        "_run_learning_stage",
        lambda *args, **kwargs: (tmp_path / "learning", {}, None),
    )
    monkeypatch.setattr(
        orchestrate,
        "_archive_and_reset_workspace",
        lambda *args, **kwargs: str(archived),
    )

    def fake_update(workspace: Path, search_dir: Path, *, cycle: int, config: object):
        calls.append((workspace, search_dir, cycle))
        return {"node_id": "b0_a0", "score": 1.0}

    monkeypatch.setattr(orchestrate, "update_baseline_champion", fake_update)
    cfg = orchestrate.PipelineConfig(
        workspace=ws,
        skip_promote_method=True,
        update_baseline_champion=enabled,
    )

    row = orchestrate.run_cycle(cfg, cycle=0, start_stage="explore")

    assert bool(calls) is enabled
    assert row["baseline_champion_updated"] is enabled
    if enabled:
        assert calls == [(ws.resolve(), archived, 0)]
