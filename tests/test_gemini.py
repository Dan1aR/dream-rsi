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

"""Gemini stage-specific tool allowlist tests (no Gemini CLI/API required)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dream_rsi.agents.gemini import (
    EXPLORE_CORE_TOOLS,
    EXPLORE_NO_SHELL_POLICY,
    LEARNING_CORE_TOOLS,
    SHELL_TOOLS,
    GeminiAgent,
    core_tools_for_stage,
)
from dream_rsi.agents.gemini import _write_gemini_project_settings


def test_gemini_stage_allowlists_are_least_privilege():
    assert "run_shell_command" not in EXPLORE_CORE_TOOLS
    assert "run_shell_command" in LEARNING_CORE_TOOLS
    assert core_tools_for_stage(None) == EXPLORE_CORE_TOOLS
    assert core_tools_for_stage("explore") == EXPLORE_CORE_TOOLS
    assert core_tools_for_stage("learning") == LEARNING_CORE_TOOLS


def test_gemini_run_writes_the_stage_specific_project_policy(
    tmp_path: Path, monkeypatch
):
    fake_bin = tmp_path / "gemini"
    fake_bin.write_text("not executed\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = GeminiAgent(
        gemini_bin=str(fake_bin),
        api_keys=["test-key"],
        retries=0,
    )

    seen_commands: list[list[str]] = []

    def fake_spawn(command, *_args, **_kwargs):
        seen_commands.append(command)
        return 0, '{"type":"result","response":"done"}\n', "", False

    monkeypatch.setattr(agent, "_spawn", fake_spawn)

    explore_log = tmp_path / "explore-log"
    explore = agent.run(
        prompt="explore",
        workspace=workspace,
        node_dir=workspace,
        log_dir=explore_log,
        stage="explore",
    )
    assert explore.status == "ok"
    settings = json.loads((workspace / ".gemini/settings.json").read_text())
    assert settings["tools"]["core"] == EXPLORE_CORE_TOOLS
    explore_policy = json.loads((explore_log / "tool_policy.json").read_text())
    assert explore_policy["stage"] == "explore"
    assert explore_policy["core_tools"] == EXPLORE_CORE_TOOLS
    assert explore_policy["shell_enabled"] is False
    policy_path = Path(explore_policy["shell_policy"])
    assert policy_path == explore_log / "no_shell_policy.toml"
    assert policy_path.read_text(encoding="utf-8") == EXPLORE_NO_SHELL_POLICY
    assert ["--policy", str(policy_path)] == seen_commands[0][
        seen_commands[0].index("--policy") : seen_commands[0].index("--policy") + 2
    ]

    learning_log = tmp_path / "learning-log"
    learning = agent.run(
        prompt="learn",
        workspace=workspace,
        node_dir=workspace,
        log_dir=learning_log,
        stage="learning",
    )
    assert learning.status == "ok"
    settings = json.loads((workspace / ".gemini/settings.json").read_text())
    assert settings["tools"]["core"] == LEARNING_CORE_TOOLS
    learning_policy = json.loads((learning_log / "tool_policy.json").read_text())
    assert learning_policy["stage"] == "learning"
    assert learning_policy["shell_enabled"] is True
    assert learning_policy["shell_policy"] is None
    assert "--policy" not in seen_commands[1]


def test_gemini_explore_shell_drops_the_no_shell_policy(tmp_path: Path, monkeypatch):
    assert core_tools_for_stage("explore", explore_shell=True) == SHELL_TOOLS
    assert core_tools_for_stage("learning", explore_shell=False) == LEARNING_CORE_TOOLS

    fake_bin = tmp_path / "gemini"
    fake_bin.write_text("not executed\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent = GeminiAgent(
        gemini_bin=str(fake_bin),
        api_keys=["test-key"],
        retries=0,
        explore_shell=True,
    )
    seen_commands: list[list[str]] = []

    def fake_spawn(command, *_args, **_kwargs):
        seen_commands.append(command)
        return 0, '{"type":"result","response":"done"}\n', "", False

    monkeypatch.setattr(agent, "_spawn", fake_spawn)

    log_dir = tmp_path / "explore-log"
    result = agent.run(
        prompt="explore",
        workspace=workspace,
        node_dir=workspace,
        log_dir=log_dir,
        stage="explore",
    )
    assert result.status == "ok"
    settings = json.loads((workspace / ".gemini/settings.json").read_text())
    assert settings["tools"]["core"] == SHELL_TOOLS
    policy = json.loads((log_dir / "tool_policy.json").read_text())
    assert policy["shell_enabled"] is True
    assert policy["shell_policy"] is None
    assert not (log_dir / "no_shell_policy.toml").exists()
    assert "--policy" not in seen_commands[0]


def test_explore_shell_setting_from_env_and_override(monkeypatch):
    from dream_rsi.agents import load_agent

    monkeypatch.delenv("GEMINI_EXPLORE_SHELL", raising=False)
    assert load_agent("gemini").explore_shell is False
    assert load_agent("gemini", explore_shell=True).explore_shell is True

    monkeypatch.setenv("GEMINI_EXPLORE_SHELL", "1")
    assert load_agent("gemini").explore_shell is True
    assert load_agent("gemini", explore_shell=False).explore_shell is False


def test_explore_shell_cli_flags():
    from dream_rsi.cli.evolve import build_parser as evolve_parser
    from dream_rsi.cli.explore import build_parser as explore_parser

    base = ["run", "-w", "ws"]
    assert explore_parser().parse_args(base).explore_shell is None
    assert explore_parser().parse_args([*base, "--explore-shell"]).explore_shell is True
    assert explore_parser().parse_args([*base, "--no-explore-shell"]).explore_shell is False
    assert evolve_parser().parse_args(["-w", "ws"]).explore_shell is None
    assert evolve_parser().parse_args(["-w", "ws", "--explore-shell"]).explore_shell is True


def test_gemini_rejects_unknown_stage_without_spawning(tmp_path: Path):
    fake_bin = tmp_path / "gemini"
    fake_bin.write_text("not executed\n", encoding="utf-8")
    agent = GeminiAgent(gemini_bin=str(fake_bin), api_keys=["test-key"], retries=0)
    result = agent.run(
        prompt="bad stage",
        workspace=tmp_path,
        node_dir=tmp_path,
        log_dir=tmp_path / "log",
        stage="other",
    )
    assert result.status == "failed"
    assert "unknown Gemini agent stage" in result.message


def test_gemini_project_settings_remain_valid_with_concurrent_writers(tmp_path: Path):
    """Parallel direction extraction must not expose a partial settings file."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def write(index: int) -> None:
        _write_gemini_project_settings(
            workspace,
            max_turns=index + 1,
            core_tools=LEARNING_CORE_TOOLS,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write, range(32)))

    settings = json.loads((workspace / ".gemini/settings.json").read_text())
    assert settings["tools"]["core"] == LEARNING_CORE_TOOLS
    assert isinstance(settings["model"]["maxSessionTurns"], int)
