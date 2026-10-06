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

"""Claude Code agent via claude-agent-sdk."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dream_rsi.agents.base import AgentResult, save_prompt, streaming_enabled


class ClaudeAgent:
    """
    Claude Code coding agent through Anthropic's agent SDK.

    Requires: pip install claude-agent-sdk
    Env: ANTHROPIC_API_KEY (or OpenRouter vars used by Claude Code)
    """

    name = "claude"

    def __init__(self, *, timeout_sec: float = 600.0) -> None:
        self.timeout_sec = timeout_sec

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        node_dir: Path,
        log_dir: Path,
        guide: dict[str, Any],
        run_cwd: Path | None = None,
        stage: str | None = None,
    ) -> AgentResult:
        del stage  # Gemini alone currently has stage-specific tool allowlists.
        save_prompt(log_dir, prompt)
        cwd = run_cwd or workspace
        try:
            final = asyncio.run(
                asyncio.wait_for(self._run_async(prompt, cwd, log_dir), self.timeout_sec)
            )
        except asyncio.TimeoutError:
            return AgentResult(status="failed", message=f"claude timeout after {self.timeout_sec}s")
        except ImportError:
            return AgentResult(status="failed", message="Install claude-agent-sdk: pip install claude-agent-sdk")
        except Exception as exc:
            return AgentResult(status="failed", message=str(exc))

        # The SDK reports a refused or errored turn in the ResultMessage instead
        # of raising; treating it as success would score an unmodified program
        # as a real candidate.
        if final is None:
            return AgentResult(status="failed", message="claude produced no ResultMessage")
        if final.is_error or final.subtype != "success":
            detail = (final.result or "").strip() or final.subtype
            return AgentResult(status="failed", message=f"claude turn failed: {detail[:500]}")

        denials = getattr(final, "permission_denials", None) or []
        note = ""
        if denials:
            names = ", ".join(sorted({str(getattr(d, "tool_name", d)) for d in denials}))
            note = f" ({len(denials)} denied: {names})"
        return AgentResult(message=f"claude finished in {final.num_turns} turns{note}")

    async def _run_async(self, prompt: str, workspace: Path, log_dir: Path) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        options = ClaudeAgentOptions(
            cwd=str(workspace.resolve()),
            allowed_tools=["Read", "Grep", "Glob", "Edit", "Write", "Bash"],
            # The SDK maps ``system_prompt=None`` to an empty system prompt, which
            # leaves the agent without tool guidance or its working directory.
            # The preset restores Claude Code's own system prompt.
            system_prompt={"type": "preset", "preset": "claude_code"},
        )
        messages = []
        final = None
        stream = streaming_enabled()
        async for msg in query(prompt=prompt, options=options):
            messages.append(msg)
            if stream:
                _emit_claude_message(msg)
            if isinstance(msg, ResultMessage):
                final = msg
        if stream:
            print("[claude] turn finished", flush=True)

        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "claude_messages.json").write_text(
            json.dumps(
                {
                    "num_messages": len(messages),
                    "final_result": getattr(final, "result", None) if final else None,
                    "subtype": getattr(final, "subtype", None) if final else None,
                    "is_error": getattr(final, "is_error", None) if final else None,
                    "num_turns": getattr(final, "num_turns", None) if final else None,
                    "total_cost_usd": getattr(final, "total_cost_usd", None) if final else None,
                    "permission_denials": [
                        str(d) for d in (getattr(final, "permission_denials", None) or [])
                    ]
                    if final
                    else [],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return final


def from_env() -> ClaudeAgent:
    return ClaudeAgent(timeout_sec=float(os.environ.get("CLAUDE_EXEC_TIMEOUT_SEC", "600")))


def _emit_claude_message(msg: object) -> None:
    """Best-effort live log for Claude SDK message objects.

    Text and tool calls arrive as blocks inside ``AssistantMessage.content``,
    so this walks the block list.
    """
    cls = type(msg).__name__
    if cls in ("AssistantMessage", "UserMessage"):
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            _emit_text(content)
        else:
            for block in content or []:
                _emit_block(block)
    elif cls == "ResultMessage":
        cost = getattr(msg, "total_cost_usd", None)
        cost_s = f" ${cost:.2f}" if isinstance(cost, (int, float)) else ""
        print(
            f"[claude] result: {msg.subtype} in {msg.num_turns} turns{cost_s}",
            flush=True,
        )
        for denial in getattr(msg, "permission_denials", None) or []:
            print(f"[claude] DENIED: {denial}", flush=True)


def _emit_text(text: str, prefix: str = "") -> None:
    if text.strip():
        print(f"[claude] {prefix}{text.strip()[:200]}", flush=True)


def _emit_block(block: object) -> None:
    cls = type(block).__name__
    if cls == "TextBlock":
        _emit_text(getattr(block, "text", ""))
    elif cls == "ThinkingBlock":
        _emit_text(getattr(block, "thinking", ""), prefix="(thinking) ")
    elif cls == "ToolUseBlock":
        name = getattr(block, "name", "?")
        # The first string input is usually the file path or command.
        args = getattr(block, "input", None) or {}
        hint = next((str(v) for v in args.values() if isinstance(v, str)), "")
        print(f"[claude] → {name} {hint[:120]}".rstrip(), flush=True)
    elif cls == "ToolResultBlock":
        if getattr(block, "is_error", False):
            detail = str(getattr(block, "content", ""))[:200].replace("\n", " ")
            print(f"[claude] ← tool ERROR: {detail}", flush=True)
