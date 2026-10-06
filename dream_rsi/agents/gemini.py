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

"""Gemini CLI coding agent."""

from __future__ import annotations

import json
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from dream_rsi.agents.base import (
    SHUTDOWN,
    AgentResult,
    kill_proc_group,
    register_proc,
    save_prompt,
    save_run_logs,
    streaming_enabled,
    unregister_proc,
)

# stderr signatures of transient API-level failures worth retrying.
# Matched against stderr ONLY: stdout is model-generated text (code, benchmark
# numbers like "503.96 ms") and would cause false positives.
TRANSIENT_PATTERNS = (
    "Invalid stream",
    "status: 429",
    "status: 503",
    "RESOURCE_EXHAUSTED",
    "returned an empty response",
    "UNAVAILABLE",
)


def _key_log_level() -> int:
    """0=off, 1=multi-key scheduling (default), 2=every acquire (verbose)."""
    v = os.environ.get("GEMINI_KEY_LOG", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return 0
    if v in ("verbose", "2", "all", "debug"):
        return 2
    return 1


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "…"
    return f"…{key[-4:]}"


def _key_log(msg: str) -> None:
    print(f"[gemini-key] {msg}", flush=True)


class KeyPool:
    """Thread-safe round-robin pool of API keys with per-key cooldown.

    Keys that just hit a quota/transient error are put on cooldown and skipped
    until they recover; if every key is cooling, the least-recently-penalized
    one is used anyway. NOTE: keys only multiply quota if they belong to
    different Google Cloud projects.
    """

    def __init__(self, keys: list[str], cooldown_sec: float = 60.0) -> None:
        # Strip and dedupe (order-preserving): duplicate keys would break
        # the exclude/penalize logic which compares by value.
        self.keys = list(dict.fromkeys(k.strip() for k in keys if k.strip()))
        self.cooldown_sec = cooldown_sec
        self._idx = 0
        self._cooling_until: dict[str, float] = {}
        self._lock = threading.Lock()
        if _key_log_level() and self.keys:
            src = "GEMINI_API_KEYS" if os.environ.get("GEMINI_API_KEYS", "").strip() else "single-key env"
            _key_log(f"pool ready: {len(self.keys)} key(s) from {src}, cooldown={self.cooldown_sec:.0f}s")

    def __len__(self) -> int:
        return len(self.keys)

    def key_label(self, key: str) -> str:
        idx = self.keys.index(key) + 1 if key in self.keys else 0
        return f"key#{idx}/{len(self.keys)} ({_mask_key(key)})"

    def acquire(self, exclude: str | None = None, *, reason: str = "run") -> str | None:
        """Pick the next usable key (round-robin), skipping cooling ones."""
        with self._lock:
            if not self.keys:
                return None
            now = time.time()
            n = len(self.keys)
            forced_cooling = False
            for off in range(n):
                key = self.keys[(self._idx + off) % n]
                if key == exclude and n > 1:
                    continue
                if self._cooling_until.get(key, 0.0) <= now:
                    self._idx = (self._idx + off + 1) % n
                    chosen = key
                    break
            else:
                # All keys cooling: take the one closest to recovery.
                chosen = min(self.keys, key=lambda k: self._cooling_until.get(k, 0.0))
                self._idx = (self.keys.index(chosen) + 1) % n
                forced_cooling = True

        level = _key_log_level()
        if level == 2 or (level == 1 and n > 1):
            parts = [
                f"acquire {self.key_label(chosen)}",
                f"reason={reason}",
                f"thread={threading.current_thread().name}",
            ]
            if exclude is not None and n > 1:
                parts.append(f"exclude={self.key_label(exclude)}")
            if forced_cooling:
                parts.append("all_keys_cooling")
            _key_log(" ".join(parts))
        return chosen

    def penalize(self, key: str, *, reason: str = "transient") -> None:
        """Put a key on cooldown after a 429/transient failure."""
        with self._lock:
            self._cooling_until[key] = time.time() + self.cooldown_sec
        if _key_log_level():
            _key_log(
                f"cooldown {self.key_label(key)} for {self.cooldown_sec:.0f}s ({reason})"
            )


def _keys_from_env() -> list[str]:
    """GEMINI_API_KEYS=key1,key2,... wins; fall back to the single-key vars."""
    multi = os.environ.get("GEMINI_API_KEYS", "").strip()
    if multi:
        return [k.strip() for k in multi.replace("\n", ",").split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()
    return [single] if single else []

# Allowlists passed to tools.core in .gemini/settings.json (gemini-cli >= 0.46).
# Only listed tools are enabled; this also blocks web_fetch, google_web_search,
# update_topic, etc.  Names match stream-json runtime tool_name values.
#
# By default exploration has no shell: the agent edits files and the evaluator
# runs the submitted program.  ``explore_shell=True`` (--explore-shell or
# GEMINI_EXPLORE_SHELL=1) also gives exploration the shell.  Learning always
# needs shell access to inspect and run replay-policy experiments in its
# isolated learning worktree.
FILE_TOOLS = [
    "read_file",
    "list_directory",
    "grep_search",
    "glob",
    "replace",
    "write_file",
]
SHELL_TOOLS = [
    *FILE_TOOLS,
    "run_shell_command",
]
EXPLORE_CORE_TOOLS = FILE_TOOLS.copy()
LEARNING_CORE_TOOLS = SHELL_TOOLS.copy()

# Gemini CLI 0.50's ``yolo`` approval rule can outrank the legacy
# ``tools.core`` setting.  Exploration uses yolo so ordinary artifact writes
# stay unattended, therefore a separate Policy Engine rule is required to make
# the no-shell contract real.  A command-line --policy is a user-tier policy;
# unlike a workspace .gemini/policies file it is honored by current Gemini CLI
# releases (workspace policies are currently disabled upstream).
EXPLORE_NO_SHELL_POLICY = """\
[[rule]]
toolName = ["run_shell_command", "ShellTool"]
decision = "deny"
priority = 999
modes = ["default", "autoEdit", "yolo", "plan"]
denyMessage = "Shell commands are disabled during DREAM-RSI exploration. Use the file tools and let the evaluator run the submitted artifact."
"""


def _write_stage_policy(
    log_dir: Path, *, stage: str | None, explore_shell: bool = False
) -> Path | None:
    """Write a per-agent Policy Engine rule when a stage must not use shell.

    ``tools.core`` remains useful as a model-facing allowlist, but is not a
    sufficient execution guard with modern Gemini CLI in ``--approval-mode
    yolo``.  The returned policy is supplied explicitly on the CLI command.
    """
    if explore_shell or (stage or "explore").strip().lower() != "explore":
        return None
    path = log_dir / "no_shell_policy.toml"
    path.write_text(EXPLORE_NO_SHELL_POLICY, encoding="utf-8")
    return path

# Direction extraction may issue several learning-stage Gemini runs at once.
# The CLI reads project settings from a shared workspace path, so protect the
# read/merge/write sequence within this process and replace the file atomically
# for the spawned CLI processes that read it.
_PROJECT_SETTINGS_LOCK = threading.Lock()


def core_tools_for_stage(stage: str | None, *, explore_shell: bool = False) -> list[str]:
    """Return Gemini's least-privilege tool allowlist for a DREAM-RSI stage.

    An omitted stage deliberately behaves as ``explore``, so a caller that
    does not label itself gets no shell unless ``explore_shell`` is set.
    """
    normalized = (stage or "explore").strip().lower()
    if normalized == "explore":
        return (SHELL_TOOLS if explore_shell else EXPLORE_CORE_TOOLS).copy()
    if normalized == "learning":
        return LEARNING_CORE_TOOLS.copy()
    raise ValueError(
        f"unknown Gemini agent stage {stage!r}; choose 'explore' or 'learning'"
    )


class GeminiAgent:
    name = "gemini"

    def __init__(
        self,
        *,
        gemini_bin: str = "gemini",
        model: str = "flash",
        timeout_sec: float = 600.0,
        max_turns: int | None = None,
        sandbox: bool = False,
        retries: int = 2,
        retry_base_sec: float = 20.0,
        api_keys: list[str] | None = None,
        extra_args: list[str] | None = None,
        explore_shell: bool = False,
    ) -> None:
        self.gemini_bin = gemini_bin
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_turns = max_turns
        self.sandbox = sandbox
        self.retries = retries
        self.retry_base_sec = retry_base_sec
        self.key_pool = KeyPool(api_keys if api_keys is not None else _keys_from_env())
        self.extra_args = extra_args or []
        self.explore_shell = explore_shell

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        log_dir: Path,
        run_cwd: Path | None = None,
        stage: str | None = None,
        **_,
    ) -> AgentResult:
        save_prompt(log_dir, prompt)

        bin_path = shutil.which(self.gemini_bin) if self.gemini_bin == "gemini" else self.gemini_bin
        if not bin_path or not Path(bin_path).is_file():
            return AgentResult(
                status="failed",
                message=f"Gemini CLI not found ({self.gemini_bin!r}). Install: npm install -g @google/gemini-cli",
            )
        if len(self.key_pool) == 0:
            return AgentResult(status="failed", message="Set GEMINI_API_KEY(S) or GOOGLE_API_KEY.")

        cwd = run_cwd or workspace
        # gemini-cli reads model.maxSessionTurns and tools.core from project settings
        # (<cwd>/.gemini/settings.json); there is no CLI flag for either.
        try:
            core_tools = core_tools_for_stage(stage, explore_shell=self.explore_shell)
        except ValueError as exc:
            return AgentResult(status="failed", message=str(exc))
        _write_gemini_project_settings(
            cwd,
            max_turns=self.max_turns,
            core_tools=core_tools,
        )
        stage_policy = _write_stage_policy(
            log_dir, stage=stage, explore_shell=self.explore_shell
        )
        (log_dir / "tool_policy.json").write_text(
            json.dumps(
                {
                    "stage": (stage or "explore").strip().lower(),
                    "core_tools": core_tools,
                    "shell_enabled": "run_shell_command" in core_tools,
                    "shell_policy": str(stage_policy) if stage_policy else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        # Feed the prompt via stdin so it works for any length (no ARG_MAX limit).
        # A piped (non-TTY) stdin makes the Gemini CLI run headless automatically.
        # --sandbox confines tool file ops to the workspace (cwd); writes outside fail.
        # stream-json gives the full turn-by-turn trajectory (saved to events.jsonl).
        cmd = [bin_path, "-m", self.model, "--approval-mode", "yolo", "--skip-trust"]
        cmd += ["--output-format", "stream-json"]
        if stage_policy is not None:
            cmd += ["--policy", str(stage_policy)]
        if self.sandbox:
            cmd.append("--sandbox")
        cmd += self.extra_args

        # Transient API failures (429 quota, "Invalid stream", empty response, ...)
        # are random server-side errors: retry with exponential backoff + jitter,
        # rotating to a different API key (cooling the one that just failed).
        # Timeouts are NOT retried (the agent already burned its time budget).
        attempts = 1 + max(0, self.retries)
        key: str | None = None
        for attempt in range(attempts):
            if SHUTDOWN.is_set():
                return AgentResult(status="failed", message="interrupted")
            key = self.key_pool.acquire(exclude=key, reason=f"attempt={attempt}")
            env = os.environ.copy()
            if key:
                env["GEMINI_API_KEY"] = key
                env.pop("GOOGLE_API_KEY", None)  # avoid precedence confusion

            returncode, raw_out, stderr, timed_out = self._spawn(
                cmd, prompt, env, cwd, stream=streaming_enabled()
            )

            # raw_out is a JSONL event stream: keep it verbatim (full trajectory)
            # and extract the assistant's final text + error events from it.
            final_text, error_text = _parse_stream(raw_out)
            (log_dir / "events.jsonl").write_text(raw_out, encoding="utf-8")
            save_run_logs(log_dir, stdout=final_text, stderr=stderr, command=cmd)

            if timed_out:
                return AgentResult(status="failed", message=f"timeout after {self.timeout_sec}s")

            # Exit 0 means the CLI finished; stderr may still mention earlier 429 retries.
            # Only treat the *current* stream as failed (empty/invalid), not stale stderr logs.
            if returncode == 0:
                stream_failed = any(
                    p in error_text
                    for p in ("returned an empty response", "Invalid stream")
                )
                if not stream_failed:
                    msg = (
                        "gemini finished"
                        if attempt == 0
                        else f"gemini finished (after {attempt} retries)"
                    )
                    return AgentResult(message=msg)
                if attempt >= attempts - 1:
                    tail = (error_text or final_text)[:500]
                    return AgentResult(
                        status="failed",
                        message=f"gemini empty/invalid stream after {attempt + 1} attempts: {tail}",
                    )
                if key:
                    self.key_pool.penalize(key, reason="empty/invalid stream")
                (log_dir / f"stderr_attempt{attempt}.txt").write_text(
                    stderr + ("\n--- error events ---\n" + error_text if error_text else ""),
                    encoding="utf-8",
                )
                if raw_out:
                    (log_dir / f"events_attempt{attempt}.jsonl").write_text(raw_out, encoding="utf-8")
                delay = (
                    random.uniform(1, 5)
                    if len(self.key_pool) > 1
                    else self.retry_base_sec * (2**attempt) + random.uniform(0, 10)
                )
                if SHUTDOWN.wait(delay):
                    return AgentResult(status="failed", message="interrupted")
                continue

            haystack = stderr + "\n" + error_text
            transient = any(p in haystack for p in TRANSIENT_PATTERNS)

            if transient and key:
                self.key_pool.penalize(key, reason="429/transient")

            if attempt < attempts - 1 and (transient or returncode != 0):
                # Keep the failed attempt's logs for post-mortem before retrying.
                (log_dir / f"stderr_attempt{attempt}.txt").write_text(
                    stderr + ("\n--- error events ---\n" + error_text if error_text else ""),
                    encoding="utf-8",
                )
                if raw_out:
                    (log_dir / f"events_attempt{attempt}.jsonl").write_text(raw_out, encoding="utf-8")
                # With spare keys the next attempt uses a fresh quota: short pause.
                delay = (
                    random.uniform(1, 5)
                    if len(self.key_pool) > 1
                    else self.retry_base_sec * (2**attempt) + random.uniform(0, 10)
                )
                # Interruptible sleep: aborts immediately on shutdown.
                if SHUTDOWN.wait(delay):
                    return AgentResult(status="failed", message="interrupted")
                continue

            tail = (error_text or stderr or final_text)[:500]
            return AgentResult(
                status="failed",
                message=f"gemini exit {returncode} after {attempt + 1} attempts: {tail}",
            )

        return AgentResult(status="failed", message="unreachable")

    def _spawn(
        self,
        cmd: list[str],
        prompt: str,
        env: dict,
        cwd: Path,
        *,
        stream: bool = False,
    ) -> tuple[int, str, str, bool]:
        """Run one gemini subprocess in its own process group.

        Returns (returncode, stdout, stderr, timed_out). When ``stream`` is
        true, tool events are printed live (stdout is still captured for logs).
        """
        if stream:
            return self._spawn_streaming(cmd, prompt, env, cwd)
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(cwd),
            start_new_session=True,
        )
        register_proc(proc)
        try:
            try:
                out, err = proc.communicate(input=prompt, timeout=self.timeout_sec)
                return proc.returncode, out or "", err or "", False
            except subprocess.TimeoutExpired:
                kill_proc_group(proc)
                out, err = proc.communicate()
                return proc.returncode, out or "", err or "", True
            except BaseException:
                kill_proc_group(proc)
                proc.wait()
                raise
        finally:
            unregister_proc(proc)

    def _spawn_streaming(
        self, cmd: list[str], prompt: str, env: dict, cwd: Path
    ) -> tuple[int, str, str, bool]:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(cwd),
            start_new_session=True,
            bufsize=1,
        )
        register_proc(proc)
        out_lines: list[str] = []
        err_lines: list[str] = []

        def _drain_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                err_lines.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        timed_out = False
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt)
            proc.stdin.close()
            assert proc.stdout is not None
            deadline = time.time() + self.timeout_sec
            while True:
                if proc.poll() is not None:
                    break
                if time.time() > deadline:
                    timed_out = True
                    kill_proc_group(proc)
                    break
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                out_lines.append(line)
                _emit_gemini_stream_line(line)
            proc.wait(timeout=5)
        except BaseException:
            kill_proc_group(proc)
            proc.wait()
            raise
        finally:
            unregister_proc(proc)
            stderr_thread.join(timeout=2)

        if proc.stdout:
            for line in proc.stdout:
                out_lines.append(line)
                _emit_gemini_stream_line(line)

        return proc.returncode or 0, "".join(out_lines), "".join(err_lines), timed_out


def _emit_gemini_stream_line(line: str) -> None:
    """Pretty-print one stream-json line to the terminal."""
    s = line.strip()
    if not s:
        return
    if not s.startswith("{"):
        print(s, flush=True)
        return
    try:
        ev = json.loads(s)
    except json.JSONDecodeError:
        return
    if not isinstance(ev, dict):
        return
    kind = ev.get("type")
    if kind == "tool_use":
        tool = ev.get("tool_name") or ev.get("name") or "tool"
        params = ev.get("parameters") or ev.get("args") or {}
        path = params.get("file_path") or params.get("path") or params.get("command") or ""
        suffix = f" {path}" if path else ""
        print(f"[gemini] → {tool}{suffix}", flush=True)
    elif kind == "tool_result":
        status = ev.get("status") or "ok"
        print(f"[gemini] ← tool {status}", flush=True)
    elif kind == "message" and ev.get("role") == "assistant":
        content = ev.get("content")
        if isinstance(content, str) and content and ev.get("delta"):
            print(content, end="", flush=True)
    elif kind == "error":
        print(f"[gemini] ERROR: {json.dumps(ev, ensure_ascii=False)[:300]}", flush=True)
    elif kind == "result":
        print("\n[gemini] turn finished", flush=True)


def _parse_stream(raw: str) -> tuple[str, str]:
    """Extract (assistant_final_text, error_event_text) from a stream-json stdout.

    Falls back to the raw text when the output is not JSONL (e.g. plain-text
    mode or a crash before the CLI could emit events).
    """
    texts: list[str] = []
    deltas: list[str] = []
    errors: list[str] = []
    saw_event = False
    for line in raw.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict) or "type" not in ev:
            continue
        saw_event = True
        kind = ev.get("type")
        if kind == "message" and ev.get("role") == "assistant":
            content = ev.get("content")
            if isinstance(content, str):
                (deltas if ev.get("delta") else texts).append(content)
        elif kind == "error":
            errors.append(json.dumps(ev, ensure_ascii=False))
        elif kind == "result":
            response = ev.get("response")
            if isinstance(response, str) and response:
                texts = [response]  # authoritative final answer
    if not saw_event:
        return raw, ""
    final = "\n".join(texts) if texts else "".join(deltas)
    return final, "\n".join(errors)


def _deep_merge_settings(base: dict, patch: dict) -> None:
    """Recursively merge patch into base (in place)."""
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_settings(base[key], value)
        else:
            base[key] = value


def _write_gemini_project_settings(
    cwd: Path,
    *,
    max_turns: int | None = None,
    core_tools: list[str] | None = None,
) -> None:
    """Merge agent run settings into <cwd>/.gemini/settings.json.

    Uses gemini-cli 0.46+ nested keys: model.maxSessionTurns, tools.core.
    Legacy flat keys (maxSessionTurns, coreTools) are removed when present.
    """
    settings_path = cwd / ".gemini" / "settings.json"
    with _PROJECT_SETTINGS_LOCK:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict = {}
        if settings_path.is_file():
            try:
                loaded = json.loads(settings_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (json.JSONDecodeError, OSError):
                data = {}
        data.pop("maxSessionTurns", None)
        data.pop("coreTools", None)
        patch: dict = {}
        if max_turns is not None:
            patch.setdefault("model", {})["maxSessionTurns"] = max_turns
        if core_tools is not None:
            patch.setdefault("tools", {})["core"] = core_tools
        _deep_merge_settings(data, patch)

        # Direct writes could expose truncated JSON to another concurrently
        # started Gemini CLI.  A same-directory replace makes every reader see
        # either the old complete settings or the new complete settings.
        temp_path = settings_path.with_name(
            f".{settings_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        try:
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_path.replace(settings_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()


def from_env() -> GeminiAgent:
    raw_turns = os.environ.get("GEMINI_MAX_TURNS", "").strip()
    return GeminiAgent(
        gemini_bin=os.environ.get("GEMINI_BIN", "gemini").strip() or "gemini",
        model=os.environ.get("GEMINI_MODEL", "flash").strip() or "flash",
        timeout_sec=float(os.environ.get("GEMINI_TIMEOUT_SEC", "600")),
        max_turns=int(raw_turns) if raw_turns else None,
        sandbox=os.environ.get("GEMINI_SANDBOX", "").strip().lower() in ("1", "true", "yes"),
        retries=int(os.environ.get("GEMINI_RETRIES", "2")),
        retry_base_sec=float(os.environ.get("GEMINI_RETRY_BASE_SEC", "20")),
        extra_args=shlex.split(os.environ.get("GEMINI_EXTRA_ARGS", "").strip()),
        explore_shell=os.environ.get("GEMINI_EXPLORE_SHELL", "").strip().lower()
        in ("1", "true", "yes"),
    )
