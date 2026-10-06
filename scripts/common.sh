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

# shellcheck shell=bash
# Shared settings and checks for the launch scripts. Source it, do not run it.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"

usage_common() {
  cat <<'EOF'

Tasks (default ROUNDS):
  sums_diffs, autocorrelation_third, circle_packing_26, circle_packing_32   10
  lasso_path                                                                5
  kb_l3_11_vgg16, kb_l1_40_layernorm, kb_l2_8_convdiv, kb_l2_43_convmax     9
  The kb_* tasks need a running KernelBench server (tasks/kernelbench/README.md).

Common settings (environment variables, defaults in brackets):
  AGENT         gemini | claude | codex [gemini]
  BRANCHES      parallel branches per round [10]
  REFINE        refinements per branch; a round evaluates BRANCHES*(REFINE+1) programs [10]
  MAX_WORKERS   agent runs at the same time [10]
  EVAL_TIMEOUT  seconds per program evaluation [task default]
  EXPLORE_SHELL 1 = the Gemini agent may run shell commands while exploring [0]
                (claude and codex always can)
  PYTHON        Python interpreter [python]
  DRY_RUN       1 = run all checks and print the commands without running them [0]

Agent settings such as GEMINI_MODEL and the API keys can also be set in .env
(see README.md and .env.example).
EOF
}

die() {
  echo "error: $*" >&2
  exit 2
}

warn() {
  echo "warning: $*" >&2
}

is_positive_int() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonnegative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

# Fail unless each named variable holds a positive integer.
require_positive() {
  local name
  for name in "$@"; do
    is_positive_int "${!name}" || die "$name must be a positive integer (got '${!name}')"
  done
}

# Fail unless each named variable is 0 or 1.
require_flag() {
  local name
  for name in "$@"; do
    case "${!name}" in
      0 | 1) ;;
      *) die "$name must be 0 or 1 (got '${!name}')" ;;
    esac
  done
}

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$PWD/$1" ;;
  esac
}

# Print a command, then run it unless DRY_RUN=1.
run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  [ "$DRY_RUN" = 1 ] || "$@"
}

# Export the variables of the .env file that dream_rsi reads (DRSI_ENV_FILE,
# else <repo>/.env), so the defaults and checks below see them. Variables
# already set in the shell win, as in dream_rsi.
load_env_file() {
  local found exports
  # The scripts run commands from $REPO_ROOT, so make the interpreter path absolute.
  found="$(command -v "$PYTHON")" || die "Python not found: $PYTHON (set PYTHON=/path/to/python)"
  PYTHON="$(abspath "$found")"
  exports="$(cd "$REPO_ROOT" && "$PYTHON" - <<'PY'
import os
import re
import shlex
import sys

if sys.version_info < (3, 10):
    sys.exit("Python >= 3.10 is required")
from dream_rsi.dotenv import load_env

before = set(os.environ)
load_env()
for key in sorted(set(os.environ) - before):
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        print(f"export {key}={shlex.quote(os.environ[key])}")
PY
)" || die "$PYTHON cannot load dream_rsi from $REPO_ROOT"
  eval "$exports"
}

# Task files and the paper's per-task defaults: rounds T (including E0) and
# the explore eval timeout in seconds.
select_task() {
  TASK="$1"
  case "$TASK" in
    sums_diffs) TASK_KIND=math ROUNDS_DEFAULT=10 EVAL_TIMEOUT_DEFAULT=240 ;;
    autocorrelation_third) TASK_KIND=math ROUNDS_DEFAULT=10 EVAL_TIMEOUT_DEFAULT=120 ;;
    circle_packing_26 | circle_packing_32) TASK_KIND=math ROUNDS_DEFAULT=10 EVAL_TIMEOUT_DEFAULT=600 ;;
    lasso_path) TASK_KIND=lasso ROUNDS_DEFAULT=5 EVAL_TIMEOUT_DEFAULT=600 ;;
    kb_l3_11_vgg16 | kb_l1_40_layernorm | kb_l2_8_convdiv | kb_l2_43_convmax)
      TASK_KIND=kernel ROUNDS_DEFAULT=9 EVAL_TIMEOUT_DEFAULT=3600 ;;
    *) die "unknown task '$TASK' (run with --help for the list)" ;;
  esac
  if [ "$TASK_KIND" = kernel ]; then
    TASK_DIR="$REPO_ROOT/tasks/kernelbench/$TASK"
    PROBLEM="$TASK_DIR/$TASK.triton"
    # The seed is the unmodified KernelBench problem (MIT), vendored separately.
    INIT_PROGRAM="$REPO_ROOT/third_party/kernelbench/$TASK/init_program.py"
  else
    TASK_DIR="$REPO_ROOT/tasks/$TASK"
    PROBLEM="$TASK_DIR/$TASK.txt"
    INIT_PROGRAM="$TASK_DIR/init_program.py"
    # The math and Lasso task files come from SimpleTES and are not shipped here.
    [ -f "$INIT_PROGRAM" ] || die "task files for $TASK not found; run: bash scripts/fetch_tasks.sh $TASK"
  fi
  EVALUATOR="$TASK_DIR/evaluator.py"
  EXPLORE_PROMPT="$REPO_ROOT/dream_rsi/prompts/explore.txt"
}

# Settings shared by all launch scripts. Defaults follow the paper.
load_settings() {
  AGENT="${AGENT:-gemini}"
  ROUNDS="${ROUNDS:-$ROUNDS_DEFAULT}"
  BRANCHES="${BRANCHES:-10}"
  REFINE="${REFINE:-10}"
  MAX_WORKERS="${MAX_WORKERS:-10}"
  EVAL_TIMEOUT="${EVAL_TIMEOUT:-$EVAL_TIMEOUT_DEFAULT}"
  GUIDANCE="${GUIDANCE:-0}"
  EXPLORE_SHELL="${EXPLORE_SHELL:-0}"
  DRY_RUN="${DRY_RUN:-0}"

  case "$AGENT" in
    gemini | claude | codex) ;;
    *) die "AGENT must be gemini, claude or codex (got '$AGENT')" ;;
  esac
  require_positive ROUNDS BRANCHES MAX_WORKERS EVAL_TIMEOUT
  is_nonnegative_int "$REFINE" || die "REFINE must be a non-negative integer (got '$REFINE')"
  require_flag GUIDANCE EXPLORE_SHELL DRY_RUN
  # shellcheck disable=SC2034  # read by the launch scripts
  if [ "$GUIDANCE" = 1 ]; then
    DIRECTION_MODE=global
  else
    DIRECTION_MODE=off
  fi
  # shellcheck disable=SC2034  # read by the launch scripts
  if [ "$EXPLORE_SHELL" = 1 ]; then
    EXPLORE_SHELL_FLAG=--explore-shell
  else
    EXPLORE_SHELL_FLAG=--no-explore-shell
  fi
}

# Agent settings of the paper's runs. Values already in the environment win.
export_agent_env() {
  export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
  if [ "$TASK_KIND" = kernel ]; then
    export GEMINI_TIMEOUT_SEC="${GEMINI_TIMEOUT_SEC:-3600}"
    export GEMINI_RETRIES="${GEMINI_RETRIES:-20}"
    export GEMINI_RETRY_BASE_SEC="${GEMINI_RETRY_BASE_SEC:-120}"
    export GPUKERNEL_VERBOSE="${GPUKERNEL_VERBOSE:-1}"
  else
    export GEMINI_TIMEOUT_SEC="${GEMINI_TIMEOUT_SEC:-12000}"
    export GEMINI_RETRIES="${GEMINI_RETRIES:-3}"
    export GEMINI_RETRY_BASE_SEC="${GEMINI_RETRY_BASE_SEC:-60}"
  fi
  export DRSI_AGENT_STREAM="${DRSI_AGENT_STREAM:-1}"
}

# Distinct Gemini API keys, read the same way as dream_rsi.agents.gemini.
gemini_key_count() {
  local keys="${GEMINI_API_KEYS:-}"
  if [ -z "${keys//[[:space:]]/}" ]; then
    keys="${GEMINI_API_KEY:-}"
    [ -n "${keys//[[:space:]]/}" ] || keys="${GOOGLE_API_KEY:-}"
  fi
  printf '%s\n' "$keys" | tr ',' '\n' \
    | awk '{ gsub(/^[ \t\r]+|[ \t\r]+$/, "") } $0 != "" && !seen[$0]++ { n++ } END { print n + 0 }'
}

# Check the task files, the evaluator's dependencies and the agent before
# anything starts. Candidate programs run in the evaluator's interpreter, so
# it also needs the libraries they commonly import.
preflight() {
  local f
  for f in "$PROBLEM" "$EVALUATOR" "$INIT_PROGRAM" "$EXPLORE_PROMPT"; do
    [ -f "$f" ] || die "missing file: $f"
  done

  local eval_python="${DRSI_EVAL_PYTHON:-$PYTHON}" modules extra
  case "$TASK_KIND" in
    math) modules="numpy, psutil, scipy" extra=tasks ;;
    lasso) modules="numpy, psutil, sklearn" extra=tasks ;;
    kernel) modules="requests, yaml" extra=kernel ;;
  esac
  "$eval_python" -c "import $modules" 2>/dev/null \
    || die "$eval_python cannot import $modules (pip install -e \".[$extra]\")"

  if [ "$TASK_KIND" = lasso ]; then
    command -v g++ >/dev/null 2>&1 || die "the Lasso evaluator needs g++"
    [ -f "$TASK_DIR/eigen/Eigen/Core" ] || [ -f /usr/include/eigen3/Eigen/Core ] \
      || die "Eigen headers not found (run: bash tasks/lasso_path/setup.sh)"
  fi

  if [ "$TASK_KIND" = kernel ]; then
    "$eval_python" - "$EVALUATOR" <<'PY' \
      || die "no KernelBench server is reachable; start one first (see tasks/kernelbench/README.md)"
import importlib.util
import sys

import requests

spec = importlib.util.spec_from_file_location("kernel_client", sys.argv[1])
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
servers, _ = client._read_server_endpoints()
for server in servers:
    url = f"http://{server['ip']}:{server['port']}/health"
    try:
        if requests.get(url, timeout=5).ok:
            print(f"KernelBench server: {url} ok")
            break
    except requests.RequestException:
        pass
else:
    sys.exit(f"no server answers on /health: {servers}")
PY
  fi

  case "$AGENT" in
    gemini)
      command -v "${GEMINI_BIN:-gemini}" >/dev/null 2>&1 \
        || die "gemini CLI not found: ${GEMINI_BIN:-gemini} (npm install -g @google/gemini-cli)"
      local keys
      keys="$(gemini_key_count)"
      [ "$keys" -gt 0 ] \
        || die "no Gemini API key: set GEMINI_API_KEYS (comma-separated) or GEMINI_API_KEY, in the environment or in .env"
      if [ "$keys" -lt "$MAX_WORKERS" ]; then
        warn "$keys Gemini API key(s) for MAX_WORKERS=$MAX_WORKERS parallel agents; expect rate-limit retries"
      fi
      ;;
    claude)
      "$PYTHON" -c "import claude_agent_sdk" 2>/dev/null \
        || die "AGENT=claude needs claude-agent-sdk (pip install -e \".[claude]\")"
      ;;
    codex)
      command -v "${CODEX_BIN:-codex}" >/dev/null 2>&1 || die "codex CLI not found: ${CODEX_BIN:-codex}"
      ;;
  esac
}

# One online round of WORKSPACE with the hand-designed starting policy.
run_fixed_round() {
  local cycle="$1" direction_mode="$2"
  run_cmd "$PYTHON" -m dream_rsi explore run -w "$WORKSPACE" -a "$AGENT" \
    --policy-source baseline --cycle "$cycle" --direction-mode "$direction_mode" \
    --branches "$BRANCHES" --refine "$REFINE" --max-workers "$MAX_WORKERS" \
    --eval-timeout "$EVAL_TIMEOUT" "$EXPLORE_SHELL_FLAG" \
    --problem "$PROBLEM" --evaluator "$EVALUATOR" --init-program "$INIT_PROGRAM" \
    --explore-prompt "$EXPLORE_PROMPT"
}

# Copy a finished E0 to WORKSPACE. The copy only gets its final name once it is
# complete, so an interrupted copy is never resumed as if it were E0.
copy_e0_workspace() {
  local partial="$WORKSPACE.partial"
  [ ! -e "$partial" ] || die "remove the incomplete copy $partial first"
  run_cmd mkdir -p "$(dirname "$WORKSPACE")"
  run_cmd cp -a "$1" "$partial"
  run_cmd mv "$partial" "$WORKSPACE"
}

# Rounds after E0 of a guided run read the direction memory. Before a resumed
# run explores round CYCLE, redo the previous round's summary if it failed.
resume_direction_memory() {
  local cycle="$1"
  [ "$GUIDANCE" = 1 ] && [ "$cycle" -gt 0 ] || return 0
  [ ! -f "$WORKSPACE/directions/current.json" ] || return 0
  run_cmd "$PYTHON" -m dream_rsi learning directions -w "$WORKSPACE" -a "$AGENT" --cycle "$((cycle - 1))" \
    || die "the direction summary of round $cycle failed; rerun with RESUME=1 to retry it"
}

# Index of the first round of a workspace that is not archived in history/.
next_round_index() {
  local next=0 dir index
  for dir in "$1"/history/iter[0-9][0-9][0-9]; do
    [ -d "$dir" ] || continue
    index=$((10#${dir##*/iter}))
    [ "$index" -lt "$next" ] || next=$((index + 1))
  done
  echo "$next"
}

# Every node of a round's raw_scores.json was evaluated and at least one scored
# above 0 (failed candidates score 0). Prints the counts and the best score.
check_round_complete() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys

nodes = json.load(open(sys.argv[1], encoding="utf-8")).get("nodes") or []
raws = [node.get("raw") or {} for node in nodes]
evaluated = sum(1 for raw in raws if raw.get("evaluated"))
scores = [raw.get("combined_score") for raw in raws]
positive = [score for score in scores if isinstance(score, (int, float)) and score > 0]
if not nodes or evaluated < len(nodes):
    sys.exit(f"{sys.argv[1]}: only {evaluated} of {len(nodes)} nodes were evaluated")
if not positive:
    sys.exit(f"{sys.argv[1]}: no node scored above 0; check the agent and evaluator logs")
print(f"{len(nodes)} nodes, {len(positive)} scored above 0, best score {max(positive):.6g}")
PY
}

# A shared E0 must be one finished, unarchived round of this task, run from the
# task's initial program on the same grid as the run that continues it.
check_e0_workspace() {
  local src="$1" f
  [ -d "$src" ] || die "E0_WORKSPACE not found: $src"
  for f in raw_scores.json history.json live_cycle_manifest.json problem.txt \
    .baseline/artifact/program.py .baseline/init/artifact/program.py; do
    [ -f "$src/$f" ] || die "E0_WORKSPACE has no $f; is it a finished E0 from scripts/run_e0.sh? ($src)"
  done
  compgen -G "$src/explore_b*" >/dev/null || die "E0_WORKSPACE has no explore_b* directories: $src"
  if [ -d "$src/history" ] && [ -n "$(ls -A "$src/history")" ]; then
    die "E0_WORKSPACE already has archived rounds in history/: $src"
  fi
  cmp -s "$src/problem.txt" "$PROBLEM" || die "E0_WORKSPACE was explored on another problem than $PROBLEM"
  cmp -s "$src/.baseline/init/artifact/program.py" "$INIT_PROGRAM" \
    || die "E0_WORKSPACE started from another initial program than $INIT_PROGRAM"
  cmp -s "$src/.baseline/artifact/program.py" "$INIT_PROGRAM" \
    || die "E0_WORKSPACE's baseline program differs from $INIT_PROGRAM"
  "$PYTHON" - "$src/live_cycle_manifest.json" "$BRANCHES" "$REFINE" <<'PY' || die "E0_WORKSPACE does not match this run"
import json
import sys

explore = json.load(open(sys.argv[1], encoding="utf-8")).get("explore") or {}
grid = (explore.get("max_width"), explore.get("max_refine"))
if grid != (int(sys.argv[2]), int(sys.argv[3])):
    sys.exit(
        f"E0 grid is BRANCHES={grid[0]} REFINE={grid[1]}, "
        f"this run uses BRANCHES={sys.argv[2]} REFINE={sys.argv[3]}"
    )
PY
  local status
  status="$(check_round_complete "$src/raw_scores.json")" || die "E0_WORKSPACE cannot be used: $src"
  echo "E0: $status ($src)"
}

# Node count and best score of every archived round.
report_rounds() {
  "$PYTHON" - "$1" <<'PY'
import json
import sys
from pathlib import Path

overall = None
for raw_path in sorted(Path(sys.argv[1]).glob("history/iter*/raw_scores.json")):
    nodes = json.loads(raw_path.read_text(encoding="utf-8")).get("nodes") or []
    scores = [
        raw["combined_score"]
        for raw in (node.get("raw") or {} for node in nodes)
        if raw.get("evaluated") and isinstance(raw.get("combined_score"), (int, float))
    ]
    best = max(scores, default=None)
    if best is not None and (overall is None or best > overall):
        overall = best
    shown = "-" if best is None else f"{best:.6g}"
    print(f"  {raw_path.parent.name}: {len(nodes)} nodes, best score {shown}")
if overall is not None:
    print(f"  best over all rounds: {overall:.6g}")
PY
}
