#!/usr/bin/env bash
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

# DREAM-RSI on one task: online rounds with the current exploration policy,
# each followed by offline policy learning on the accumulated replay worlds.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_dream_rsi.sh TASK

Runs DREAM-RSI on TASK for ROUNDS online rounds. After each round, offline
learning rewrites the exploration policy, replaying it on all rounds so far,
and promotes the best rewrite for the next round. The policy learned after
the last round is not used.

Settings (environment variables, defaults in brackets):
  WORKSPACE         output directory [workspace/dream_rsi/TASK, or
                    workspace/dream_rsi_guidance/TASK with GUIDANCE=1]
  E0_WORKSPACE      finished E0 from scripts/run_e0.sh to continue from;
                    without it, the first round is explored here
  RESUME            1 = continue an existing WORKSPACE from its first
                    unfinished round [0]
  ROUNDS            online rounds T, including E0 [task default]
  DISCOVER_ROUNDS   policy rewrites per learning step [3]
  DISCOVER_EVAL_TIMEOUT  seconds per policy replay [3600]
  GUIDANCE          1 = also summarize the explored directions after each
                    round and inject them into the next round's prompts
                    (DREAM-RSI + Guidance in Figure 6) [0]
EOF
  usage_common
}

case "${1:-}" in
  -h | --help) usage; exit 0 ;;
  "") usage >&2; exit 2 ;;
esac
[ $# -eq 1 ] || die "expected one TASK argument, got $#"

select_task "$1"
load_env_file
load_settings
DISCOVER_ROUNDS="${DISCOVER_ROUNDS:-3}"
DISCOVER_EVAL_TIMEOUT="${DISCOVER_EVAL_TIMEOUT:-3600}"
RESUME="${RESUME:-0}"
require_positive DISCOVER_ROUNDS DISCOVER_EVAL_TIMEOUT
require_flag RESUME
if [ "$GUIDANCE" = 1 ]; then
  method=dream_rsi_guidance
else
  method=dream_rsi
fi
WORKSPACE="$(abspath "${WORKSPACE:-$REPO_ROOT/workspace/$method/$TASK}")"

start_cycle=0
from_stage=explore
copy_e0=0
if [ "$RESUME" = 1 ] && [ -d "$WORKSPACE" ]; then
  start_cycle="$(next_round_index "$WORKSPACE")"
  [ "$start_cycle" -lt "$ROUNDS" ] || die "all $ROUNDS rounds in $WORKSPACE are finished"
  # A round's live_cycle_manifest.json is written when its exploration ends.
  if [ -f "$WORKSPACE/live_cycle_manifest.json" ]; then
    from_stage=learning
  fi
  echo "[dream-rsi] resuming $WORKSPACE at round $((start_cycle + 1)) ($from_stage)"
else
  [ ! -e "$WORKSPACE" ] || die "WORKSPACE already exists: $WORKSPACE (RESUME=1 continues it)"
  if [ -n "${E0_WORKSPACE:-}" ]; then
    E0_WORKSPACE="$(abspath "$E0_WORKSPACE")"
    check_e0_workspace "$E0_WORKSPACE"
    from_stage=learning
    copy_e0=1
  fi
fi
export_agent_env
preflight

echo "[dream-rsi] task=$TASK rounds=$ROUNDS grid=${BRANCHES}x$((REFINE + 1)) agent=$AGENT guidance=$GUIDANCE explore_shell=$EXPLORE_SHELL"
echo "[dream-rsi] workspace: $WORKSPACE"
cd "$REPO_ROOT"
# With a copied E0, cycle 0 starts at learning: the copy is E0.
if [ "$copy_e0" = 1 ]; then
  copy_e0_workspace "$E0_WORKSPACE"
elif [ "$from_stage" = explore ]; then
  resume_direction_memory "$start_cycle"
fi

rc=0
run_cmd "$PYTHON" -m dream_rsi evolve -w "$WORKSPACE" -a "$AGENT" \
  --cycles "$ROUNDS" --start-cycle "$start_cycle" --from "$from_stage" \
  --branches "$BRANCHES" --refine "$REFINE" --max-workers "$MAX_WORKERS" \
  --explore-eval-timeout "$EVAL_TIMEOUT" "$EXPLORE_SHELL_FLAG" --direction-mode "$DIRECTION_MODE" \
  --discover-rounds "$DISCOVER_ROUNDS" --eval-timeout "$DISCOVER_EVAL_TIMEOUT" \
  --target run_best --no-update-baseline-champion \
  --problem "$PROBLEM" --evaluator "$EVALUATOR" --init-program "$INIT_PROGRAM" \
  --explore-prompt "$EXPLORE_PROMPT" || rc=$?
[ "$DRY_RUN" = 0 ] || exit 0

# evolve exits 1 when its last learning step selects no policy. Once the last
# round is archived, that policy would only matter for a round after the end.
if [ "$rc" -eq 1 ] && [ -d "$WORKSPACE/history/$(printf 'iter%03d' $((ROUNDS - 1)))" ]; then
  echo "[dream-rsi] the final learning step selected no policy; all rounds are complete"
  rc=0
fi
if [ "$rc" -ne 0 ]; then
  echo "error: evolve exited with code $rc; rerun with RESUME=1 to continue" >&2
  exit "$rc"
fi
echo "[dream-rsi] finished $ROUNDS rounds in $WORKSPACE"
report_rounds "$WORKSPACE"
