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

# E0: the first online round, explored once with the hand-designed starting
# policy and shared by DREAM-RSI and the Recursive Fixed Exploration baseline.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_e0.sh TASK

Explores the first online round E0 of TASK with the hand-designed starting
policy and leaves it unarchived, so that both methods can continue from the
same E0 (the paper's protocol):

  bash scripts/run_e0.sh TASK
  E0_WORKSPACE=workspace/e0/TASK bash scripts/run_dream_rsi.sh TASK
  E0_WORKSPACE=workspace/e0/TASK bash scripts/run_baseline.sh TASK

Settings (environment variables, defaults in brackets):
  WORKSPACE     output directory [workspace/e0/TASK]
  RESUME        1 = finish an interrupted E0 in an existing WORKSPACE [0]
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
RESUME="${RESUME:-0}"
require_flag RESUME
WORKSPACE="$(abspath "${WORKSPACE:-$REPO_ROOT/workspace/e0/$TASK}")"

if [ "$RESUME" = 1 ] && [ -d "$WORKSPACE" ]; then
  [ "$(next_round_index "$WORKSPACE")" = 0 ] || die "$WORKSPACE has archived rounds; it is not an E0 workspace"
  if [ -f "$WORKSPACE/live_cycle_manifest.json" ]; then
    status="$(check_round_complete "$WORKSPACE/raw_scores.json")" \
      || die "E0 in $WORKSPACE cannot be used; fix the problem above, delete the workspace and run again"
    echo "[e0] already finished: $status"
    exit 0
  fi
  echo "[e0] resuming $WORKSPACE"
else
  [ ! -e "$WORKSPACE" ] || die "WORKSPACE already exists: $WORKSPACE (RESUME=1 finishes an interrupted E0)"
fi
export_agent_env
preflight

echo "[e0] task=$TASK grid=${BRANCHES}x$((REFINE + 1)) agent=$AGENT explore_shell=$EXPLORE_SHELL"
echo "[e0] workspace: $WORKSPACE"
cd "$REPO_ROOT"
run_fixed_round 0 off || die "E0 stopped early; rerun with RESUME=1 to finish it"
[ "$DRY_RUN" = 0 ] || exit 0

status="$(check_round_complete "$WORKSPACE/raw_scores.json")" \
  || die "E0 in $WORKSPACE cannot be used; fix the problem above, delete the workspace and run again"
echo "[e0] done: $status"
echo "[e0] next: E0_WORKSPACE=$WORKSPACE bash scripts/run_dream_rsi.sh $TASK (and run_baseline.sh)"
