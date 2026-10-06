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

# Recursive Fixed Exploration: every online round uses the hand-designed
# starting policy, which is never updated.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_baseline.sh TASK

Runs the Recursive Fixed Exploration baseline on TASK for ROUNDS online
rounds. Each round is explored with the hand-designed starting policy and its
agents can read all earlier rounds, but the policy itself never changes.

Settings (environment variables, defaults in brackets):
  WORKSPACE     output directory [workspace/baseline/TASK, or
                workspace/baseline_guidance/TASK with GUIDANCE=1]
  E0_WORKSPACE  finished E0 from scripts/run_e0.sh to use as the first round;
                without it, the first round is explored here
  RESUME        1 = continue an existing WORKSPACE from its first unfinished
                round [0]
  ROUNDS        online rounds T, including E0 [task default]
  GUIDANCE      1 = also summarize the explored directions after each round
                and inject them into the next round's prompts
                (Fixed + Guidance in Figure 6) [0]
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
if [ "$GUIDANCE" = 1 ]; then
  method=baseline_guidance
else
  method=baseline
fi
WORKSPACE="$(abspath "${WORKSPACE:-$REPO_ROOT/workspace/$method/$TASK}")"

first_cycle=0
copy_e0=0
if [ "$RESUME" = 1 ] && [ -d "$WORKSPACE" ]; then
  first_cycle="$(next_round_index "$WORKSPACE")"
  [ "$first_cycle" -lt "$ROUNDS" ] || die "all $ROUNDS rounds in $WORKSPACE are finished"
  echo "[baseline] resuming $WORKSPACE at round $((first_cycle + 1))"
else
  [ ! -e "$WORKSPACE" ] || die "WORKSPACE already exists: $WORKSPACE (RESUME=1 continues it)"
  if [ -n "${E0_WORKSPACE:-}" ]; then
    E0_WORKSPACE="$(abspath "$E0_WORKSPACE")"
    check_e0_workspace "$E0_WORKSPACE"
    copy_e0=1
  fi
fi
export_agent_env
preflight

echo "[baseline] task=$TASK rounds=$ROUNDS grid=${BRANCHES}x$((REFINE + 1)) agent=$AGENT guidance=$GUIDANCE explore_shell=$EXPLORE_SHELL"
echo "[baseline] workspace: $WORKSPACE"
cd "$REPO_ROOT"
if [ "$copy_e0" = 1 ]; then
  copy_e0_workspace "$E0_WORKSPACE"
fi

for ((cycle = first_cycle; cycle < ROUNDS; cycle++)); do
  echo "[baseline] round $((cycle + 1))/$ROUNDS"
  # The first round may already be explored: a copied E0, or a resumed round
  # whose exploration finished (its live_cycle_manifest.json is written last).
  if [ "$cycle" = "$first_cycle" ] \
    && { [ "$copy_e0" = 1 ] || [ -f "$WORKSPACE/live_cycle_manifest.json" ]; }; then
    echo "[baseline] round $((cycle + 1)) is already explored"
  else
    [ "$cycle" != "$first_cycle" ] || resume_direction_memory "$cycle"
    run_fixed_round "$cycle" "$DIRECTION_MODE" \
      || die "round $((cycle + 1)) stopped early; rerun with RESUME=1 to continue"
  fi
  if [ "$GUIDANCE" = 1 ] \
    && ! run_cmd "$PYTHON" -m dream_rsi learning directions -w "$WORKSPACE" -a "$AGENT" --cycle "$cycle"; then
    # The round stays live, so RESUME=1 retries the summary without exploring again.
    [ -f "$WORKSPACE/directions/current.json" ] \
      || die "the direction summary of round $((cycle + 1)) failed and there is no earlier one; rerun with RESUME=1 to retry it"
    warn "the direction summary of round $((cycle + 1)) failed; the next round uses the previous one"
  fi
  run_cmd "$PYTHON" -m dream_rsi explore archive -w "$WORKSPACE" --cycle "$cycle" \
    || die "could not archive round $((cycle + 1))"
done
[ "$DRY_RUN" = 0 ] || exit 0

echo "[baseline] finished $ROUNDS rounds in $WORKSPACE"
report_rounds "$WORKSPACE"
