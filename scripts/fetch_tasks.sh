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

# Fetch the math and Lasso task files from SimpleTES.
#
# The problem statements, evaluators and initial programs of the tasks
# sums_diffs, autocorrelation_third, circle_packing_26, circle_packing_32 and
# lasso_path come from SimpleTES (https://github.com/wq-will/SimpleTES), which
# is distributed under the GNU AGPL-3.0. They are not part of this repository.
# This script downloads them from the pinned SimpleTES commit into tasks/<task>/
# so that the launch scripts find them where they expect them.
#
# Usage: bash scripts/fetch_tasks.sh            # all five tasks
#        bash scripts/fetch_tasks.sh lasso_path  # one or more tasks

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMPLETES_REPO="${SIMPLETES_REPO:-https://github.com/wq-will/SimpleTES.git}"
# The commit the paper's experiments used. Override with SIMPLETES_COMMIT.
SIMPLETES_COMMIT="${SIMPLETES_COMMIT:-47d3413da1d85dc24341219d47452d2601e56a57}"

# task -> directory inside SimpleTES/datasets
declare -A SOURCE=(
  [sums_diffs]=sums_diffs/sums_diffs
  [autocorrelation_third]=autocorrelation/autocorrelation_third
  [circle_packing_26]=circle_packing/circle_packing_26
  [circle_packing_32]=circle_packing/circle_packing_32
  [lasso_path]=numerical_tasks/lasso_path
)
# Files copied for every task, plus extras per task.
COMMON_FILES=(evaluator.py init_program.py)
declare -A EXTRA_FILES=(
  [lasso_path]="generate_results.py"
)

if [ "$#" -gt 0 ]; then
  TASKS=("$@")
else
  TASKS=(sums_diffs autocorrelation_third circle_packing_26 circle_packing_32 lasso_path)
fi
for task in "${TASKS[@]}"; do
  [ -n "${SOURCE[$task]:-}" ] || { echo "unknown task: $task (choose from ${!SOURCE[*]})" >&2; exit 1; }
done

command -v git >/dev/null 2>&1 || { echo "git is required" >&2; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Fetching SimpleTES @ ${SIMPLETES_COMMIT:0:12} ..."
git -C "$tmp" init -q
git -C "$tmp" remote add origin "$SIMPLETES_REPO"
git -C "$tmp" fetch -q --depth 1 origin "$SIMPLETES_COMMIT"
git -C "$tmp" checkout -q FETCH_HEAD

for task in "${TASKS[@]}"; do
  src="$tmp/datasets/${SOURCE[$task]}"
  dst="$REPO_ROOT/tasks/$task"
  mkdir -p "$dst"
  files=("${COMMON_FILES[@]}" "$task.txt")
  # shellcheck disable=SC2206
  files+=(${EXTRA_FILES[$task]:-})
  for f in "${files[@]}"; do
    [ -f "$src/$f" ] || { echo "missing in SimpleTES: datasets/${SOURCE[$task]}/$f" >&2; exit 1; }
    cp "$src/$f" "$dst/$f"
  done
  # The math evaluators call an optional SimpleTES hook that records the best
  # construction for the SimpleTES CLI. It needs the simpletes package, which is
  # not installed here, so drop the import and the call (the paper's runs did
  # the same). Scores are unaffected.
  python3 - "$dst/evaluator.py" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path, encoding="utf-8").read()
text = re.sub(r"^.*capture_construction_if_requested.*\n", "", text, flags=re.M)
open(path, "w", encoding="utf-8").write(text)
PY
  cp "$tmp/LICENSE" "$dst/LICENSE.SimpleTES"
  echo "  tasks/$task: ${files[*]}"
done

echo "Done. These files are AGPL-3.0 licensed by SimpleTES (see tasks/<task>/LICENSE.SimpleTES)."
if printf '%s\n' "${TASKS[@]}" | grep -qx lasso_path; then
  echo "lasso_path also needs Eigen and g++: bash tasks/lasso_path/setup.sh"
fi
