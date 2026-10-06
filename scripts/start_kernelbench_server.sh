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

# Start the KernelBench evaluation server used by the kb_* tasks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

case "${1:-}" in
  -h | --help)
    cat <<'EOF'
Usage: bash scripts/start_kernelbench_server.sh [SERVER OPTIONS]

Starts the KernelBench evaluation server on SERVER_HOST:SERVER_PORT, running
one evaluation at a time. The address must match
tasks/kernelbench/server_info.yaml. Other options (--eval-timeout,
--build-dir) are passed to tasks/kernelbench/server/launch_server.py.
Setup instructions: tasks/kernelbench/README.md.

Settings (environment variables, defaults in brackets):
  PYTHON            interpreter with torch, KernelBench, fastapi and uvicorn [python]
  KERNELBENCH_ROOT  KernelBench checkout to use if KernelBench is not installed
  SERVER_HOST       bind address [127.0.0.1]
  SERVER_PORT       port [8010]
EOF
    exit 0
    ;;
esac

exec "${PYTHON:-python}" "$REPO_ROOT/tasks/kernelbench/server/launch_server.py" \
  --host "${SERVER_HOST:-127.0.0.1}" --port "${SERVER_PORT:-8010}" --max-concurrency 1 "$@"
