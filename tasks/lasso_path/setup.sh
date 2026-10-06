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

# Fetch the Eigen headers the Lasso evaluator compiles against and check for g++.
# Run once per machine: bash tasks/lasso_path/setup.sh

set -euo pipefail

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EIGEN_VERSION=3.4.0
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz"

if [[ -f "${TASK_DIR}/eigen/Eigen/Core" ]]; then
    echo "Eigen already present at ${TASK_DIR}/eigen"
else
    echo "Downloading Eigen ${EIGEN_VERSION} ..."
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    curl -fsSL "${EIGEN_URL}" -o "${tmp}/eigen.tar.gz"
    tar -xzf "${tmp}/eigen.tar.gz" -C "${tmp}"
    rm -rf "${TASK_DIR}/eigen"
    mv "${tmp}/eigen-${EIGEN_VERSION}" "${TASK_DIR}/eigen"
    echo "Eigen installed at ${TASK_DIR}/eigen"
fi

if command -v g++ >/dev/null 2>&1; then
    echo "g++ found: $(g++ --version | head -1)"
else
    echo "WARNING: g++ not found; the evaluator compiles candidate solvers with g++ -O3 -std=c++17." >&2
    exit 1
fi
