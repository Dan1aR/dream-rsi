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

"""Canonical paths for discovered policies.

``METHOD_REL`` and ``METHOD_TEMPLATE_REL`` are relative to the directory that
contains the ``dream_rsi`` package and to every learning job, which mirrors
that layout so the proposer edits the same module path it reads.
``PROMOTED_POLICY_REL`` is relative to an exploration workspace and holds the
policy promoted by the latest learning phase.
"""

METHOD_REL = "dream_rsi/policy/discovered/method.py"
METHOD_TEMPLATE_REL = "dream_rsi/policy/discovered/method.template.py"
PROMOTED_POLICY_REL = "policy/method.py"
