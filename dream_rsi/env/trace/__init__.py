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

"""Trace data: ``raw_scores.json`` format and in-memory :class:`Trace` trees."""

from dream_rsi.env.trace.raw import (
    RAW_TRACE_VERSION,
    CellRaw,
    RawNode,
    RawTrace,
    export_raw_scores,
    load_raw_scores,
    parse_score_json,
    raw_trace_to_trace,
    trace_from_raw_file,
    trace_from_workspace,
    write_raw_scores,
)
from dream_rsi.env.trace.types import Node, Trace

__all__ = [
    "RAW_TRACE_VERSION",
    "CellRaw",
    "RawNode",
    "RawTrace",
    "Node",
    "Trace",
    "export_raw_scores",
    "load_raw_scores",
    "write_raw_scores",
    "parse_score_json",
    "raw_trace_to_trace",
    "trace_from_workspace",
    "trace_from_raw_file",
]
