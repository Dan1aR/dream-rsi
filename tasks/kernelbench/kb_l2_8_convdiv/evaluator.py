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

"""Client evaluator for the KernelBench tasks.

Sends the candidate program to the evaluation server in ``../server`` (address
from ``../server_info.yaml`` or ``$GPUKERNEL_SERVER_INFO``). The task name is
this file's directory name.
"""

import os
from typing import Any

import requests
import yaml

TASK = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
_FALLBACK_RESULT = {"compiled": False, "correctness": False, "combined_score": 0.0}


def _read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _is_debug_enabled() -> bool:
    return _env_bool("GPUKERNEL_VERBOSE", False) or _env_bool("GPUKERNEL_CLIENT_DEBUG", False)


def _read_server_endpoints() -> tuple[list[dict[str, Any]], int]:
    gpukernel_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.environ.get("GPUKERNEL_SERVER_INFO") or os.path.join(
        gpukernel_root, "server_info.yaml"
    )
    default_servers = [{"ip": "localhost", "port": 8010}]
    if not os.path.exists(config_path):
        return default_servers, 3000
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    servers = []
    for item in config.get("servers") or []:
        ip = item.get("ip") or item.get("host")
        port = item.get("port", 8010)
        if ip is not None:
            servers.append({"ip": str(ip), "port": int(port)})
    if not servers:
        ip = config.get("server_ip", "localhost")
        port = config.get("server_port", 8010)
        servers = [{"ip": str(ip), "port": int(port)}]
    return servers, int(config.get("request_timeout", 3000))


def _build_request_data(program_src: str) -> dict[str, Any]:
    request_data: dict[str, Any] = {
        "program_src": program_src,
        "task": TASK,
        "eval_mode": os.environ.get("GPUKERNEL_EVAL_MODE", "kernelbench_official"),
    }
    request_data["program_format"] = os.environ.get("GPUKERNEL_PROGRAM_FORMAT", "auto")
    request_data["backend"] = os.environ.get("GPUKERNEL_BACKEND", "cuda")
    request_data["precision"] = os.environ.get("GPUKERNEL_PRECISION", "fp32")
    request_data["timing_method"] = os.environ.get("GPUKERNEL_TIMING_METHOD", "cuda_event")
    request_data["num_correct_trials"] = _env_int("GPUKERNEL_NUM_CORRECT_TRIALS", 5)
    request_data["num_perf_trials"] = _env_int("GPUKERNEL_NUM_PERF_TRIALS", 100)
    request_data["seed_num"] = _env_int("GPUKERNEL_SEED_NUM", 42)
    request_data["gpu_id"] = _env_int("GPUKERNEL_GPU_ID", 0)
    request_data["verbose"] = _env_bool("GPUKERNEL_VERBOSE", False)
    request_data["include_reference_timing"] = _env_bool(
        "GPUKERNEL_INCLUDE_REFERENCE_TIMING", False
    )
    request_data["ref_num_trials"] = _env_int("GPUKERNEL_REF_NUM_TRIALS", 100)
    return request_data


def evaluate(program_path):
    servers, request_timeout = _read_server_endpoints()
    program_src = _read_file(program_path)
    if not program_src:
        if _is_debug_enabled():
            print(f"[Client] empty program source: {program_path}")
        return dict(_FALLBACK_RESULT)

    request_data = _build_request_data(program_src)
    debug = _is_debug_enabled()
    if debug:
        print(
            "[Client] start evaluate "
            f"task={TASK} endpoints={servers} timeout={request_timeout}s "
            f"eval_mode={request_data.get('eval_mode')} "
            f"program_format={request_data.get('program_format')} "
            f"backend={request_data.get('backend')} "
            f"gpu_id={request_data.get('gpu_id')}"
        )
    for server in servers:
        url = f"http://{server['ip']}:{server['port']}/evaluate"
        if debug:
            print(f"[Client] POST {url}")
        try:
            response = requests.post(url, json=request_data, timeout=request_timeout)
            if debug:
                print(f"[Client] {url} -> status={response.status_code}")
            if response.status_code == 200:
                data = response.json()
                if data.get("result"):
                    if debug:
                        result = data["result"] or {}
                        print(
                            "[Client] got result "
                            f"compiled={result.get('compiled')} "
                            f"correctness={result.get('correctness')} "
                            f"score={result.get('combined_score', result.get('score'))} "
                            f"error={result.get('error')}"
                        )
                    return data["result"]
                if debug:
                    print(f"[Client] {url} returned 200 but missing `result` key")
            elif debug:
                body_preview = response.text[:500].replace("\n", "\\n")
                print(f"[Client] {url} non-200 body={body_preview}")
        except Exception as e:
            print(f"[Client] {url} failed: {e}")
    if debug:
        print("[Client] fallback result returned (all endpoints failed or invalid response)")
    return dict(_FALLBACK_RESULT)
