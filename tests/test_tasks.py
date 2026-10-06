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

"""The shipped benchmark tasks are complete and their evaluators score the seeds."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dream_rsi.exploration.evaluate import run_evaluator

TASKS = Path(__file__).resolve().parents[1] / "tasks"
THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
MATH_TASKS = ["sums_diffs", "autocorrelation_third", "circle_packing_26", "circle_packing_32"]
KERNEL_TASKS = ["kb_l3_11_vgg16", "kb_l1_40_layernorm", "kb_l2_8_convdiv", "kb_l2_43_convmax"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_fetched(task: str) -> Path:
    """The SimpleTES task files are downloaded by scripts/fetch_tasks.sh, not shipped."""
    task_dir = TASKS / task
    if not (task_dir / "evaluator.py").is_file():
        pytest.skip(f"{task} files missing: run scripts/fetch_tasks.sh {task}")
    return task_dir


@pytest.mark.parametrize("task", [*MATH_TASKS, "lasso_path"])
def test_fetched_task_has_problem_evaluator_and_seed(task: str):
    task_dir = _require_fetched(task)
    for name in (f"{task}.txt", "evaluator.py", "init_program.py"):
        assert (task_dir / name).is_file(), task_dir / name


@pytest.mark.parametrize("task", KERNEL_TASKS)
def test_kernel_task_has_problem_evaluator_and_seed(task: str):
    task_dir = TASKS / "kernelbench" / task
    for path in (task_dir / f"{task}.triton", task_dir / "evaluator.py",
                 THIRD_PARTY / "kernelbench" / task / "init_program.py"):
        assert path.is_file(), path


@pytest.mark.parametrize("task", MATH_TASKS)
def test_math_evaluator_scores_the_initial_program(task: str):
    task_dir = _require_fetched(task)
    metrics = run_evaluator(task_dir / "evaluator.py", task_dir / "init_program.py", timeout_sec=300)
    assert not metrics.get("error"), metrics.get("error")
    assert metrics["score"] > 0


def test_kernel_client_defaults_follow_the_paper_protocol(monkeypatch):
    for key in ("GPUKERNEL_SERVER_INFO", "GPUKERNEL_EVAL_MODE", "GPUKERNEL_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    client = _load(TASKS / "kernelbench/kb_l2_8_convdiv/evaluator.py", "kb_client")

    assert client.TASK == "kb_l2_8_convdiv"
    servers, timeout = client._read_server_endpoints()
    assert servers == [{"ip": "localhost", "port": 8010}]
    assert timeout == 3000
    request = client._build_request_data("class ModelNew: pass\n")
    assert request["eval_mode"] == "kernelbench_official"
    assert request["backend"] == "cuda"
    assert request["timing_method"] == "cuda_event"


def test_kernel_server_uses_the_seed_program_as_reference():
    for module in ("torch", "fastapi", "uvicorn"):
        pytest.importorskip(module)
    server = _load(TASKS / "kernelbench/server/launch_server.py", "kb_server")

    reference = server._load_task_reference("kb_l1_40_layernorm")
    assert reference == (THIRD_PARTY / "kernelbench/kb_l1_40_layernorm/init_program.py").read_text()
    with pytest.raises(FileNotFoundError):
        server._load_task_reference("../../tasks/lasso_path")
    converted, fmt = server._to_modelnew_program(reference, "auto")
    assert fmt == "reference_model"
    assert "class ModelNew(" in converted
