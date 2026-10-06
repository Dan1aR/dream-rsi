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

"""KernelBench evaluation server for the GPU kernel tasks.

Each task's ``evaluator.py`` POSTs the candidate program to ``/evaluate``. The
server scores it with the official ``kernelbench.eval.eval_kernel_against_ref``
against the task's reference model (``<task>/init_program.py``) and returns
``combined_score = 1 / latency_ms`` for correct candidates, 0 otherwise.

    python tasks/kernelbench/server/launch_server.py --port 8010 --max-concurrency 1

KernelBench must be importable: install it (``pip install -e <KernelBench>``)
or point ``KERNELBENCH_ROOT`` at a checkout.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import numbers
import os
from queue import Empty, Queue
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# third_party/kernelbench/<task>/init_program.py holds each task's reference
# model: the unmodified KernelBench problem.
TASKS_ROOT = Path(__file__).resolve().parents[3] / "third_party" / "kernelbench"

_EVAL_SEM = BoundedSemaphore(1)
_BUILD_DIR = Path(tempfile.gettempdir()) / "dream_rsi_kernelbench_build"

# Process-isolation config: each /evaluate runs in a fresh spawned subprocess so
# a poisoned CUDA context (e.g. cudaErrorIllegalAddress) or a hard crash cannot
# leak into the server process or other evaluations.
_SPAWN_CTX = None
_EVAL_TIMEOUT = 900.0
_EVAL_ISOLATION = True


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(req_id: str, message: str, emit: "callable | None" = None) -> None:
    line = f"[{_ts()}] [official-server] [{req_id}] {message}"
    # Use original stdout to avoid duplicate SSE logs when stdout is redirected.
    print(line, file=sys.__stdout__, flush=True)
    if emit is not None:
        emit(line)


class _LineFanoutStream:
    """Stream wrapper: keep terminal output while emitting complete lines."""

    def __init__(self, base_stream: Any, emit: "callable | None" = None):
        self._base_stream = base_stream
        self._emit = emit
        self._buf = ""

    def write(self, data: str) -> int:
        text = str(data)
        self._base_stream.write(text)
        if self._emit is not None:
            self._buf += text
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                if line:
                    self._emit(line)
        return len(text)

    def flush(self) -> None:
        self._base_stream.flush()
        if self._emit is not None and self._buf:
            self._emit(self._buf)
            self._buf = ""


@contextlib.contextmanager
def _heartbeat(req_id: str, stage: str, interval_sec: float = 15.0, emit: "callable | None" = None):
    stop = threading.Event()

    def _runner() -> None:
        while not stop.wait(interval_sec):
            _log(req_id, f"still running: {stage}", emit=emit)

    th = threading.Thread(target=_runner, daemon=True)
    th.start()
    try:
        yield
    finally:
        stop.set()
        th.join(timeout=0.1)


def _ensure_dotenv_compat() -> None:
    """KernelBench may import dotenv unconditionally; provide no-op fallback."""
    try:
        import dotenv  # type: ignore # noqa: F401
    except Exception:
        stub = types.ModuleType("dotenv")

        def load_dotenv(*args: Any, **kwargs: Any) -> bool:
            return False

        stub.load_dotenv = load_dotenv  # type: ignore[attr-defined]
        sys.modules["dotenv"] = stub


def _ensure_kernelbench_importable() -> None:
    """Use an installed ``kernelbench`` package, else ``$KERNELBENCH_ROOT/src``."""
    _ensure_dotenv_compat()
    root = os.environ.get("KERNELBENCH_ROOT", "").strip()
    if root:
        src = Path(root).expanduser().resolve() / "src"
        if not (src / "kernelbench").is_dir():
            raise RuntimeError(f"KERNELBENCH_ROOT={root} has no src/kernelbench package")
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        return
    try:
        import kernelbench  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import kernelbench: install KernelBench "
            "(pip install -e <KernelBench checkout>) or set KERNELBENCH_ROOT"
        ) from exc


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _detect_program_format(
    program_src: str,
) -> Literal["model_new", "reference_model", "custom_kernel"]:
    if re.search(r"class\s+ModelNew\b", program_src):
        return "model_new"
    if re.search(r"class\s+Model\b", program_src):
        return "reference_model"
    if re.search(r"def\s+custom_kernel\s*\(", program_src):
        return "custom_kernel"
    raise ValueError(
        "program_src must contain `class ModelNew`, `class Model`, or `def custom_kernel(data)`."
    )


def _to_modelnew_program(program_src: str, program_format: str) -> tuple[str, str]:
    resolved = _detect_program_format(program_src) if program_format == "auto" else program_format

    if resolved == "model_new":
        return program_src, resolved

    if resolved == "reference_model":
        converted = re.sub(r"\bclass\s+Model(\s*\()", r"class ModelNew\1", program_src, count=1)
        converted = converted.replace("super(Model, self)", "super(ModelNew, self)")
        converted = converted.replace("super(Model,self)", "super(ModelNew, self)")
        if converted == program_src:
            raise ValueError("Failed to convert reference-style `class Model` into `class ModelNew`.")
        return converted, resolved

    if resolved != "custom_kernel":
        raise ValueError(f"Unsupported program_format: {program_format}")

    has_torch_import = re.search(r"^\s*import\s+torch\b", program_src, flags=re.MULTILINE) is not None
    prefix = "" if has_torch_import else "import torch\n"
    wrapper = (
        "\n\n"
        "class ModelNew(torch.nn.Module):\n"
        "    def __init__(self, *args, **kwargs):\n"
        "        super().__init__()\n\n"
        "    def forward(self, *args):\n"
        "        return custom_kernel(tuple(args))\n"
    )
    return prefix + program_src + wrapper, resolved


def _load_task_reference(task_name: str) -> str:
    """Return the reference model source: the task's unmodified KernelBench problem."""
    task_dir = (TASKS_ROOT / task_name).resolve()
    reference_path = task_dir / "init_program.py"
    if task_dir.parent != TASKS_ROOT or not reference_path.is_file():
        raise FileNotFoundError(f"Unknown task `{task_name}` (missing {reference_path})")
    return reference_path.read_text(encoding="utf-8")


def _extract_error(metadata: dict[str, Any]) -> tuple[str, str]:
    for key in ("compilation_error", "runtime_error", "correctness_issue", "other_error"):
        val = metadata.get(key)
        if val:
            return str(val), key
    return "kernelbench_official_failed", "KernelBenchOfficialFailed"


def _build_failed_result(error: str, error_name: str) -> dict[str, Any]:
    return {
        "compiled": False,
        "correctness": False,
        "combined_score": 0.0,
        "metadata": {
            "status": "failed",
            "mode": "benchmark",
            "run_success": False,
            "run_passed": False,
            "latency_ms": None,
            "per_benchmark_means_us": [],
            "eval_mode": "kernelbench_official",
        },
        "runtime_stats": {"summary": {"latency_ms": None, "score": 0.0, "passed": False}},
        "score": 0.0,
        "error": error,
        "error_name": error_name,
    }


def _infer_gpu_arch_from_name(device_name: str) -> list[str]:
    name = (device_name or "").lower()
    if "h100" in name or "h200" in name:
        return ["Hopper"]
    if "a100" in name or "a10" in name:
        return ["Ampere"]
    if "l40" in name or "l4" in name or "rtx 40" in name or "4090" in name:
        return ["Ada"]
    if "t4" in name:
        return ["Turing"]
    return ["Ada"]


def _resolve_build_dir(req: "EvalReq", model_src: str) -> str:
    digest = hashlib.sha1(model_src.encode("utf-8")).hexdigest()
    base = Path(req.build_dir_prefix).expanduser() if req.build_dir_prefix else _BUILD_DIR
    build_dir = base / f"{req.task}_{digest}"
    if req.clear_cache:
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    return str(build_dir)


def _evaluate_with_kernelbench(
    req: "EvalReq",
    req_id: str,
    *,
    emit: "callable | None" = None,
) -> dict[str, Any]:
    _log(req_id, "prepare kernelbench imports", emit=emit)
    _ensure_kernelbench_importable()
    from kernelbench import eval as kb_eval
    from kernelbench import timing as kb_timing
    from kernelbench import utils as kb_utils

    if not torch.cuda.is_available():
        _log(req_id, "CUDA unavailable on this node", emit=emit)
        return _build_failed_result("CUDA is not available", "CudaUnavailable")

    device_name = torch.cuda.get_device_name(req.gpu_id)
    gpu_arch = req.gpu_arch or _infer_gpu_arch_from_name(device_name)
    kb_utils.set_gpu_arch(gpu_arch)
    _log(
        req_id,
        f"CUDA ready, using cuda:{req.gpu_id} ({device_name}), gpu_arch={gpu_arch}",
        emit=emit,
    )
    reference_src = _load_task_reference(req.task)
    _log(req_id, f"task loaded: {req.task}", emit=emit)
    modelnew_src, resolved_format = _to_modelnew_program(req.program_src, req.program_format)
    _log(req_id, f"program format resolved: {resolved_format}", emit=emit)
    build_dir = _resolve_build_dir(req, modelnew_src)
    _log(req_id, f"build_dir={build_dir}", emit=emit)

    _log(
        req_id,
        "start eval_kernel_against_ref "
        f"(correct_trials={req.num_correct_trials}, perf_trials={req.num_perf_trials})",
        emit=emit,
    )
    stdout_stream = _LineFanoutStream(sys.stdout, emit=emit)
    stderr_stream = _LineFanoutStream(sys.stderr, emit=emit)
    with _heartbeat(req_id, "eval_kernel_against_ref", emit=emit):
        with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
            eval_result = kb_eval.eval_kernel_against_ref(
                original_model_src=reference_src,
                custom_model_src=modelnew_src,
                seed_num=req.seed_num,
                num_correct_trials=req.num_correct_trials,
                num_perf_trials=req.num_perf_trials,
                measure_performance=req.measure_performance,
                timing_method=req.timing_method,
                verbose=req.verbose,
                build_dir=build_dir,
                device=torch.device(f"cuda:{req.gpu_id}"),
                backend=req.backend,
                precision=kb_eval.get_torch_dtype_from_string(req.precision),
            )
    _log(req_id, "eval_kernel_against_ref finished", emit=emit)

    ref_runtime_stats = None
    if req.include_reference_timing:
        _log(req_id, f"start reference timing (trials={req.ref_num_trials})", emit=emit)
        stdout_stream = _LineFanoutStream(sys.stdout, emit=emit)
        stderr_stream = _LineFanoutStream(sys.stderr, emit=emit)
        with _heartbeat(req_id, "measure_ref_program_time", emit=emit):
            with contextlib.redirect_stdout(stdout_stream), contextlib.redirect_stderr(stderr_stream):
                ref_runtime_stats = kb_timing.measure_ref_program_time(
                    ref_arch_name=f"{req.task}-reference",
                    ref_arch_src=reference_src,
                    num_trials=req.ref_num_trials,
                    timing_method=req.timing_method,
                    device=torch.device(f"cuda:{req.gpu_id}"),
                    precision=req.precision,
                    verbose=False,
                    use_torch_compile=False,
                )
        _log(req_id, "reference timing finished", emit=emit)

    compiled = bool(getattr(eval_result, "compiled", False))
    correctness = bool(getattr(eval_result, "correctness", False))
    runtime = getattr(eval_result, "runtime", None)
    latency_ms = float(runtime) if (_is_number(runtime) and float(runtime) > 0.0) else None

    score = (1.0 / latency_ms) if (compiled and correctness and latency_ms) else 0.0
    status = "passed" if (compiled and correctness and latency_ms) else "failed"
    official_runtime_stats = _json_safe(getattr(eval_result, "runtime_stats", {}) or {})
    official_metadata = _json_safe(getattr(eval_result, "metadata", {}) or {})
    ref_runtime_stats = _json_safe(ref_runtime_stats or {})

    per_benchmark_means_us = []
    if latency_ms is not None:
        per_benchmark_means_us = [float(latency_ms) * 1000.0]

    metadata: dict[str, Any] = {
        "status": status,
        "mode": "benchmark",
        "run_success": compiled,
        "run_passed": correctness,
        "latency_ms": latency_ms,
        "per_benchmark_means_us": per_benchmark_means_us,
        "eval_mode": "kernelbench_official",
        "program_format": resolved_format,
        "task": req.task,
        "backend": req.backend,
        "precision": req.precision,
        "timing_method": req.timing_method,
        "official_runtime_stats": official_runtime_stats,
        "official_metadata": official_metadata,
    }
    if ref_runtime_stats:
        metadata["reference_runtime_stats"] = ref_runtime_stats
        ref_mean = ref_runtime_stats.get("mean")
        if _is_number(ref_mean) and latency_ms and latency_ms > 0:
            metadata["speedup_vs_reference"] = float(ref_mean) / float(latency_ms)

    payload: dict[str, Any] = {
        "compiled": compiled,
        "correctness": correctness,
        "combined_score": score,
        "metadata": metadata,
        "runtime_stats": {
            "summary": {
                "latency_ms": latency_ms,
                "score": score,
                "passed": bool(compiled and correctness),
            },
            "per_benchmark_means_us": per_benchmark_means_us,
            "official_runtime_stats": official_runtime_stats,
        },
        "score": score,
        "official_raw": {
            "compiled": compiled,
            "correctness": correctness,
            "runtime": _json_safe(runtime),
            "runtime_stats": official_runtime_stats,
            "metadata": official_metadata,
        },
    }

    if not (compiled and correctness):
        error, error_name = _extract_error(copy.deepcopy(official_metadata))
        payload["error"] = error
        payload["error_name"] = error_name

    return payload


class EvalReq(BaseModel):
    program_src: str
    task: str
    eval_mode: str = "kernelbench_official"
    program_format: Literal["auto", "custom_kernel", "model_new", "reference_model"] = "auto"
    backend: str = "cuda"
    precision: str = "fp32"
    timing_method: str = "cuda_event"
    num_correct_trials: int = Field(default=5, ge=1)
    num_perf_trials: int = Field(default=100, ge=1)
    seed_num: int = 42
    gpu_id: int = 0
    gpu_arch: list[str] | None = None
    build_dir_prefix: str = ""
    clear_cache: bool = False
    verbose: bool = False
    measure_performance: bool = True
    include_reference_timing: bool = False
    ref_num_trials: int = Field(default=100, ge=1)


app = FastAPI(title="KernelBench evaluation server")


def _get_spawn_ctx():
    """Lazily create a `spawn` multiprocessing context.

    `spawn` (not `fork`) is mandatory here: the server process may have already
    touched CUDA, and forking a CUDA-initialized process yields undefined
    behavior. `spawn` gives every eval a brand-new interpreter + CUDA context.
    """
    global _SPAWN_CTX
    if _SPAWN_CTX is None:
        import multiprocessing as mp

        _SPAWN_CTX = mp.get_context("spawn")
    return _SPAWN_CTX


def _subprocess_eval_entry(req_dict: dict, req_id: str, out_queue) -> None:
    """Entry point executed inside the isolated child process."""

    def _emit(line: str) -> None:
        try:
            out_queue.put({"type": "log", "line": line})
        except Exception:
            pass

    try:
        req = EvalReq(**req_dict)
        result = _evaluate_with_kernelbench(req, req_id, emit=_emit)
        out_queue.put(
            {
                "type": "result",
                "payload": {
                    "success": bool(result.get("correctness")),
                    "result": result,
                },
            }
        )
    except Exception as e:  # noqa: BLE001 - report any failure back to parent
        _emit(f"[official-server] [{req_id}] subprocess exception: {type(e).__name__}: {e}")
        _emit(traceback.format_exc().rstrip())
        out_queue.put(
            {
                "type": "result",
                "payload": {
                    "success": False,
                    "result": _build_failed_result(str(e), type(e).__name__),
                },
            }
        )
    finally:
        try:
            out_queue.close()
        except Exception:
            pass


def _terminate_process(proc, req_id: str, emit: "callable | None" = None) -> None:
    if proc is None:
        return
    try:
        if proc.is_alive():
            _log(req_id, f"killing eval subprocess pid={proc.pid}", emit=emit)
            proc.kill()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)
        else:
            proc.join(timeout=1)
    except Exception:
        pass


def _run_isolated_eval(
    req: "EvalReq",
    req_id: str,
    *,
    emit: "callable | None" = None,
) -> dict[str, Any]:
    """Run one evaluation in a throwaway subprocess and stream its logs back."""
    ctx = _get_spawn_ctx()
    out_queue = ctx.Queue()
    req_dict = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    proc = ctx.Process(
        target=_subprocess_eval_entry,
        args=(req_dict, req_id, out_queue),
        daemon=True,
    )
    result_payload: dict[str, Any] | None = None
    try:
        proc.start()
        _log(req_id, f"eval subprocess started pid={proc.pid}", emit=emit)
        deadline = time.time() + _EVAL_TIMEOUT
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                _log(req_id, f"eval timed out after {_EVAL_TIMEOUT:.0f}s", emit=emit)
                break
            try:
                item = out_queue.get(timeout=min(remaining, 5.0))
            except Empty:
                if not proc.is_alive():
                    # Hard crash (e.g. segfault from illegal memory access) with
                    # no result delivered. The child dies; the server survives.
                    _log(
                        req_id,
                        f"eval subprocess exited without result (exitcode={proc.exitcode})",
                        emit=emit,
                    )
                    break
                continue
            if item.get("type") == "log":
                # Child already printed to its inherited stdout; only forward to SSE.
                if emit is not None:
                    emit(item.get("line", ""))
            elif item.get("type") == "result":
                result_payload = item.get("payload")
                break
    finally:
        _terminate_process(proc, req_id, emit)
        try:
            out_queue.close()
        except Exception:
            pass

    if result_payload is None:
        return {
            "success": False,
            "result": _build_failed_result(
                "Evaluation crashed or timed out in isolated subprocess",
                "IsolatedEvalFailed",
            ),
        }
    return result_payload


def _run_evaluate_request(
    req: EvalReq,
    req_id: str,
    *,
    emit: "callable | None" = None,
) -> dict[str, Any]:
    if not req.build_dir_prefix:
        # Spawned eval processes re-import this module and do not see --build-dir,
        # so the server's build dir travels with the request.
        req.build_dir_prefix = str(_BUILD_DIR)
    _log(
        req_id,
        "/evaluate "
        f"task={req.task} eval_mode={req.eval_mode} "
        f"program_format={req.program_format} backend={req.backend} "
        f"precision={req.precision} gpu_id={req.gpu_id}",
        emit=emit,
    )
    _log(req_id, "waiting for evaluation semaphore", emit=emit)
    wait_round = 0
    acquired = False
    while not acquired:
        acquired = _EVAL_SEM.acquire(timeout=5.0)
        if acquired:
            break
        wait_round += 1
        _log(req_id, f"still waiting semaphore ({wait_round * 5}s)", emit=emit)
    if not acquired:
        return {"success": False, "result": _build_failed_result("Server busy", "ServerBusy")}
    try:
        if _EVAL_ISOLATION:
            _log(req_id, "semaphore acquired, begin isolated evaluation", emit=emit)
            payload = _run_isolated_eval(req, req_id, emit=emit)
            result = payload.get("result", {}) or {}
        else:
            _log(req_id, "semaphore acquired, begin in-process evaluation", emit=emit)
            result = _evaluate_with_kernelbench(req, req_id, emit=emit)
            payload = {"success": bool(result.get("correctness")), "result": result}
        _log(
            req_id,
            "done "
            f"compiled={result.get('compiled')} "
            f"correctness={result.get('correctness')} "
            f"score={result.get('combined_score', result.get('score'))} "
            f"error={result.get('error')}",
            emit=emit,
        )
        return payload
    except Exception as e:
        _log(req_id, f"exception: {type(e).__name__}: {e}", emit=emit)
        _log(req_id, traceback.format_exc().rstrip(), emit=emit)
        return {
            "success": False,
            "result": _build_failed_result(str(e), type(e).__name__),
        }
    finally:
        _EVAL_SEM.release()
        _log(req_id, "semaphore released", emit=emit)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate")
def evaluate(req: EvalReq) -> dict[str, Any]:
    req_id = uuid.uuid4().hex[:8]
    return _run_evaluate_request(req, req_id)


@app.post("/evaluate/stream")
def evaluate_stream(req: EvalReq):
    req_id = uuid.uuid4().hex[:8]
    events: "Queue[dict[str, Any]]" = Queue()

    def _emit(line: str) -> None:
        events.put({"type": "log", "line": line})

    def _worker() -> None:
        try:
            out = _run_evaluate_request(req, req_id, emit=_emit)
            events.put({"type": "result", "payload": out})
        except Exception as e:
            events.put(
                {
                    "type": "result",
                    "payload": {
                        "success": False,
                        "result": _build_failed_result(str(e), type(e).__name__),
                    },
                }
            )
        finally:
            events.put({"type": "end"})

    threading.Thread(target=_worker, daemon=True).start()

    def _iter_sse():
        yield f"event: meta\ndata: {json.dumps({'request_id': req_id}, ensure_ascii=False)}\n\n"
        while True:
            try:
                item = events.get(timeout=10.0)
            except Empty:
                # keepalive to avoid idle proxy timeouts
                yield ": keepalive\n\n"
                continue
            etype = item.get("type")
            if etype == "log":
                payload = {"request_id": req_id, "line": item.get("line", "")}
                yield f"event: log\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif etype == "result":
                payload = item.get("payload", {})
                yield f"event: result\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif etype == "end":
                yield "event: end\ndata: {}\n\n"
                break

    return StreamingResponse(_iter_sse(), media_type="text/event-stream")


def main() -> None:
    global _EVAL_SEM, _EVAL_TIMEOUT, _EVAL_ISOLATION, _BUILD_DIR
    parser = argparse.ArgumentParser(description="Official KernelBench evaluator over server-call API")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. The server executes submitted code, so only expose it "
        "(e.g. --host 0.0.0.0) on a trusted network.",
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--eval-timeout",
        type=float,
        default=900.0,
        help="Per-request wall-clock timeout (seconds) for the isolated eval subprocess.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=_BUILD_DIR,
        help=f"Where CUDA extensions are compiled (default: {_BUILD_DIR}).",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Disable subprocess isolation and evaluate in-process (for debugging).",
    )
    args = parser.parse_args()

    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be > 0")
    _EVAL_SEM = BoundedSemaphore(args.max_concurrency)
    _EVAL_TIMEOUT = args.eval_timeout
    _EVAL_ISOLATION = not args.no_isolation
    _BUILD_DIR = args.build_dir.expanduser().resolve()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
