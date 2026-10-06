# KernelBench GPU kernel tasks

Four KernelBench problems (Ouyang et al., 2025). The agent writes a Triton
`ModelNew` that must match the PyTorch reference model; the score is inverse
runtime (1/ms). Section 4.1 of the paper describes the tasks.

| Task directory | Paper name | KernelBench problem |
| --- | --- | --- |
| `kb_l3_11_vgg16` | VGG16 | `level3/11_VGG16.py` |
| `kb_l1_40_layernorm` | LayerNorm | `level1/40_LayerNorm.py` |
| `kb_l2_8_convdiv` | ConvDiv | `level2/8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py` |
| `kb_l2_43_convmax` | ConvMax | `level2/43_Conv3d_Max_LogSumExp_ReLU.py` |

Each task directory contains two files, and its seed lives under
`third_party/`:

| File | Role |
| --- | --- |
| `<task>.triton` | Problem statement shown to the agent |
| `evaluator.py` | Thin client that sends the candidate to the evaluation server |
| `../../third_party/kernelbench/<task>/init_program.py` | The unmodified KernelBench problem (MIT License). It seeds the search and is the reference model for correctness and timing |

Evaluation runs in a separate server process (`server/launch_server.py`) that
owns the GPU. The server scores each candidate with KernelBench's official
`eval_kernel_against_ref`. A candidate scores `1 / latency_ms` if it compiles
and matches the reference on 5 random inputs, and 0 otherwise. Latency is the
mean of 100 CUDA-event timed runs in fp32.

## Setup

The server needs a CUDA GPU, Python 3.10 and KernelBench at commit
[`423217d`](https://github.com/ScalingIntelligence/KernelBench/tree/423217d9fda91e0c2d67e4a43bf62f96f6d104f1):

```bash
git clone https://github.com/ScalingIntelligence/KernelBench.git
cd KernelBench && git checkout 423217d9fda91e0c2d67e4a43bf62f96f6d104f1
pip install -e .                  # torch >= 2.9; its CUDA wheels include Triton
pip install fastapi uvicorn
```

Instead of installing KernelBench, you can set `KERNELBENCH_ROOT=/path/to/KernelBench`.
The paper's numbers were measured on an NVIDIA L40S. Set
`TORCH_CUDA_ARCH_LIST` to match your GPU (8.9 for the L40S).

The DREAM-RSI process only needs `requests` and `pyyaml` for the client side
(`pip install -e ".[kernel]"` from the repository root).

## Running the server

```bash
bash scripts/start_kernelbench_server.sh      # or:
python tasks/kernelbench/server/launch_server.py --port 8010 --max-concurrency 1
```

Server options:

- `--host`: bind address, default `127.0.0.1`. The server executes the code it
  receives, so only expose it on a trusted network.
- `--port`: default 8010.
- `--max-concurrency`: default 1. Concurrent evaluations on one GPU distort the timings.
- `--eval-timeout`: per-candidate limit in seconds, default 900.
- `--build-dir`: KernelBench build cache, default `$TMPDIR/dream_rsi_kernelbench_build`.

Each request runs in a fresh subprocess, so a crashing kernel cannot corrupt
the server's CUDA context.

The clients read the server address from `server_info.yaml` (`localhost:8010`,
request timeout 3000 s). Set `GPUKERNEL_SERVER_INFO` to use another file. With
several servers listed, the client tries them in order until one answers.

## Client settings

`evaluator.py` reads these environment variables. The defaults are the paper's protocol.

| Variable | Default | Meaning |
| --- | --- | --- |
| `GPUKERNEL_SERVER_INFO` | `../server_info.yaml` | Server list |
| `GPUKERNEL_EVAL_MODE` | `kernelbench_official` | Evaluation mode (the only one the server supports) |
| `GPUKERNEL_PROGRAM_FORMAT` | `auto` | How to read the candidate: `class ModelNew`, a reference-style `class Model` (renamed to `ModelNew`), or `def custom_kernel(data)`. `auto` detects it |
| `GPUKERNEL_BACKEND` | `cuda` | KernelBench backend (see below) |
| `GPUKERNEL_PRECISION` | `fp32` | Precision for inputs and weights |
| `GPUKERNEL_TIMING_METHOD` | `cuda_event` | KernelBench timing method |
| `GPUKERNEL_NUM_CORRECT_TRIALS` | 5 | Random inputs checked for correctness |
| `GPUKERNEL_NUM_PERF_TRIALS` | 100 | Timed runs |
| `GPUKERNEL_SEED_NUM` | 42 | Seed for the inputs |
| `GPUKERNEL_GPU_ID` | 0 | GPU index on the server |
| `GPUKERNEL_INCLUDE_REFERENCE_TIMING` | off | Also time the reference model and report the speedup |
| `GPUKERNEL_REF_NUM_TRIALS` | 100 | Timed runs of the reference model when reference timing is on |
| `GPUKERNEL_VERBOSE` | off | Print client and server progress |

### Backend

With the `cuda` backend, KernelBench loads the candidate with `exec`. Triton's
`@triton.jit` needs the kernel's source file, so candidates write their Triton
kernels to a module file and import it. The discovered ConvDiv kernel
(Listing 4 in Appendix E.2 of the paper) does this. `GPUKERNEL_BACKEND=triton` makes
KernelBench load the candidate from a temporary file instead. The paper's
runs used `cuda`.

## Discovery settings

The launch scripts use a 3600 s outer timeout per evaluation (`EVAL_TIMEOUT`)
and set `GPUKERNEL_VERBOSE=1` unless it is already set. Before starting, they
check that a server from `server_info.yaml` answers on `/health`.
