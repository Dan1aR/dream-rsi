# Benchmark tasks

Every task directory has the same three files:

| File | Role |
| --- | --- |
| `<task>.txt` (`<task>.triton` for kernels) | Problem statement shown to the agent |
| `init_program.py` | Initial program. The agent edits the code between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` |
| `evaluator.py` | `evaluate(program_path) -> dict`. The returned `combined_score` is the task score, higher is better. A failed candidate scores 0 |

The launch scripts in `scripts/` pass these files to `dream_rsi` with
`--problem`, `--evaluator` and `--init-program`. To add a task, provide the
same three files.

Not all task files are in this repository:

- `sums_diffs`, `autocorrelation_third`, `circle_packing_26`,
  `circle_packing_32` and `lasso_path` are SimpleTES tasks. Their files are
  licensed under the GNU AGPL-3.0, so `bash scripts/fetch_tasks.sh` downloads
  them from the pinned SimpleTES commit into the task directories (they are
  git-ignored). For `lasso_path`, `README.md` and `setup.sh` are ours and are
  shipped.
- For the `kernelbench/kb_*` tasks, `init_program.py` is the unmodified
  KernelBench problem (MIT License) and lives in
  [`../third_party/kernelbench/<task>/`](../third_party/kernelbench/). The
  problem statements and evaluators are in `kernelbench/<task>/`.

| Task | Paper | `combined_score` | Evaluator timeout (s) | Explore eval timeout (s) | Rounds T |
| --- | --- | --- | --- | --- | --- |
| `sums_diffs` | Sum–Difference (Table 1) | C(A), see below | 180 | 240 | 10 |
| `autocorrelation_third` | Autocorrelation (Table 1) | 1.4556427953745406 / C₃ | 70 | 120 | 10 |
| `circle_packing_26` | Circle Packing (Table 1) | Sum of radii | 530 | 600 | 10 |
| `circle_packing_32` | Circle Packing, n = 32 (Appendix B.3) | Sum of radii | 530 | 600 | 10 |
| `lasso_path` | Lasso path (Section 4.3) | 1 / geomean(solve time in ms) | 600 | 600 | 5 |
| `kernelbench/kb_*` | VGG16, LayerNorm, ConvDiv, ConvMax (Section 4.1) | 1 / latency in ms | – | 3600 | 9 |

There are two timeouts:

- The **evaluator timeout** limits one run of the candidate program inside
  `evaluator.py`. The problem statement tells the agent this limit. You can
  override it with `EVALUATOR_TIMEOUT_SECONDS`.
- The **explore eval timeout** limits the whole `evaluate()` call made by
  DREAM-RSI. It is set a little higher so that process startup and
  verification fit inside it. The launch scripts take it from `EVAL_TIMEOUT`.

The math and Lasso evaluators run each candidate in a subprocess with a memory
limit. They read two more environment variables:

- `EVALUATOR_CONCURRENT_PROCESSES` (64; 256 for Lasso) is the number of
  evaluations expected to share the machine. Each one's memory limit is
  derived from it.
- `EVALUATOR_OS_BUFFER_PERCENT` (0.10) is the share of memory kept back for
  the OS.

## Mathematical tasks

The formal definitions are in Appendix B.3 of the paper. The candidate
program must define `run_code()`.

- **`sums_diffs`**: `run_code()` returns a set of integers A, or `(A, claimed_c)`.
  The score is C(A) = log(|A+A|/|A|) / log(|A−A|/|A|). After deduplication, A
  must hold 2 to 512 integers in [−10⁶, 10⁶]. Both ratios must be greater
  than 1. The evaluator recomputes C(A) from A.
- **`autocorrelation_third`**: `run_code()` returns the values of f on a uniform
  grid over [−1/4, 1/4]. Optionally it also returns its own C₃, a loss and the
  grid size. The evaluator computes
  C₃ = 2·n·max|f ∗ f| / (∑f)², which is lower-is-better. The score is
  1.4556427953745406 / C₃, with AlphaEvolve's C₃ as the reference point.
  Table 1 of the paper reports C₃ itself. If the program reports its own C₃,
  the value must be within 1e-3 of the recomputed one, and the reported value
  is then used for the score.
- **`circle_packing_26`, `circle_packing_32`**: `construct_circles()` returns an
  (n, 3) array of (x, y, r). The circles must lie in the unit square and must
  not overlap (tolerance 1e-12). The score is the sum of radii. Table 1 uses
  n = 26.

These evaluators need `numpy` and `psutil`. Candidate programs run in the
evaluator's interpreter and often import `scipy`, so the launch scripts check
for all three (`pip install -e ".[tasks]"`).

## Lasso path and KernelBench

See [`lasso_path/README.md`](lasso_path/README.md) and
[`kernelbench/README.md`](kernelbench/README.md). The Lasso task needs `g++`
and Eigen. The kernel tasks need a GPU evaluation server.

## Credits

- The mathematical tasks and their evaluators are adapted from SimpleTES
  (Ye et al., 2026).
- The Lasso task follows the SimpleTES protocol.
- The kernel tasks are problems from KernelBench (Ouyang et al., 2025) and are
  scored with KernelBench's official evaluation code.
