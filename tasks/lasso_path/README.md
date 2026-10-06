# Lasso regularization path (algorithm engineering)

The agent edits a C++ solver (`CPP_CODE` inside `init_program.py`) for the full
Gaussian Lasso path. Section 4.3 of the paper describes the task.

| File | Role |
| --- | --- |
| `lasso_path.txt` | Problem statement shown to the agent |
| `init_program.py` | Starting solver: glmnet-style coordinate descent with strong rules. It uses the covariance method when p < 500 and the naive method otherwise |
| `evaluator.py` | Discovery-time score: compiles the solver and times it on 17 synthetic problems. Score is `1 / geomean(solve time in ms)`. The score is 0 unless every problem's objective is within 1e-6 of scikit-learn's |
| `generate_results.py` | Held-out benchmark (Figure 4): compares two solver files against `sklearn.linear_model.lasso_path` |
| `setup.sh` | Downloads Eigen 3.4.0 into `eigen/` and checks for `g++` |

## Setup

```bash
bash tasks/lasso_path/setup.sh
```

The evaluator compiles with `g++ -O3 -march=native -std=c++17 -I tasks/lasso_path/eigen`.
If `eigen/` is missing, it falls back to `/usr/include/eigen3`.

These environment variables tune the evaluator:

- `EVALUATOR_TIMEOUT_SECONDS`: default 600.
- `EVALUATOR_CONCURRENT_PROCESSES`: default 256.
- `EVALUATOR_OS_BUFFER_PERCENT`: default 0.10.

## Held-out evaluation (Figure 4)

The six held-out datasets are Gisette, RCV1, DNA, Leukemia, Colon Cancer and
Duke Breast Cancer. The script downloads them with `libsvmdata`:

```bash
pip install libsvmdata
cd tasks/lasso_path
python generate_results.py --init init_program.py --tte /path/to/discovered_program.py \
    --no-synthetic --no-genomics
```

`--tte` is the solver being evaluated, e.g. the best `program.py` from an
explore workspace. `--init` is the reference solver. Leaving out
`--no-synthetic` adds the Toeplitz and glmnet-style synthetic designs.

Leaving out `--no-genomics` adds the TCGA/Kaggle RNA-seq matrices. The script
looks for them in `--genomics-dir` (or `$GENOMICS_DATA_DIR`). Those files are
not shipped with this repository.
