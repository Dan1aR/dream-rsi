# dream-rsi

An agent skill for running iterative empirical research with the Dream-RSI protocol,
for any problem that has one measured value: hyperparameters, prompts, configs, code,
ablations.

A study is a tree of attempts. Each attempt has an input (a parameters file, a
program, a patch, a directory) and one measured value. A **lead** agent decides where
the study spends its measurements - a new direction from the baseline, a refinement of
a branch, or the repair of a failed attempt - and keeps W workers busy in parallel.
Each **worker** reads the complete history, proposes one new attempt, implements it,
measures it to the end and records the result.

## Contents

| File | Contents |
| --- | --- |
| `SKILL.md` | The protocol: the lead prompt and the worker prompt |
| `scripts/rsi_tree.py` | The study tree CLI. It stores and summarizes attempts; it never measures anything |

## Installation

The repository root is the skill directory. For Claude Code:

```bash
git clone git@github.com:Dan1aR/dream-rsi.git ~/.claude/skills/dream-rsi
```

or link an existing checkout:

```bash
ln -s /path/to/dream-rsi ~/.claude/skills/dream-rsi
```

`rsi_tree.py` needs Python 3.9+ and nothing else.

## Tool

```bash
T=~/.claude/skills/dream-rsi/scripts/rsi_tree.py
python3 $T study init --problem problem.md --metric acc --goal max \
    --input params.json --workers 4 --budget 40 --patience 3
# write study/attempt_0000/{proposal.md,params.json}, measure the baseline, then:
python3 $T study record 0 --value 0.741 --evidence "logs/baseline.log"
python3 $T study add --parent 0          # open an attempt -> 0001 study/attempt_0001
python3 $T study brief 1                 # the complete prompt for its worker
python3 $T study summary                 # branch trajectories, budget, free slots
python3 $T study show                    # the tree
```

`python3 scripts/rsi_tree.py --help` lists every command.

## Origin

The protocol adapts the prompts of *DREAM-RSI: Recursive Self-Improvement Through
Evolving Worlds* (Appendix B, Listings 1 and 2) from the authors' code release, of
which this repository is a fork. In the paper the exploration policy is Python code
learned offline for one benchmark; here the lead agent is the policy. The original
code is in this repository's git history.

## License

Apache License 2.0, see [LICENSE](LICENSE). The prompts in `SKILL.md` are modified
versions of prompts from the original DREAM-RSI code, Copyright 2026 Google LLC,
licensed under the Apache License 2.0.
