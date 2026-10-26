---
name: dream-rsi
description: Run an iterative empirical research study with the Dream-RSI protocol - a tree of measured attempts (one input, one measured value), a lead that decides where to spend the measurement budget and keeps parallel workers busy, and workers that read the complete history before proposing. Use when the user wants to improve or investigate something through repeated measurement (hyperparameters, prompts, configs, code, ablations) and asks to run it as a study, as research, or with Dream-RSI.
---

# Dream-RSI

Dream-RSI (*Recursive Self-Improvement Through Evolving Worlds*) splits empirical
discovery into two roles. A **worker** reads the complete history, proposes one new
attempt, implements it and measures it. An **exploration policy** decides where the
study spends its next measurements - a new direction from the baseline, a refinement
of a branch, or the repair of a failed attempt - and keeps the parallel workers busy.
The paper learns that policy as code for one benchmark. Here **the lead agent is the
policy**, and the study is any problem with one measured value.

The two prompts below are the paper's prompts (Appendix B, Listings 1 and 2) adapted
to general-purpose research and to `rsi_tree.py`. Read both completely before you
start: the lead prompt governs your decisions; the worker prompt is what every worker
receives, and you follow it yourself whenever you run an attempt.

If your task is a brief rendered by `rsi_tree.py brief`, you are a worker: follow that
brief only.

## Tool

`scripts/rsi_tree.py` next to this file (Python 3.9+, standard library only) stores the
study. It never runs or measures anything: agents measure in their own working
locations and record the result. Every command takes the study directory first:
`python3 <skill dir>/scripts/rsi_tree.py <study> <command> ...`.

| Command | Used by | Effect |
| --- | --- | --- |
| `init --problem F --metric M --goal max\|min --input NAME --workers W --budget N --patience P` | lead | Create the study and the open baseline `attempt_0000` |
| `add --parent ID` | lead | Open an attempt under a measured node; prints its id and directory |
| `brief ID` | lead | Print the complete worker prompt for an open attempt |
| `record ID --value X --evidence E` | worker, lead | Record a measured value |
| `record ID --error TEXT --evidence E` | worker, lead | Record a failed attempt |
| `handoff ID --run R` | worker | Register a started measurement that the lead collects when it ends |
| `close ID --reason R` / `reopen ID` | lead | Stop or resume extending a node and its subtree |
| `summary` | lead | Budget, slots, best result, every branch's trajectory, what can be extended |
| `show` | all | The whole tree: value, change vs parent, proposal title, best, closed marks |
| `export` | all | All attempts as CSV |

A study directory holds `study.json`, `problem.md`, an optional `guidance.md`, and one
`attempt_NNNN/` per attempt with `meta.json`, `proposal.md`, the input, `eval/score.json`
after a successful measurement and `error.txt` after a failure. `attempt_0000` is the
baseline; each of its children opens a **direction**; a direction with all its
descendants is a **branch**.

The tool enforces:

- a parent must be recorded before a child is opened;
- at most W attempts are open at once, at most one per branch (any number of new
  directions);
- the budget counts every attempt except the baseline, failed ones included;
- a closed node and its subtree cannot be extended until reopened;
- a value is recorded only with `proposal.md`, the input and evidence, and never for
  an input identical to one already measured successfully;
- a record is final; `--force` exists only for the lead's correction below.

A new attempt's input starts as a copy of its parent's input.

## Setting up a study (lead)

1. Agree with the user on the objective, the one metric and whether higher or lower is
   better, the measurement procedure, what an attempt's input is (a parameters file, a
   program, a patch, a directory), where attempts work, and W, budget and patience.
2. Write the problem file. It states the problem; the metric and its goal; the exact
   measurement procedure, run to completion (shortened runs, subsets or extrapolation
   only if the procedure itself defines them); what the input is and how it is
   applied; where each attempt works and measures (by default `attempt_NNNN/work/`; for
   shared resources, how parallel attempts avoid colliding); whether the worker waits
   for its measurement or hands it off (for runs too long for an agent to wait on),
   and for handed-off runs how to tell that a run finished and where its value is;
   what an attempt may change; and other constraints. The metric and the procedure are fixed for the
   whole study: changing either makes earlier values incomparable, so start a new
   study instead.
3. Run `init`. Measure the baseline yourself: write `attempt_0000/proposal.md` (what
   the baseline is) and its input, measure as the problem file prescribes, and
   `record 0000`.

## Resuming a study (lead)

The study directory holds the whole state. In a new session, or after you lose
context, read this skill completely again, then the problem file, `guidance.md` if it
exists, `summary` and `show`. `summary` lists every open attempt as handed off or
with its worker. For a handed-off attempt, find out whether its run is still going,
finished, or lost, and continue as for any handed-off run; a lost run is an
environment failure. An attempt with its worker from an earlier session has no
worker any more: start a new worker on it with the same `brief`. Then hold a
decision round.

## Lead prompt (adapted from Listing 2)

You are the exploration policy of this study. You decide where the study spends its
measurements and when it stops; workers decide what each attempt tries. Do not solve
the problem yourself inside a decision and do not hand workers ideas: a worker's
assignment is its parent, its brief, and the shared guidance.

### Objective: quality, work, and parallelism

Each attempt costs one unit of the budget, whether it succeeds or fails. Reach the
best measured value with few attempts, and use the W parallel slots: a decision that
opens one attempt while other slots stay idle wastes time the study cannot recover.
Therefore choose only promising attempts, but batch independent promising attempts
whenever possible.

A local implementation failure does not by itself prove that its parent direction is
poor. Weigh recovery value against new directions and ordinary refinements while
keeping batches parallel.

### Measured evidence

An attempt with a recorded value is a successful measurement. A *successful anchor*
below means the best recorded value of a branch. Derive every decision statistic from
recorded measurements (`summary`, `show`, the attempts' files), not from what
proposals or workers claim. Before you build on a new best value, open its evidence
and check that it was measured as the problem file prescribes. If it was not, replace
the record with `record ID --force --error "<what was wrong>" --evidence "<what you
checked>"`; this is the only use of `--force`. Record a value only from a
measurement you made yourself or from a handed-off run you checked to its end, and
never edit a worker's files.

### Required branch trajectory and failure interpretation

For each opened branch, reconstruct its ordered trajectory, not only its latest result
or best value: successful anchor, value trend, regressions, failure/repair sequence,
and explored depth.

Before closing or deprioritizing a failed attempt, classify it as hard-unrecoverable,
repairable implementation failure, weak-but-underexplored, or repeatedly unpromising
after sufficient valid evidence. Bugs, wrong or untuned parameters, resource limits
(memory, time) and configuration or shape errors are normally repairable. Do not infer
that an idea fails from one such error. An environment failure - unavailable
infrastructure, quota, a broken dependency - is evidence about the environment, not
about the idea: repair the environment and retry. Classify the current failure
episode: a later successful result reopens the branch and cancels closure based only
on an earlier failure.

### Required batch decision loop

A decision round happens at the start and every time one or more attempts are
recorded. At each decision round:

1. Read `summary` and `show`, reconstruct trajectories, and close (`close --reason`)
   only branches or subtrees with cumulative evidence of being hard-unrecoverable or
   repeatedly unpromising.
2. Rank the candidates - a new direction from the baseline, or a child of any
   measured, unclosed node in a branch without an open attempt - using only measured
   signals: successful anchor, parent-to-child gain, complete branch trajectory,
   actual success versus failure evidence, failure recoverability, prior repair
   outcomes, explored depth, and cross-branch comparison.
3. Rank actual repairable failures and underexplored branches in deterministic queues
   using trajectory, recoverability, explored depth, repeated failures, and patience.
   A repairable failure retains eligibility unless cumulative evidence lowers its
   relative priority.
4. Build one **dynamic portfolio** batch of independent attempts that fills the free
   slots: exploitation (strong normal refinements), exploration (new directions or
   underexplored branches), and at most one recovery (a child of an actual repairable
   failure). When multiple roles are eligible, give exploration and justified recovery
   representation before filling the remaining slots by priority; adapt this to the
   measured evidence rather than fixed quotas. Recovery must not displace normal
   successful refinements or leave workers idle. Never sample randomly, and do not
   default to a singleton merely because its top candidate is clear.
5. Stop only after considering the whole portfolio: active, underexplored,
   recoverable, unopened, and remaining candidates. Do not stop while an eligible
   high-priority recovery or underexplored candidate remains; every remaining
   candidate needs an evidence-based decision to continue, reserve, or close. The
   study also stops when the budget is used up and the open attempts are recorded.

A batch may contain several new directions and at most one attempt per opened
branch, and it never contains a parent together with its child. Do not use a fixed
widen-all / deepen-all wave schedule: adapt batch composition after every recorded
result.

For each selected attempt run `add --parent <id>`, then start a worker whose complete
prompt is exactly the output of `brief <id>`: nothing added, nothing summarized.
Start the batch's workers in parallel with whatever runs agents in this environment
(subagents, OpenCode workers, other agent sessions). With one slot, or with no way to
start workers, run the attempt yourself by the worker prompt.

When a worker finishes, its attempt is recorded or handed off. If it is neither: when
the worker reached a failure, record that failure with `--error` and its transcript
or logs as evidence; when the worker was cut off (crash, limits, end of session),
start a new worker on the same attempt with the same `brief`.

A handed-off attempt stays open and keeps its slot until you record it. Wait for its
run with the environment's completion notifications or scheduled wake-ups, not by
repeated polling. When the run ends, check that it ran the procedure of the problem
file to its end, then record its value - or its failure - with the run id and the
place you read the result as evidence. The decision round follows that record.

### Hard constraints

- Every close, widen, deepen, batch, and stop decision must be explainable from the
  recorded measurements. Shallow weak values are not enough to discard a branch:
  deeper attempts can recover. A repairable latest failure must not erase its
  historical successful anchor or by itself cause permanent starvation.
- Keep all thresholds relative to the study's own measurements (the baseline, the
  anchors, the observed gains); never use absolute value cutoffs.

### Patience: fixed per study

Patience takes the place of the paper's beta knob. It is set once by `init` and does
not change during the study. `summary` marks a branch `[patience reached]` when
`patience` or more of its attempts were recorded after its best one. The mark makes
the branch a candidate for "repeatedly unpromising"; it does not close it: close the
branch only if its trajectory leaves no repairable failure or underexplored
refinement. High patience means deeper patience and weaker pruning; low patience means
earlier stagnation stops and stronger pruning, while one repairable failure is still
never an automatic closure.

### Width versus depth

Every batch decides how many slots go to new directions and how many to refinements.
Choose from evidence, not a default preference:

- many semantically distinct directions improve early while deeper refinements stall:
  open more directions and hold depth;
- high gains arrive late on a small, repeatable set of directions: open fewer
  directions and refine those deeper;
- all explored directions plateau after sufficient depth while meaningful direction
  classes remain uncovered: open new directions;
- repeated hard, unrecoverable failures or strongly redundant directions: narrow both,
  conservatively;
- conflicting or insufficient evidence: keep a balanced portfolio and state that the
  evidence is insufficient.

Do not choose a branch merely because its id is small.

### Direction guidance (optional)

This is the paper's directional-guidance variant. After a decision round with new
records, you may rewrite `<study>/guidance.md`; `brief` puts it into every later
worker prompt. Keep it a compact catalog of directions. A direction is a
mechanism, not a paraphrase, a parameter value, a superficial change, or a claimed
result. For each direction give its status (improving, plateaued, repairable failure,
conceptual failure, untried) and the attempts and recorded values behind it.
Distinguish conceptual failure from implementation failure, never merge genuinely
different mechanisms only because they share a goal, and use only recorded evidence.

### Report

When the study stops, report the best attempt with its recorded value, its change
against the baseline, its input and evidence; one line per branch with its
trajectory; closed branches with their reasons; the budget used; and what still looks
promising. Quote values only as recorded.

## Worker prompt (adapted from Listing 1)

`brief` renders this section with the attempt's paths filled in.

<!-- worker-prompt:start -->
You must read every historical proposal before proposing or implementing a new solution.

$direction_guidance

Variables are filled in by the calling system: your attempt directory `$node_dir` (attempt `$node_id`), the study `$study_dir`, the baseline `$baseline_dir`, the problem file `$problem_file`, and your input `$input`, a path inside your attempt directory. `$node_dir` is your own attempt directory -- exclude it when scanning `attempt_*/` dirs. $assignment `$node_dir/$input` starts as a copy of your parent's input, when the parent has one.

## 1. Read the complete history first

Before proposing anything, read `$problem_file`, then every `proposal.md` under `$study_dir/attempt_*/` (the baseline included) in full -- not a sample, not just recent attempts or the current branch. For each, read its matching `eval/score.json` (and `error.txt` if it failed). `$cli $study_dir show` prints the tree that connects them. Trust the measured result over what the proposal claims about itself.

## 2. Learn from both successes and failures

For every past attempt, note the mechanism and how it did. For failures, figure out *why*: a flawed core idea, or a good idea let down by a bug, bad parameters, or an implementation slip? Don't repeat the former. The latter is worth retrying -- but only once you've actually located the bug in the code (not just guessed from the proposal), and only with a specific fix in hand.

## 3. Don't converge into a local optimum

Look at the shape of what's been tried. If most attempts cluster around small variations of one mechanism with flattening returns, that's a local optimum -- resist proposing another small tweak there. Deliberately favor a structurally different mechanism or an untried combination over a safer marginal refinement. Exploration diversity matters as much as the next incremental gain.

## 4. Propose and implement

The new idea must be a genuinely new mechanism, a new combination of previously-successful pieces, or a targeted fix to a specific bug found in step 2 -- never a repeat or rename of something already tried. Implement it in `$node_dir/$input`. Write `$node_dir/proposal.md` (mechanism, evidence from history, why it's not a repeat, expected benefit/risk) before you measure. Don't claim it works, is correct, or beats the best result until it's actually measured.

## 5. Measure to the end and record

Measure exactly as `$problem_file` prescribes and run the measurement to completion: no shortened runs, subsets, or extrapolation unless the procedure itself defines them. Then record the measured value:

    $cli $study_dir record $node_id --value <number> --evidence "<command, log path or run id>"

If the attempt failed -- it did not run, crashed, or broke a constraint of the problem -- record the failure instead, with what failed and why, as far as you located it:

    $cli $study_dir record $node_id --error "<what failed and why>" --evidence "<log path or command>"

Record only a value this attempt measured -- never an estimate, a partial run, or a value from another attempt. The tool refuses a value whose input is identical to one already measured: that would be a repeat.

If `$problem_file` says measurements are handed off, start the measurement as it prescribes, check that it started correctly, and register it instead of waiting; the lead records the value when the run ends:

    $cli $study_dir handoff $node_id --run "<run id, and where its result will appear>"

## Files

Write only `$node_dir/proposal.md`, `$node_dir/$input`, your measurement files (logs, outputs) under `$node_dir/`, and the working location `$problem_file` assigns to your attempt (`$node_dir/work/` if it assigns none). Everything else is read-only. Of the `$cli` commands, run only the read-only `show`, `summary` and `export`, and `record` or `handoff` for your own attempt. If `$node_dir` already holds work beyond the copied input -- a proposal, changes to the input, other files -- it comes from an interrupted earlier worker on this attempt: it is not a measured result, and you reuse it only after checking it.

## Note:
Never execute pkill, kill, killall, or terminate unrelated processes.
<!-- worker-prompt:end -->
