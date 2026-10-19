# Resumable reference LongBench batches

`scripts/run_sparse_longbench_batch.py` uses only the Python standard library.
It launches the unchanged `run_sparse_longbench.py` sequentially with
`--backend reference --limit-cases 0`. Each evaluation runs original attention,
full reference attention, and sparse reference attention on all 80 frozen
examples, at context 4096 with the first-reference policy. Each arm retains the
single-model runner's warmup pass. There is no new scoring or attention code.

## Preview and launch the initial six evaluations

Preview on a CPU machine with both repositories present. No imports of torch,
GPU access, downloads, output-directory creation, or result writes occur:

```bash
cd ~/Sparge_Analysis
python -B scripts/run_sparse_longbench_batch.py \
  --preset initial-4k --keep-fraction 0.75 0.90 \
  --backend reference --context 4096 --local-blocks 1 \
  --batch-name paper_keep075_090 --dry-run
```

The preset selects exactly these frozen-manifest entries:

| Checkpoint | Immutable revision |
| --- | --- |
| `adpretko/celerity-271m-4k-ad0p0` | `be032b277561058f3215d0862ba282c164e73d35` |
| `adpretko/celerity-504m-4k-ad0p0` | `dc839b30ba3760c9cf66db3172e6df182108d678` |
| `adpretko/celerity-906m-4k-ad0p0` | `f9777bdd0f7ad72c59a86fbb11257d2a0e920bdc` |

On Hyperstack, first use the existing validated BF16 LongBench environment
(PyTorch 2.5.1, Transformers 4.44.2, datasets 2.19.2). Place the new wrapper in
the existing Sparge_Analysis checkout; keep the sibling Perplexity_LongBench
checkout and frozen manifest unchanged. No dependency installation is needed.
Run from a persistent terminal such as an existing tmux session:

```bash
cd ~/Sparge_Analysis
python -B scripts/run_sparse_longbench_batch.py \
  --preset initial-4k --keep-fraction 0.75 0.90 \
  --backend reference --context 4096 --local-blocks 1 \
  --batch-name paper_keep075_090 --gpu 0
```

The wrapper uses the same Python executable for every child. `--gpu` identifies
one device via `CUDA_VISIBLE_DEVICES`; comma-separated device lists are rejected.
The unchanged evaluator may fetch its pinned checkpoint/data if absent from the
GPU machine's cache. Neither preview nor the CPU tests fetch anything.

## Resume and failures

Rerun the **identical launch command**. No separate resume flag is needed.
Each invocation creates new batch records and revalidates candidate results
before reuse. Completed evaluations can also be reused by another batch name
when their entire scientific identity matches. Existing standalone or archived
results are not automatically imported.

An evaluation identity includes the entire model manifest entry (including its
immutable revision), raw manifest SHA256, context, reference policy, all sparse
settings including block size and local blocks, and hashes of the scorer,
utilities, v2 runner, single-model launcher, adapter, attention implementation,
validation helper, and package initializer. Changing these settings or sources
requires a new batch name. Original result provenance must independently match
the identity. Runtime environment and tokenizer identities remain recorded in
the evaluator's provenance; a different current Python/GPU environment alone
does not invalidate an otherwise compatible historical result.

Reuse requires `summary.json` status `complete`, all three arm artifacts, exactly
80 frozen examples in order, matching sample/case identities, paired scoring
windows and answer-token counts, internally consistent per-example and total
NLL/PPL, complete backend/layer coverage, no fallback or backend failures, and
a passing full-reference control. The original **0.02 nat/token** threshold is
unchanged. CSV distinguishes `passed_exact` from `passed_within_0.02`.
Inspect any non-exact reference control before using it in the paper.

An incomplete, corrupt, or previously failed attempt is preserved and rejected
for reuse, with the reason logged. A new directory is used for the retry.
An interrupted evaluation restarts all three arms; there is no mid-arm or
mid-example resume. If the evaluator completed before its parent died, its
complete artifacts can be verified and recovered without a terminal wrapper
status file. An existing failed/interrupted wrapper status disallows reuse.

Default behavior stops on the first failed evaluation. `--continue-on-error`
allows later evaluations to proceed sequentially. Exit codes: 0 for all complete,
1 for failed/incomplete work, 2 for configuration/preflight errors, 130 for a
handled interruption. Pending rows have blank scientific metrics.

Ctrl-C/SIGTERM stops and waits for the active child. A Linux file lock serializes
batch invocations in this checkout. The child inherits the lock, so a killed
parent cannot leave an unprotected child using the GPU. After a hard kill,
wait for that child to finish or stop it before restarting; do not delete the
lock file. The lock does not coordinate other checkouts, other GPU tools, or
manual invocations of the single-model evaluator.

## Outputs

All new artifacts stay under `results/sparse_longbench/`:

```text
evaluations/<checkpoint>_keep-<fraction>_<identity>/
  attempt_<UTC timestamp>_<unique suffix>/
    request.json       # immutable scientific identity
    launch.json        # exact command and selected GPU
    stdout.log         # child stdout and stderr
    status.json        # terminal attempt status and process wall time
    result/            # unchanged evaluator's provenance, cases, all arms
batches/<batch-name>/
  campaign.json       # immutable campaign configuration
  invocation_<UTC timestamp>_<unique suffix>/
    plan.json          # jobs, paths, Python, GPU, batch-driver hash
    events.jsonl       # flushed event journal, including rejected reuse
    progress_0001.csv  # new snapshot after each evaluation/reuse
    ...
    results.csv        # consolidated final snapshot for this invocation
    status.json        # invocation outcome and counts
```

Every CSV contains one row for every requested checkpoint/fraction, including
failed and pending rows. Every invocation and progress snapshot has a fresh
filename. Nothing replaces completed results, prior CSVs, or earlier attempts.
After an unhandled hard kill, the event journal and last progress CSV remain;
the next invocation reconstructs completion from the evaluation artifacts.

CSV preserves total NLL and token counts as well as NLL/token and PPL. It never
averages PPL across models. `evaluation_seconds` sums measured scorer wall time
over the three arms, excluding warmup and model loading. `warmup_seconds` is
separate, and `sparse_evaluation_seconds` covers just the sparse measured arm.
GPU peaks are the maximum of the three measured arm peaks, in bytes. Process
wall time, including loading and warmup, is in the attempt status. Execution
coverage is verified completed backend coverage; the JSON CSV field retains
each arm's original execution counters. Result paths point to the original
evaluator artifacts, including runtime provenance.

The reference backend computes dense attention scores with a sparse mask.
Its timing and memory measurements **do not establish sparse-attention
acceleration or SpargeAttention speedups**.

## Explicit subsets and future dropout studies

Inspect exact IDs and manifest metadata without launching anything:

```bash
python -B scripts/run_sparse_longbench_batch.py --list-models
```

This prints all 189 manifest entries as CSV, including `supported=False` for
the 1.81B family. Selecting an unsupported or unknown entry fails the entire
preflight with its ID; it is never silently dropped. There is no `--all` mode.
For a subset, use `--model ID1 ID2 ...` (repeatable) or `--models-file PATH`
with one exact manifest ID per line and optional whole-line `#` comments.
Do not combine explicit selection with the preset. Fractions accept multiple
values or repeated flags; omitted fractions default to 0.75 and 0.90.

The manifest explicitly records training context, model type, vocabulary size,
and revision. Size and dropout labels are derived only from the actual IDs,
marked as such, and not claimed to be verified checkpoint configuration fields.
The full variant suffix is retained; ambiguous dropout configurations are
reported as `unknown`. For dropout comparisons, choose entries with identical
size and training context and check all other training conditions against the
training records. In particular, `lr`, `bs`, `tema`, and `good-lr` variants must
not be pooled with ordinary baselines merely because an attention-dropout tag
matches. The wrapper does not invent matches or aggregate comparisons.

Only reference backend, evaluation context 4096, first references, and 80 cases
are supported here. Flex, 1.81B support, and the full 189-checkpoint campaign
remain outside this implementation. Checkpoint execution still performs the
adapter's original source/layout checks; catalog support is only family screening.

## CPU verification

```bash
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/lb_ppl_cpu/bin/python -B \
  -m unittest discover -s tests_sparse -v
```

The batch tests use synthetic outputs, mocked child execution, temporary
directories, and the local frozen manifest. They exercise selection, forwarding,
resume, corruption rejection, output preservation, locks, interruption cleanup,
CSV calculations, and a dry-run subprocess with site packages disabled.
