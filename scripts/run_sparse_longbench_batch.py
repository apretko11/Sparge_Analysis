#!/usr/bin/env python3
"""Sequential, resumable reference-only LongBench campaigns (stdlib-only driver)."""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/sparse_longbench"
LB = ROOT.parent / "Perplexity_LongBench"
MANIFEST = LB / "experiments/celerity_all_4k_a6000_bf16/manifest.json"
INITIAL_MODELS = [
    "adpretko/celerity-271m-4k-ad0p0",
    "adpretko/celerity-504m-4k-ad0p0",
    "adpretko/celerity-906m-4k-ad0p0",
]
ARMS = ("original", "reference_full", "reference_sparse")
SOURCE_FILES = {
    "scorer_sha256": ("lb", "longbench_perplexity.py"),
    "utils_sha256": ("lb", "longbench_utils.py"),
    "v2_runner_sha256": ("lb", "benchmark_celerity_longbench_v2.py"),
    "adapter_sha256": ("root", "celerity_sparse/adapter.py"),
    "attention_sha256": ("root", "celerity_sparse/attention.py"),
    "launcher_sha256": ("root", "scripts/run_sparse_longbench.py"),
    "validator_sha256": ("root", "scripts/validate_sparse_gpu.py"),
    "package_sha256": ("root", "celerity_sparse/__init__.py"),
}
PROVENANCE_HASHES = tuple(SOURCE_FILES)[:6]
CSV_FIELDS = [
    "model_id", "checkpoint_revision", "model_size", "training_context",
    "checkpoint_variant", "dropout_configuration", "metadata_source",
    "evaluation_context", "reference_policy", "backend", "keep_fraction",
    "local_blocks", "manifest_sha256", "evaluation_id", "causal_pair_sparsity",
    "original_ppl", "full_reference_ppl", "sparse_ppl", "original_nll_per_token",
    "sparse_nll_per_token", "delta_nll_per_token", "ppl_change_percent",
    "original_nll_sum", "full_reference_nll_sum", "sparse_nll_sum",
    "evaluated_examples", "scored_answer_tokens", "numerical_control_status",
    "full_reference_delta_nll_per_token", "execution_coverage",
    "execution_coverage_by_arm", "evaluation_seconds", "warmup_seconds",
    "sparse_evaluation_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
    "status", "reused", "error", "result_path",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(value):
    # Same serialization as the unchanged v2 scorer's case/sample digests.
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str,
                                     ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)


def save_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]


def literal(path, name):
    """Read constants without importing torch, datasets, or model-loading code."""
    for node in ast.parse(Path(path).read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name} in {path}")


def metadata(spec):
    match = re.fullmatch(r"adpretko/celerity-(\d+)m-(\d+)k-(.+)", spec["model"])
    size, variant = (match[1] + "M", match[3]) if match else ("unknown", "unknown")
    # Preserve the entire suffix: lr/bs/tema ablations are NOT matched baselines.
    dropout = re.fullmatch(r"(ad|ld|rd)(\d+p\d+)(-ild)?", variant)
    label = "unknown"
    if dropout:
        kind = {"ad": "attention", "ld": "layer", "rd": "residual"}[dropout[1]]
        label = f"{kind}={dropout[2].replace('p', '.')}; ILD={'yes' if dropout[3] else 'no'} (ID label)"
    supported = bool(match and size in ("271M", "504M", "906M")
                     and spec.get("model_type") == "celerity")
    return dict(model_size=size, checkpoint_variant=variant, dropout_configuration=label,
                metadata_source="size/variant/dropout: checkpoint ID; training context: manifest",
                supported=supported)


def load_manifest(path, lb):
    doc = read_json(path)
    cfg = doc["settings"]
    expected = dict(reference_policy="first", use_chat_template=False, dtype="bfloat16",
                    context_mode="common", common_context=4096, answer_stride=128,
                    examples_per_task=5, dataset="THUDM/LongBench",
                    scorer="longbench_perplexity.answer_conditional_ppl",
                    truncation_policy="longbench_utils.middle_truncate_tokens")
    require(doc["version"] == 2 and all(cfg.get(k) == v for k, v in expected.items()),
            "Requires the frozen BF16 4K first-reference manifest")
    require(cfg["tasks"] == literal(lb / "longbench_perplexity.py", "DEFAULT_LONGBENCH_TASKS")
            and len(cfg["tasks"]) == 16, "Expected the original 16 tasks in order")
    require(re.fullmatch(r"[0-9a-f]{40}", doc["longbench_revision"]), "Unpinned dataset revision")
    for task in cfg["tasks"]:
        indices = doc["sample_indices"][task]
        require(isinstance(indices, list) and len(indices) == 5
                and all(type(i) is int and i >= 0 for i in indices) and len(set(indices)) == 5,
                f"Missing frozen indices: {task}")
        require(re.fullmatch(r"[0-9a-f]{64}", doc["sample_id_digest"][task]), f"Missing sample digest: {task}")
    names = [m["model"] for m in doc["models"]]
    require(len(names) == len(set(names)), "Duplicate checkpoint IDs in manifest")
    return doc


def source_hashes(lb):
    hashes = {key: digest((lb if owner == "lb" else ROOT) / name)
              for key, (owner, name) in SOURCE_FILES.items()}
    require(hashes["scorer_sha256"] == literal(ROOT / "scripts/run_sparse_longbench.py", "SCORER_SHA256"),
            "Canonical scorer hash differs from the validated single-model runner")
    return hashes


def select_models(args, manifest):
    names = list(INITIAL_MODELS if args.preset else args.model or [])
    if args.models_file:
        names += [line.strip() for line in args.models_file.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
    require(names, "Select --preset initial-4k, --model IDs, or --models-file")
    require(len(names) == len(set(names)), "Duplicate checkpoint selection")
    by_name = {m["model"]: m for m in manifest["models"]}
    unknown = [n for n in names if n not in by_name]
    require(not unknown, f"Checkpoint IDs absent from frozen manifest: {unknown}")
    specs = [by_name[n] for n in names]
    unsupported = [m["model"] for m in specs if not metadata(m)["supported"]]
    require(not unsupported, f"Unsupported checkpoints (adapter supports only 271M/504M/906M): {unsupported}")
    for spec in specs:
        require(re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), f"Unpinned checkpoint: {spec['model']}")
        require(type(spec["trained_context_tokens"]) is int and spec["trained_context_tokens"] >= 4096,
                f"Training context shorter than 4096: {spec['model']}")
    return specs


def make_jobs(args, manifest):
    hashes = source_hashes(args.longbench_repo)
    jobs = []
    for spec in select_models(args, manifest):
        for keep in args.keep_fraction:
            config = dict(schema=1, model=spec, manifest_sha256=digest(args.manifest),
                          sources=hashes, evaluation_context=4096, reference_policy="first",
                          case_count=80, settings=dict(backend="reference", keep_fraction=keep,
                                                     block_size=128, local_blocks=args.local_blocks))
            jobs.append(dict(evaluation_id=sha_json(config), config=config))
    return jobs


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"Invalid nonnegative numeric field: {name}")
    return value


def close(a, b, name):
    number(a, name)
    require(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), f"Inconsistent {name}")


def verify_result(path, config, manifest):
    """Fail closed on partial, incompatible, or internally inconsistent evidence."""
    try:
        return _verify_result(path, config, manifest)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _verify_result(path, config, manifest):
    require(not (path / "failure.json").exists(), "Evaluator recorded a failure")
    summary, provenance = read_json(path / "summary.json"), read_json(path / "provenance.json")
    require(summary["status"] == "complete" and summary["paired_window_identity"] is True,
            "Missing completed, paired evaluation")
    require(set(summary["arms"]) == set(ARMS), "Expected all three reference arms")
    for key in ("model", "manifest_sha256", "evaluation_context", "reference_policy", "case_count", "settings"):
        require(provenance[key] == config[key], f"Provenance mismatch: {key}")
    for key in PROVENANCE_HASHES:
        require(provenance[key] == config["sources"][key], f"Implementation mismatch: {key}")
    require(provenance["diagnostic_subset"] is False, "Diagnostic subset is not reusable")
    require(re.fullmatch(r"[0-9a-f]{64}", provenance["tokenizer_digest"]), "Missing tokenizer identity")
    require(isinstance(provenance["environment"], dict) and provenance["environment"].get("torch")
            and provenance["environment"].get("gpu"), "Missing runtime provenance")
    require(read_json(path / "source_manifest.json") == manifest, "Source manifest mismatch")
    cases = read_json(path / "cases.json")
    require(len(cases) == 80 and sha_json(cases) == provenance["cases_sha256"], "Cases incomplete or digest changed")
    expected = [(task, i) for task in manifest["settings"]["tasks"] for i in manifest["sample_indices"][task]]
    require([(c["task"], c["example_index"]) for c in cases] == expected, "Cases differ from frozen order/indices")
    for task in manifest["settings"]["tasks"]:
        selected = [c for c in cases if c["task"] == task]
        require(sha_json([(str(c["example_id"] or ""), c["answer"]) for c in selected])
                == manifest["sample_id_digest"][task], f"Frozen sample identity mismatch: {task}")
    for case in cases:
        require(isinstance(case["prompt"], str) and isinstance(case["answer"], str)
                and case["answer"].strip(), "Malformed prompt/answer")
    all_windows, all_rows, all_layers = [], [], []
    for arm in ARMS:
        stats = summary["arms"][arm]
        standalone = read_json(path / f"{arm}_summary.json")
        require(all(stats[k] == v for k, v in standalone.items())
                and set(stats) - set(standalone) <= {"delta_nll_per_token_vs_original",
                    "delta_perplexity_vs_original", "delta_nll_per_token_vs_same_backend_full"},
                f"Conflicting arm summary: {arm}")
        rows = [json.loads(line) for line in (path / f"{arm}_examples.jsonl").read_text().splitlines()]
        require(len(rows) == stats["examples"] == 80, f"Incomplete examples: {arm}")
        for row, case in zip(rows, cases):
            require(all(row[k] == case[k] for k in ("task", "example_index", "example_id")), "Example identity mismatch")
            require(row["reference_index"] == 0 and row["reference_policy"] == "first"
                    and row["evaluation_context_tokens"] == 4096, "Example scoring configuration mismatch")
            tokens = row["tokens_scored"]
            require(type(tokens) is int and tokens > 0 and row["answer_tokens"] == tokens, "Incomplete answer tokens")
            number(row["nll_sum"], "example NLL")
            close(row["avg_nll"], row["nll_sum"] / tokens, "example avg NLL")
            close(row["perplexity"], math.exp(row["avg_nll"]), "example PPL")
        tokens = sum(r["tokens_scored"] for r in rows)
        require(stats["scored_tokens"] == tokens, "Scored token total mismatch")
        close(stats["nll_sum"], sum(r["nll_sum"] for r in rows), "total NLL")
        close(stats["avg_nll"], stats["nll_sum"] / tokens, "average NLL")
        close(stats["perplexity"], math.exp(stats["avg_nll"]), "PPL")
        windows = read_json(path / f"{arm}_windows.json")
        require(len(windows) >= 80, "Incomplete scoring windows")
        for window in windows:
            require(window["shape"][0] == 1 and len(window["shape"]) == 2
                    and type(window["shape"][1]) is int and 1 < window["shape"][1] <= 4096
                    and re.fullmatch(r"[0-9a-f]{64}", window["token_ids_sha256"]), "Malformed window identity")
            number(window["forward_seconds"], "window time")
        calls = read_json(path / f"{arm}_attention_calls.json")
        execution = stats["execution"]
        backend = "original" if arm == "original" else "reference"
        require(calls and len(calls) == execution["completed_attention_calls"]
                and execution["calls_by_backend"] == {backend: len(calls)}
                and all(c["backend"] == backend for c in calls)
                and execution["failures"] == {} and execution["dense_fallback_calls"] == 0,
                f"Incomplete backend coverage: {arm}")
        depth, remainder = divmod(len(calls), len(windows))
        require(depth > 0 and remainder == 0
                and [c["layer"] for c in calls] == list(range(depth)) * len(windows), "Incomplete layer/window coverage")
        if backend == "reference":
            for c, window in zip(calls, [w for w in windows for _ in range(depth)]):
                require(c["q_len"] == c["kv_len"] == window["shape"][1]
                        and c["dtype"] == "torch.bfloat16", "Attention shape/precision mismatch")
                for retained, eligible in (("retained_blocks", "eligible_blocks"),
                                            ("retained_causal_pairs", "eligible_causal_pairs")):
                    require(type(c[retained]) is int and type(c[eligible]) is int
                            and 0 < c[retained] <= c[eligible], "Invalid attention counts")
            for key in ("retained_blocks", "eligible_blocks", "retained_causal_pairs", "eligible_causal_pairs"):
                require(execution[key] == sum(c[key] for c in calls), f"Coverage total mismatch: {key}")
            close(execution["causal_pair_sparsity"],
                  1 - execution["retained_causal_pairs"] / execution["eligible_causal_pairs"], "causal-pair sparsity")
            omitted = sum(c["retained_blocks"] < c["eligible_blocks"] for c in calls)
            require(execution["calls_omitting_blocks"] == omitted, "Omitted-block count mismatch")
            require((omitted > 0 and execution["causal_pair_sparsity"] > 0) if arm == "reference_sparse"
                    else (omitted == 0 and execution["causal_pair_sparsity"] == 0), "Incorrect full/sparse execution")
        for key in ("measured_evaluation_wall_seconds", "warmup_and_compilation_wall_seconds",
                    "model_forward_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
            number(stats[key], key)
        all_windows.append([(w["shape"], w["token_ids_sha256"]) for w in windows])
        all_rows.append(rows)
        all_layers.append(depth)
    require(all_windows[0] == all_windows[1] == all_windows[2] and len(set(all_layers)) == 1,
            "Arms did not execute identical windows/layers")
    require(all([r["tokens_scored"] for r in rows] == [r["tokens_scored"] for r in all_rows[0]]
                for rows in all_rows), "Arms scored different targets")
    # Retain the evaluator's existing 0.02 nat/token gate; report exactness separately.
    arms = summary["arms"]
    delta = arms["reference_full"]["avg_nll"] - arms["original"]["avg_nll"]
    require(abs(delta) <= 0.02, "Full-reference numerical control failed (>0.02 nat/token)")
    exact = all(a["nll_sum"] == b["nll_sum"] for a, b in zip(all_rows[0], all_rows[1]))
    return summary, "passed_exact" if exact else "passed_within_0.02"


def job_directory(job):
    config = job["config"]
    slug = config["model"]["model"].replace("/", "--")
    keep = format(config["settings"]["keep_fraction"], ".12g")
    return OUTPUT / "evaluations" / f"{slug}_keep-{keep}_{job['evaluation_id']}"


def reusable(job, manifest):
    rejected = []
    for attempt in sorted(job_directory(job).glob("attempt_*"), reverse=True):
        try:
            require(read_json(attempt / "request.json") == job, "Attempt scientific configuration mismatch")
            if (attempt / "status.json").exists():
                require(read_json(attempt / "status.json")["status"] == "complete", "Attempt previously failed/interrupted")
            verified = verify_result(attempt / "result", job["config"], manifest)
            return attempt / "result", verified, rejected
        except (OSError, ValueError, KeyError, TypeError) as exc:
            rejected.append(dict(path=str(attempt), reason=str(exc)))
    return None, None, rejected


def command(args, job, result):
    config = job["config"]
    return [sys.executable, "-B", "-u", str(ROOT / "scripts/run_sparse_longbench.py"),
            "--model", config["model"]["model"], "--longbench-repo", str(args.longbench_repo),
            "--manifest", str(args.manifest), "--backend", "reference", "--keep-fraction",
            str(config["settings"]["keep_fraction"]), "--local-blocks", str(args.local_blocks),
            "--limit-cases", "0", "--output-dir", str(result)]


@contextmanager
def campaign_lock():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".batch.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another batch evaluator still holds results/sparse_longbench/.batch.lock") from exc
        # Do not explicitly unlock: the inherited descriptor keeps the lock alive
        # if this parent is killed while the GPU child is still running.
        yield handle.fileno()


def run_child(cmd, log_path, gpu, lock_fd):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    with log_path.open("x") as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True, pass_fds=(lock_fd,))
        try:
            return child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def result_row(job, path=None, verified=None, status="pending", error="", reused=False):
    config, spec = job["config"], job["config"]["model"]
    meta = metadata(spec)
    row = dict(model_id=spec["model"], checkpoint_revision=spec["revision"],
               training_context=spec["trained_context_tokens"],
               **{key: meta[key] for key in ("model_size", "checkpoint_variant", "dropout_configuration", "metadata_source")},
               **{key: config[key] for key in ("evaluation_context", "reference_policy", "manifest_sha256")},
               **{key: config["settings"][key] for key in ("backend", "keep_fraction", "local_blocks")},
               evaluation_id=job["evaluation_id"], status=status, error=error, reused=reused,
               result_path=str(path) if path else "")
    if verified is None:
        return row
    summary, control = verified
    arms = summary["arms"]
    original, full, sparse = [arms[a] for a in ARMS]
    row.update(original_ppl=original["perplexity"], full_reference_ppl=full["perplexity"],
               sparse_ppl=sparse["perplexity"], original_nll_per_token=original["avg_nll"],
               sparse_nll_per_token=sparse["avg_nll"],
               delta_nll_per_token=sparse["avg_nll"] - original["avg_nll"],
               ppl_change_percent=100 * (sparse["perplexity"] / original["perplexity"] - 1),
               original_nll_sum=original["nll_sum"], full_reference_nll_sum=full["nll_sum"],
               sparse_nll_sum=sparse["nll_sum"], evaluated_examples=sparse["examples"],
               scored_answer_tokens=sparse["scored_tokens"], numerical_control_status=control,
               full_reference_delta_nll_per_token=full["avg_nll"] - original["avg_nll"],
               causal_pair_sparsity=sparse["execution"]["causal_pair_sparsity"], execution_coverage=1.0,
               execution_coverage_by_arm=json.dumps({a: arms[a]["execution"] for a in ARMS}, sort_keys=True),
               evaluation_seconds=sum(a["measured_evaluation_wall_seconds"] for a in arms.values()),
               warmup_seconds=sum(a["warmup_and_compilation_wall_seconds"] for a in arms.values()),
               sparse_evaluation_seconds=sparse["measured_evaluation_wall_seconds"],
               peak_allocated_bytes=max(a["peak_allocated_bytes"] for a in arms.values()),
               peak_reserved_bytes=max(a["peak_reserved_bytes"] for a in arms.values()))
    return row


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def campaign_path(args, jobs):
    path = OUTPUT / "batches" / args.batch_name
    require(OUTPUT.resolve() == OUTPUT.absolute() and path.resolve().is_relative_to(OUTPUT)
            and not OUTPUT.is_relative_to(ROOT / "experiments"), "Unsafe output location (symlink or archive)")
    identity = dict(schema=1, jobs=jobs)
    if path.exists():
        require(read_json(path / "campaign.json") == identity,
                "Batch name already has a different scientific configuration; choose a new --batch-name")
    return path, identity


def run_campaign(args, jobs, manifest):
    campaign, identity = campaign_path(args, jobs)
    if not campaign.exists():
        campaign.mkdir(parents=True)
        save_new(campaign / "campaign.json", identity)
    invocation = campaign / ("invocation_" + stamp())
    invocation.mkdir()
    rows = [result_row(job) for job in jobs]
    save_new(invocation / "plan.json", dict(jobs=jobs, gpu=args.gpu,
             python=sys.executable, batch_driver_sha256=digest(__file__),
             manifest=str(args.manifest), longbench_repo=str(args.longbench_repo)))
    print(f"Batch records: {invocation}", flush=True)
    interrupted = False
    with (invocation / "events.jsonl").open("x") as journal:
        def event(kind, **fields):
            journal.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), event=kind, **fields)) + "\n")
            journal.flush()
            os.fsync(journal.fileno())
        try:
            for index, job in enumerate(jobs):
                # Detect edits to sources/manifest during a long campaign.
                require(source_hashes(args.longbench_repo) == job["config"]["sources"]
                        and digest(args.manifest) == job["config"]["manifest_sha256"], "Inputs changed during campaign")
                result, verified, rejected = reusable(job, manifest)
                for rejection in rejected:
                    event("reuse_rejected", **rejection)
                if result:
                    rows[index] = result_row(job, result, verified, "complete", reused=True)
                    event("reused", evaluation_id=job["evaluation_id"], result_path=str(result))
                    print(f"[{index+1}/{len(jobs)}] Reused verified result: {result}", flush=True)
                else:
                    attempt = job_directory(job) / ("attempt_" + stamp())
                    require(attempt.resolve().is_relative_to(OUTPUT.resolve()), "Unsafe attempt location")
                    attempt.mkdir(parents=True, exist_ok=False)
                    save_new(attempt / "request.json", job)
                    result = attempt / "result"  # The single-model runner must create this itself.
                    cmd = command(args, job, result)
                    save_new(attempt / "launch.json", dict(command=cmd, gpu=args.gpu))
                    event("started", evaluation_id=job["evaluation_id"], attempt=str(attempt))
                    print(f"[{index+1}/{len(jobs)}] Running {job['config']['model']['model']} keep={job['config']['settings']['keep_fraction']}; log: {attempt / 'stdout.log'}", flush=True)
                    started = time.monotonic()
                    status, error, verified = "failed", "", None
                    try:
                        code = run_child(cmd, attempt / "stdout.log", args.gpu, args.lock_fd)
                        require(code == 0, f"Evaluator exited {code}; see stdout.log")
                        require(source_hashes(args.longbench_repo) == job["config"]["sources"]
                                and digest(args.manifest) == job["config"]["manifest_sha256"], "Inputs changed during evaluation")
                        verified = verify_result(result, job["config"], manifest)
                        status = "complete"
                    except KeyboardInterrupt:
                        status, error, interrupted = "interrupted", "Interrupted; child stopped", True
                    except (OSError, ValueError) as exc:
                        error = str(exc)
                    save_new(attempt / "status.json", dict(status=status, error=error,
                             process_wall_seconds=time.monotonic() - started))
                    rows[index] = result_row(job, result, verified, status, error)
                    event(status, evaluation_id=job["evaluation_id"], result_path=str(result), error=error)
                    if error:
                        print(f"{status}: {error}", file=sys.stderr, flush=True)
                # Immutable progress snapshots; no completed CSV is ever replaced.
                write_csv(invocation / f"progress_{index+1:04d}.csv", rows)
                if interrupted or (rows[index]["status"] != "complete" and not args.continue_on_error):
                    break
        except KeyboardInterrupt:
            interrupted = True
            event("interrupted")
        finally:
            write_csv(invocation / "results.csv", rows)
            state = "complete" if all(r["status"] == "complete" for r in rows) else "interrupted" if interrupted else "incomplete"
            save_new(invocation / "status.json", dict(status=state, counts=dict(Counter(r["status"] for r in rows))))
            print(f"CSV: {invocation / 'results.csv'}", flush=True)
    return 0 if state == "complete" else 130 if interrupted else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["initial-4k"], help="the three requested 4K ad0p0 baselines")
    p.add_argument("--model", nargs="+", action="extend", help="exact frozen checkpoint IDs; repeatable")
    p.add_argument("--models-file", type=Path, help="one exact checkpoint ID per line; # comments allowed")
    p.add_argument("--keep-fraction", nargs="+", action="extend", type=float, help="default: 0.75 0.90")
    p.add_argument("--backend", choices=["reference"], default="reference")
    p.add_argument("--context", type=int, choices=[4096], default=4096)
    p.add_argument("--reference-policy", choices=["first"], default="first")
    p.add_argument("--local-blocks", type=int, default=1)
    p.add_argument("--longbench-repo", type=Path, default=LB)
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--batch-name", help="stable name for resume; changed configuration requires a new name")
    p.add_argument("--gpu", default="0", help="one CUDA_VISIBLE_DEVICES index or GPU UUID")
    p.add_argument("--dry-run", action="store_true", help="verify/print plan without writes, imports of torch, or downloads")
    p.add_argument("--list-models", action="store_true", help="print frozen catalog as CSV, including unsupported entries; never run")
    p.add_argument("--continue-on-error", action="store_true", help="default stops on first failure")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        args.longbench_repo, args.manifest = args.longbench_repo.resolve(), args.manifest.resolve()
        manifest = load_manifest(args.manifest, args.longbench_repo)
        if args.list_models:
            fields = ["model", "revision", "trained_context_tokens", "model_type", "vocab_size",
                      "model_size", "checkpoint_variant", "dropout_configuration", "metadata_source", "supported"]
            writer = csv.DictWriter(sys.stdout, fieldnames=fields)
            writer.writeheader()
            for spec in manifest["models"]:
                writer.writerow({**spec, **metadata(spec)})
            return 0
        require(not (args.preset and (args.model or args.models_file)), "Use the preset OR explicit checkpoint selection")
        require(args.batch_name and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.batch_name), "Provide a simple --batch-name")
        require(re.fullmatch(r"\d+|GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9/-]+", args.gpu), "--gpu must identify exactly one GPU")
        require(args.local_blocks >= 1, "--local-blocks must be at least 1")
        args.keep_fraction = args.keep_fraction or [0.75, 0.90]
        require(all(math.isfinite(k) and 0 < k < 1 for k in args.keep_fraction), "Keep fractions must be finite and strictly between 0 and 1")
        require(len(set(args.keep_fraction)) == len(args.keep_fraction), "Duplicate keep fractions")
        jobs = make_jobs(args, manifest)
        campaign_path(args, jobs)
        print(f"{len(jobs)} evaluations; reference backend; 4096 context; all 80 frozen examples; local_blocks={args.local_blocks}.", flush=True)
        print("Dense masked reference timings do not establish sparse-attention acceleration.", flush=True)
        if args.dry_run:
            for job in jobs:
                result, _, rejected = reusable(job, manifest)
                for item in rejected:
                    print(f"REJECT REUSE: {item['path']}: {item['reason']}")
                print(f"{'REUSE' if result else 'RUN'} {job['config']['model']['model']} @ {job['config']['model']['revision']} keep={job['config']['settings']['keep_fraction']} id={job['evaluation_id']}")
                if not result:
                    print("  CUDA_VISIBLE_DEVICES=" + args.gpu + " " + shlex.join(command(args, job, job_directory(job) / "attempt_NEW" / "result")))
            return 0
        def terminate(signum, frame):
            raise KeyboardInterrupt
        old_handler = signal.signal(signal.SIGTERM, terminate)
        try:
            with campaign_lock() as args.lock_fd:
                return run_campaign(args, jobs, manifest)
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, SyntaxError) as exc:
        print(f"Batch error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
