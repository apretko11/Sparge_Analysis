#!/usr/bin/env python3
"""One-checkpoint matched first-reference diagnostic, using the unchanged v2 scorer."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from celerity_sparse import CeleritySparseAdapter, SparseSettings
from validate_sparse_gpu import environment, digest

SCORER_SHA256 = "b58da9adce20542e832bff5d35ede48d205daf2774b399bcca60184b152814ec"


def save_new(path, document):
    with path.open("x") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


class RecordedModel(torch.nn.Module):
    """Observe actual scorer inputs and time forwards; do not construct windows."""
    def __init__(self, model):
        super().__init__()
        self.wrapped = model
        self.records = []

    def forward(self, *args, **kwargs):
        ids = kwargs["input_ids"]
        fingerprint = hashlib.sha256(ids.detach().cpu().numpy().tobytes()).hexdigest()
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.wrapped(*args, **kwargs)
        torch.cuda.synchronize()
        self.records.append(dict(shape=list(ids.shape), token_ids_sha256=fingerprint,
                                 forward_seconds=time.perf_counter() - start))
        return out


def validate_manifest(manifest, model_name):
    cfg = manifest["settings"]
    if (manifest["version"] != 2 or cfg["reference_policy"] != "first"
            or cfg["use_chat_template"] or cfg["dtype"] != "bfloat16"
            or cfg["context_mode"] != "common" or cfg["common_context"] != 4096
            or cfg["answer_stride"] != 128 or cfg["examples_per_task"] != 5
            or cfg["dataset"] != "THUDM/LongBench"
            or cfg["scorer"] != "longbench_perplexity.answer_conditional_ppl"
            or cfg["truncation_policy"] != "longbench_utils.middle_truncate_tokens"):
        raise ValueError("This launcher requires the frozen BF16 4K first-reference manifest")
    from longbench_perplexity import DEFAULT_LONGBENCH_TASKS
    if cfg["tasks"] != DEFAULT_LONGBENCH_TASKS:
        raise ValueError("Expected the original 16 tasks, in original order")
    for task in cfg["tasks"]:
        indices = manifest["sample_indices"].get(task)
        if (not isinstance(indices, list) or len(indices) != 5
                or len(set(indices)) != 5 or any(type(i) is not int or i < 0 for i in indices)
                or len(manifest["sample_id_digest"].get(task, "")) != 64):
            raise ValueError(f"Missing frozen sampling: {task}; refusing to resample")
    if len(manifest["longbench_revision"]) != 40:
        raise ValueError("Dataset revision is not an immutable SHA")
    matches = [m for m in manifest["models"] if m["model"] == model_name]
    if len(matches) != 1 or len(matches[0]["revision"]) != 40:
        raise ValueError("Requested checkpoint must have one frozen manifest revision")
    if matches[0]["trained_context_tokens"] < 4096:
        raise ValueError("Checkpoint training context is shorter than 4K")
    return matches[0]


def require_gpu_validation(path, spec, settings):
    report = json.loads(path.read_text())
    validated = report.get("model_validation", {})
    if (report["status"] != "passed" or validated.get("model") != spec
            or report.get("environment") != environment()
            or validated.get("settings") != asdict(settings)
            or report.get("adapter_sha256") != digest(ROOT / "celerity_sparse/adapter.py")
            or report.get("attention_sha256") != digest(ROOT / "celerity_sparse/attention.py")):
        raise ValueError("GPU validation must pass for this exact checkpoint, adapter, settings and environment")
    return report


def identities(records):
    return [(r["shape"], r["token_ids_sha256"]) for r in records]


@torch.no_grad()
def evaluate(args, output):
    lb = args.longbench_repo.resolve()
    sys.path.insert(0, str(lb))
    from benchmark_celerity_longbench_v2 import load_model, select_cases, score_one, sha_json
    scorer = lb / "longbench_perplexity.py"
    if digest(scorer) != SCORER_SHA256:
        raise ValueError("Canonical scorer changed; inspect before running")
    manifest = json.loads(args.manifest.read_text())
    spec = validate_manifest(manifest, args.model)
    sparse_settings = SparseSettings(args.backend, args.keep_fraction, local_blocks=args.local_blocks)
    validation = None
    if args.backend == "flex":
        if args.validation_report is None:
            raise ValueError("--validation-report from validate_sparse_gpu.py is required for Flex")
        validation = require_gpu_validation(args.validation_report, spec, sparse_settings)
    if not torch.cuda.is_available():
        raise RuntimeError("This checkpoint diagnostic requires CUDA; use unit tests on CPU")

    # select_cases cannot resample after the mandatory preflight above. Its path
    # is a NEW output path, never an archived manifest. Reuse existing prompt code.
    before = sha_json(manifest)
    cases = select_cases(manifest, output / "unexpected_manifest_write.json")
    if before != sha_json(manifest) or (output / "unexpected_manifest_write.json").exists():
        raise AssertionError("Frozen manifest unexpectedly mutated")
    if len(cases) != 80:
        raise ValueError("Expected all 80 frozen examples before diagnostic subsetting")
    if args.limit_cases:
        cases = cases[:args.limit_cases]
    save_new(output / "source_manifest.json", manifest)
    save_new(output / "cases.json", cases)  # exact prompt and first reference, both arms reuse this list
    model, tok, tokenizer_digest = load_model(spec, "bfloat16", "cuda")
    if validation and tokenizer_digest != validation["model_validation"]["tokenizer_digest"]:
        raise ValueError("Tokenizer identity differs from the validated checkpoint")
    if int(model.config.max_position_embeddings) != spec["trained_context_tokens"]:
        raise ValueError("Actual checkpoint training context differs from manifest")
    adapter = CeleritySparseAdapter(model, sparse_settings)
    wrapped = RecordedModel(model)

    provenance = dict(model=spec, tokenizer_digest=tokenizer_digest, environment=environment(),
                      manifest_sha256=digest(args.manifest), scorer_sha256=digest(scorer),
                      utils_sha256=digest(lb / "longbench_utils.py"),
                      v2_runner_sha256=digest(lb / "benchmark_celerity_longbench_v2.py"),
                      adapter_sha256=digest(ROOT / "celerity_sparse/adapter.py"),
                      attention_sha256=digest(ROOT / "celerity_sparse/attention.py"),
                      launcher_sha256=digest(__file__),
                      longbench_commit=subprocess.check_output(
                          ["git", "-C", str(lb), "rev-parse", "HEAD"], text=True).strip(),
                      settings=asdict(sparse_settings), reference_policy="first",
                      evaluation_context=4096, case_count=len(cases),
                      diagnostic_subset=len(cases) != 80, cases_sha256=sha_json(cases),
                      precision="BF16 weights/QKV; Flex FP32 score accumulation; no INT8",
                      timing="synchronized model wall time, includes selection/metadata/instrumentation; excludes scorer loss",
                      warmup="all selected cases once per arm; includes compilation; excluded from measured pass")
    save_new(output / "provenance.json", provenance)
    summaries, windows_by_arm = {}, {}
    arms = [("original", False, 1.0), (args.backend + "_full", True, 1.0),
            (args.backend + "_sparse", True, args.keep_fraction)]
    for name, enabled, keep in arms:
        adapter.enabled = enabled
        adapter.settings = SparseSettings(args.backend, keep, local_blocks=args.local_blocks)
        wrapped.records.clear()
        adapter.reset_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for case in cases:
            score_one(wrapped, tok, case, 4096, 128)
        torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - start
        warmup_windows = identities(wrapped.records)
        wrapped.records.clear()
        adapter.reset_stats()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        rows = []
        with (output / (name + "_examples.jsonl")).open("x") as handle:
            for case in cases:
                row = score_one(wrapped, tok, case, 4096, 128)
                if row is None:
                    raise RuntimeError("A frozen example unexpectedly scored zero tokens")
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
        torch.cuda.synchronize()
        records = wrapped.records[:]
        if identities(records) != warmup_windows:
            raise AssertionError("Warmup and measured scoring windows changed")
        windows_by_arm[name] = identities(records)
        save_new(output / (name + "_windows.json"), records)
        save_new(output / (name + "_attention_calls.json"), adapter.calls)
        nll, tokens = sum(r["nll_sum"] for r in rows), sum(r["tokens_scored"] for r in rows)
        execution = adapter.statistics()
        expected_calls = len(records) * len(model.model.layers)
        if execution["completed_attention_calls"] != expected_calls or execution["failures"]:
            raise AssertionError("Missing attention-layer coverage or a backend failure")
        if enabled and (execution["calls_by_backend"] != {args.backend: expected_calls}
                        or (keep < 1 and execution["calls_omitting_blocks"] == 0)):
            raise AssertionError("Sparse backend did not execute / omitted no eligible blocks")
        summaries[name] = dict(nll_sum=nll, scored_tokens=tokens, avg_nll=nll / tokens,
                               perplexity=math.exp(nll / tokens), execution=execution,
                               warmup_and_compilation_wall_seconds=warmup_seconds,
                               measured_evaluation_wall_seconds=time.perf_counter() - start,
                               model_forward_seconds=sum(r["forward_seconds"] for r in records),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                               examples=len(rows))
        save_new(output / (name + "_summary.json"), summaries[name])
        if enabled and keep == 1 and abs(summaries[name]["avg_nll"] - summaries["original"]["avg_nll"]) > 0.02:
            raise AssertionError("Full-backend NLL control failed (>0.02 nat/token); stopping before sparse evaluation")

    original = summaries["original"]
    for name, summary in summaries.items():
        if windows_by_arm[name] != windows_by_arm["original"] or summary["scored_tokens"] != original["scored_tokens"]:
            raise AssertionError("Dense/sparse scoring windows or target counts differ")
        summary["delta_nll_per_token_vs_original"] = summary["avg_nll"] - original["avg_nll"]
        summary["delta_perplexity_vs_original"] = summary["perplexity"] - original["perplexity"]
    full = summaries[args.backend + "_full"]
    if abs(full["delta_nll_per_token_vs_original"]) > 0.02:
        raise AssertionError("Full-backend NLL differs by >0.02 nat/token: investigate numerical control before sparse interpretation")
    sparse = summaries[args.backend + "_sparse"]
    sparse["delta_nll_per_token_vs_same_backend_full"] = sparse["avg_nll"] - full["avg_nll"]
    if digest(scorer) != SCORER_SHA256:
        raise AssertionError("Scorer source changed during execution")
    save_new(output / "summary.json", dict(status="complete", arms=summaries,
             method="prefix-safe ALiBi-aware Sparge-inspired block selection",
             execution="compiled Flex block sparsity" if args.backend == "flex" else "dense sparse-mask reference; no acceleration claim",
             paired_window_identity=True))
    print(json.dumps(summaries, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--longbench-repo", type=Path, default=ROOT.parent / "Perplexity_LongBench")
    p.add_argument("--manifest", type=Path, default=ROOT.parent / "Perplexity_LongBench/experiments/celerity_all_4k_a6000_bf16/manifest.json")
    p.add_argument("--backend", choices=("flex", "reference"), default="flex")
    p.add_argument("--keep-fraction", type=float, default=0.5)
    p.add_argument("--local-blocks", type=int, default=1)
    p.add_argument("--limit-cases", type=int, default=2, help="first N frozen cases; 0 = all 80")
    p.add_argument("--validation-report", type=Path)
    p.add_argument("--output-dir", type=Path)
    args = p.parse_args()
    if args.limit_cases < 0 or args.limit_cases > 80 or not 0 < args.keep_fraction < 1:
        p.error("limit-cases must be 0..80 and sparse keep-fraction must be strictly between 0 and 1")
    output = args.output_dir or ROOT / "results/sparse_longbench" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    output = output.resolve()
    if not output.is_relative_to(ROOT / "results/sparse_longbench"):
        p.error("--output-dir must be a NEW directory inside Sparge_Analysis/results/sparse_longbench")
    output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)
    try:
        evaluate(args, output)
    except BaseException:
        save_new(output / "failure.json", dict(status="failed", error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
