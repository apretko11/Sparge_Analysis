#!/usr/bin/env python3
"""Small CUDA gate; --kernel-only never loads checkpoint weights.

Run on the A6000, not the local CPU workstation. Nonzero exit = NO GO.
Reports are exclusive-create and cannot overwrite previous validation.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from celerity_sparse import CeleritySparseAdapter, SparseSettings
from celerity_sparse.adapter import MODEL_SOURCE_SHA256
from celerity_sparse.attention import (select_blocks, reference_attention,
                                      flex_attention_call, block_statistics)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import transformers
    import triton
    return dict(torch=torch.__version__, transformers=transformers.__version__,
                triton=triton.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def error_metrics(actual, expected):
    diff = actual.float() - expected.float()
    return dict(max_abs=diff.abs().max().item(), rms=diff.square().mean().sqrt().item(),
                relative_rms=(diff.square().mean().sqrt() /
                              expected.float().square().mean().sqrt().clamp_min(1e-8)).item())


@torch.no_grad()
def kernel_gate():
    from torch.nn.attention.flex_attention import flex_attention
    results = []
    torch.manual_seed(74)
    last = None
    for dim in (64, 128):
        # Five static sequence shapes per head dimension; isolate the two
        # batches to avoid the PyTorch 2.5.1 eight-entry compiler cache limit.
        torch._dynamo.reset()
        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
        for length in (17, 128, 129, 517, 4096):
            q, k, v = [torch.randn(1, 3, length, dim, device="cuda", dtype=torch.bfloat16)
                       for _ in range(3)]
            slopes = torch.tensor([0.003, 0.03, 0.3], device="cuda", dtype=torch.bfloat16)
            pos = torch.arange(length, device="cuda")
            bias = -slopes[:, None, None] * (pos[:, None] - pos[None, :]).abs()
            mask = torch.zeros(1, 1, length, length, device="cuda", dtype=q.dtype)
            mask.masked_fill_(pos[None, :] > pos[:, None], torch.finfo(q.dtype).min)
            for keep in (1.0, 0.5, 0.25):
                settings = SparseSettings("flex", keep)
                blocks = select_blocks(q, k, slopes, settings)
                expected = reference_attention(q, k, v, bias, mask, blocks)
                actual = flex_attention_call(q, k, v, bias, blocks, compiled)
                if not torch.isfinite(actual).all():
                    raise AssertionError("Nonfinite Flex output")
                metrics = error_metrics(actual, expected)
                if metrics["max_abs"] > 0.03 or metrics["relative_rms"] > 0.02:
                    raise AssertionError(f"Flex/reference mismatch: {metrics}")
                results.append(dict(dim=dim, length=length, keep=keep, **metrics,
                                    **block_statistics(blocks, length)))
                if length == 517 and keep == 0.5:
                    for cut in (127, 128, 129, 255, 256, 300, 512):
                        altered = [x.clone() for x in (q, k, v)]
                        for x in altered:
                            x[:, :, cut:] = torch.randn_like(x[:, :, cut:])
                        new_blocks = select_blocks(altered[0], altered[1], slopes, settings)
                        torch.testing.assert_close(blocks[:, :, :(cut+127)//128],
                                                   new_blocks[:, :, :(cut+127)//128], atol=0, rtol=0)
                        changed = flex_attention_call(*altered, bias, new_blocks, compiled)
                        torch.testing.assert_close(changed[:, :, :cut], actual[:, :, :cut],
                                                   atol=1e-5, rtol=1e-5)
                last = (q, k, v, bias, blocks, compiled)
    # Device-level evidence: require an actual Flex GPU kernel in a profiler trace.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        flex_attention_call(*last)
        torch.cuda.synchronize()
    kernels = sorted({event.name for event in prof.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA})
    # PyTorch 2.5.1 can report Inductor-generated Flex kernels as "triton_".
    # This profiler scope contains the compiled FlexAttention call.
    compiled_kernels = [
        name for name in kernels
        if "flex" in name.lower() or name.startswith("triton_")
    ]

    if not compiled_kernels:
        raise AssertionError(
            f"No compiled Flex/Triton CUDA kernel visible in profiler: {kernels}"
        )
    return dict(comparisons=results, cuda_kernels=kernels,
                attention_atol=0.03, attention_relative_rms=0.02,
                causality_atol=1e-5, causality_rtol=1e-5)


@torch.no_grad()
def model_gate(args):
    sys.path.insert(0, str(Path(args.longbench_repo).resolve()))
    from benchmark_celerity_longbench_v2 import load_model
    manifest = json.loads(Path(args.manifest).read_text())
    spec = next(m for m in manifest["models"] if m["model"] == args.model)
    model, tok, tokenizer_digest = load_model(spec, "bfloat16", "cuda")
    ids = torch.arange(517, device="cuda")[None].remainder(model.config.vocab_size)
    original = model(ids).logits
    adapter = CeleritySparseAdapter(model, SparseSettings("reference", 1))
    torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
    adapter.enabled = True
    torch.testing.assert_close(model(ids).logits, original, atol=0, rtol=0)
    adapter.settings = SparseSettings("flex", 1)
    full = model(ids).logits
    metrics = error_metrics(full, original)
    if metrics["max_abs"] > 0.5 or metrics["relative_rms"] > 0.02:
        raise AssertionError(f"Full Flex/original model mismatch: {metrics}")
    adapter.settings = SparseSettings("flex", args.keep_fraction, local_blocks=args.local_blocks)
    adapter.reset_stats()
    baseline = model(ids).logits
    for cut in (127, 128, 129, 255, 256, 300, 512):
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:] + 43) % model.config.vocab_size
        torch.testing.assert_close(model(changed).logits[:, :cut], baseline[:, :cut],
                                   atol=1e-5, rtol=1e-5)
    # A real 4K forward checks memory and the intended production sequence length.
    ids4k = torch.arange(4096, device="cuda")[None].remainder(model.config.vocab_size)
    torch.cuda.reset_peak_memory_stats()
    logits4k = model(ids4k).logits
    if not torch.isfinite(logits4k).all():
        raise AssertionError("Nonfinite 4K logits")
    stats = adapter.statistics()
    if stats["flex_execution_coverage"] != 1 or stats["calls_omitting_blocks"] == 0:
        raise AssertionError(f"Unknown/absent sparse execution: {stats}")
    return dict(model=spec, tokenizer_digest=tokenizer_digest, settings=asdict(adapter.settings),
                disabled_exact=True, full_reference_exact=True, full_flex_error=metrics,
                causality_atol=1e-5, causality_rtol=1e-5,
                full_flex_max_abs_limit=0.5, full_flex_relative_rms_limit=0.02,
                peak_allocated_bytes_4k=torch.cuda.max_memory_allocated(),
                execution=stats)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel-only", action="store_true")
    p.add_argument("--model", default="adpretko/celerity-271m-4k-ad0p0")
    p.add_argument("--longbench-repo", default=str(ROOT.parent / "Perplexity_LongBench"))
    p.add_argument("--manifest", default=str(ROOT.parent / "Perplexity_LongBench/experiments/celerity_all_4k_a6000_bf16/manifest.json"))
    p.add_argument("--keep-fraction", type=float, default=0.5)
    p.add_argument("--local-blocks", type=int, default=1)
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    # Reserve the report BEFORE expensive work, with no overwrite.
    report = args.report.resolve()
    if not report.is_relative_to(ROOT / "results"):
        p.error("--report must be inside Sparge_Analysis/results")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x") as handle:
        started = time.perf_counter()
        result = dict(status="failed", model_source_sha256=MODEL_SOURCE_SHA256,
                      adapter_sha256=digest(ROOT / "celerity_sparse/adapter.py"),
                      attention_sha256=digest(ROOT / "celerity_sparse/attention.py"))
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; do not interpret CPU tests as a GPU pass")
            result["environment"] = environment()
            result["kernel"] = kernel_gate()
            if not args.kernel_only:
                # Separate synthetic-shape specializations from the model gate;
                # do not increase cache limits or permit compiler fallbacks.
                torch._dynamo.reset()
                result["model_validation"] = model_gate(args)
            result["status"] = "passed"
        except BaseException:
            result["error"] = traceback.format_exc()
            raise
        finally:
            result["validation_wall_seconds_including_compile"] = time.perf_counter() - started
            json.dump(result, handle, indent=2)
            handle.write("\n")
            handle.flush()
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
