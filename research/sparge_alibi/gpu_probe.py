#!/usr/bin/env python3
"""LATER ONLY: falsification probe of upstream on cached 271M QKV, not an adapter.

No downloads, installations, monkey patches, PPL evaluation, or speedup claims.
Exit 1 means the audit completed but upstream is NOT qualified. Exit 2 means the
probe could not complete. This probe cannot authorize a LongBench integration.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = "adpretko/celerity-271m-4k-ad0p0"
REVISION = "be032b277561058f3215d0862ba282c164e73d35"
MODEL_SOURCE_SHA = "09ba09638f0b832efffd09951895455e043b7dfd152d344ff048bada8dce235c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(actual, expected):
    import torch
    require(bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()), "Nonfinite output; stopping probe")
    diff = actual.float() - expected.float()
    return dict(max_abs=float(diff.abs().max()), relative_rms=float(
        diff.square().mean().sqrt() / expected.float().square().mean().sqrt().clamp_min(1e-8)))


def inspect_installation(pin):
    spec = importlib.util.find_spec("spas_sage_attn")
    require(spec is not None and spec.origin, "SpargeAttention must already be installed; no installation is performed")
    package = Path(spec.origin).resolve().parent
    revision = subprocess.check_output(["git", "-C", str(package.parent), "rev-parse", "HEAD"], text=True).strip()
    require(revision == pin["revision"], "Installed upstream checkout is not the audited revision; stop for review")
    # A source installation is required so compiled-kernel sources can be checked.
    for name, expected in pin["sha256"].items():
        require(sha(package.parent / name) == expected, f"Installed source differs: {name}")
    binaries = {str(p): sha(p) for p in package.glob("*.so")}
    require(binaries, "No installed CUDA extension binaries found")
    return dict(package=str(package), revision=revision, binary_sha256=binaries,
                binary_build_attestation="unavailable; hashes identify binaries but do not prove their build inputs")


def probe(report):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import transformers
    from cpu_audit import PIN, verify_sources, source_findings
    verify_sources()
    report.update(upstream_revision=PIN["revision"], source_findings=source_findings(),
                  installation=inspect_installation(PIN))
    require(torch.cuda.is_available(), "CUDA unavailable; do not run this probe on the local CPU machine")
    require(torch.cuda.device_count() == 1, "Expose exactly one GPU using CUDA_VISIBLE_DEVICES")
    require(torch.cuda.get_device_capability() == (8, 6), "This diagnostic is scoped to the A6000/SM86")
    from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda as upstream
    from spas_sage_attn import _qattn
    from spas_sage_attn.utils import get_block_map_meansim_fuse_quant
    report["environment"] = dict(torch=torch.__version__, transformers=transformers.__version__,
                                 cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
    report["model"] = dict(id=MODEL, revision=REVISION, local_files_only=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    require(sha(inspect.getfile(type(model))) == MODEL_SOURCE_SHA, "Unexpected Celerity model source")
    cfg = model.config
    require((cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim) == (640, 10, 64)
            and cfg.num_key_value_heads == 10 and cfg.position_embedding_type == "alibi"
            and cfg.scale_qk_dot_by_d, "Requires original 271M eager ALiBi 1/d attention")
    attention = model.model.layers[0].self_attn
    report["cases"] = []

    class Captured(Exception):
        pass

    with torch.no_grad():
        for length in (128, 256, 257):
            captured = {}
            def capture(module, args, kwargs):
                captured.update(inspect.signature(module.forward).bind(*args, **kwargs).arguments)
                raise Captured
            handle = attention.register_forward_pre_hook(capture, with_kwargs=True)
            try:
                ids = torch.arange(length, device="cuda")[None] % cfg.vocab_size
                model.model(input_ids=ids, use_cache=False)
            except Captured:
                pass
            finally:
                handle.remove()
            require(captured and captured.get("past_key_value") is None, "Expected uncached original attention inputs")
            hidden = captured["hidden_states"]
            q, k, v = [projection(hidden).view(1, length, 10, 64).transpose(1, 2).contiguous()
                       for projection in (attention.q_proj, attention.k_proj, attention.v_proj)]
            mask = captured["attention_mask"][..., :length]
            bias = captured["position_bias"].to(q.dtype).unsqueeze(0)
            # Test oracle follows the inspected original BF16 operation order.
            scores = q @ k.transpose(-1, -2) / 64
            scores = scores + mask
            scores += bias
            expected = scores.softmax(-1, dtype=torch.float32).to(q.dtype) @ v
            projected = attention.o_proj(expected.transpose(1, 2).reshape(1, length, 640))
            require(torch.equal(projected, attention(**captured)[0]), "Test oracle disagrees with unmodified Celerity")

            def call(values=v, topk=1.0, additive=mask + bias):
                return upstream(q, k, values, scale=1/64, is_causal=True, topk=topk,
                                attn_mask=additive, tensor_layout="HND")
            call()  # warmup/JIT, excluded from the measured call
            torch.cuda.synchronize()
            started = time.perf_counter()
            full = call()
            torch.cuda.synchronize()
            full_seconds = time.perf_counter() - started
            without_mask = call(additive=None)
            changed_v = v.clone()
            changed_v[..., 1:, :] += 16
            future_changed = call(values=changed_v)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as profile:
                sparse = call(topk=0.5)
                torch.cuda.synchronize()
            kernels = sorted({event.name for event in profile.events()
                              if event.device_type == torch.autograd.DeviceType.CUDA})
            require(any("block_sparse_attn_kernel" in name for name in kernels), "No actual sparse CUDA kernel observed")
            # Exercise the upstream instrumentation flag, with its ORIGINAL causal=False.
            # This observes PV counts; it does not repair or replace the upstream path.
            lut, counts, qi, qs, ki, ks = get_block_map_meansim_fuse_quant(
                q, k, k.mean(-2, keepdim=True), is_causal=True, simthreshd1=-0.1,
                cdfthreshd=None, topk=0.5, return_lut=True)
            require(bool((counts > 0).all()), "Empty LUT row; stopping before uninitialized output")
            counted_output = torch.empty_like(q)
            pv_counts = _qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
                qi, ki, v.half(), counted_output, lut, counts,
                torch.full((10,), 50.0, device="cuda"), qs, ks, 1, False, 1, 1/64, 1)
            require(torch.equal(counted_output, sparse), "Instrumented kernel differs from public API")
            require(bool((pv_counts > 0).all()), "Missing completed PV computation")
            error = metrics(full, expected)
            report["cases"].append(dict(length=length, qkv_shape=list(q.shape), dtype=str(q.dtype),
                full_selection_vs_original=error,
                existing_kernel_tolerance_passed=error["max_abs"] <= 0.03 and error["relative_rms"] <= 0.02,
                mask_argument_has_no_effect=bool(torch.equal(full, without_mask)),
                future_v_token0_change=metrics(future_changed[..., :1, :], full[..., :1, :]),
                sparse_vs_original=metrics(sparse, expected), full_call_wall_seconds=full_seconds,
                selected_qk_tiles=int(counts.sum()), selected_qk_tiles_by_row=counts.cpu().tolist(),
                completed_pv_tiles_per_warp=pv_counts.cpu().tolist(), cuda_kernels=kernels))
    report.update(status="not_qualified", integration_available=False,
                  reason="Known ignored masks, disabled token causality, and non-prefix-safe selection; see CPU audit",
                  scope="layer-0 QKV from cached 271M; diagnostic evidence, not full-model validation or a speedup benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    base = ROOT / "results/sparge_alibi_audit"
    output = args.output_dir or base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    require(base.resolve() == base and output.resolve().is_relative_to(base), "Output must be a new directory inside results/sparge_alibi_audit")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="probe_failed", integration_available=False, probe_sha256=sha(__file__))
    try:
        probe(report)
        code = 1
    except BaseException:
        report["error"] = traceback.format_exc()
        code = 2
    with (output / "report.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"{report['status']}: {output / 'report.json'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
