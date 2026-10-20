"""CPU source/semantic audit, NOT an attention backend or GPU-kernel emulator.

The upstream selector's Python body runs unchanged. Only its Triton primitives
are replaced with small mathematical CPU probes. Tail behavior in these probes
is explicitly defined; it does not certify upstream's masked loads or rounding.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PIN = json.loads((HERE / "source_pin.json").read_text())


def verify_sources():
    for name, expected in PIN["sha256"].items():
        actual = hashlib.sha256((HERE / "upstream" / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream inspection source changed: {name}")


def function_ast(filename, name):
    tree = ast.parse((HERE / "upstream" / filename).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_findings():
    verify_sources()
    node = function_ast("spas_sage_attn/core.py", "spas_sage2_attn_meansim_topk_cuda")
    loads = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    call = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold")
    # Signature in attn_cuda.h: ..., k_scale, tensor_layout, is_causal, qk_quant_gran, sm_scale, ...
    return dict(attn_mask_consumed="attn_mask" in loads,
                ampere_causal_argument=ast.literal_eval(call.args[10]),
                selector_receives_scale=any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "get_block_map_meansim_fuse_quant"
                    and any(k.arg == "scale" for k in n.keywords) for n in ast.walk(node)))


def pool_quant_cpu(x, mean, block_size, threshold):
    """Idealized fused pool/mean-cosine/INT8 primitive, with valid-tail rows only."""
    pools, gates, scales, quantized = [], [], [], []
    for start in range(0, x.shape[-2], block_size):
        block = x[..., start:start + block_size, :].float()
        if mean is not None:
            block = block - mean.float()
        pools.append(block.mean(-2).to(x.dtype))
        unit = block / block.norm(dim=-1, keepdim=True)
        # Match the source's FP16 normalized vectors, FP32 dot accumulation in intent.
        unit = unit.half().float()
        similarity = (unit @ unit.transpose(-1, -2)).mean((-1, -2))
        gates.append(similarity > threshold[None, :])
        scale = block.abs().amax((-1, -2)) / 127 + 1e-7
        z = block / scale[..., None, None]
        quantized.append((z + 0.5 * torch.where(z >= 0, 1, -1)).to(torch.int8))
        scales.append(scale)
    return (torch.stack(pools, -2), torch.stack(gates, -1),
            torch.cat(quantized, -2), torch.stack(scales, -1))


def fill_causal_cpu(mask, ratio):
    q = torch.arange(mask.shape[0])[:, None]
    k = torch.arange(mask.shape[1])[None, :]
    return k < (q + 1) * ratio


def fill_selected_cpu(mask, count, sorted_indices):
    for b in range(mask.shape[0]):
        for h in range(mask.shape[1]):
            for q in range(mask.shape[2]):
                n = max(1, int(count[b, h, q]))  # upstream fill kernel promotes zero to one
                mask[b, h, q, sorted_indices[b, h, q, :n]] = True
    return mask


def upstream_selector():
    """Execute the pinned Python selector without importing CUDA/Triton modules."""
    verify_sources()
    nodes = [function_ast("spas_sage_attn/utils.py", name) for name in
             ("hyperparameter_check", "get_block_map_meansim_fuse_quant")]
    namespace = dict(torch=torch, Tensor=torch.Tensor,
                     get_pool_sim_triton_simmean_fuse_quant=pool_quant_cpu,
                     fill_causal_mask_triton=fill_causal_cpu,
                     fill_block_map_triton=fill_selected_cpu)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned upstream selector>", "exec"), namespace)
    return namespace["get_block_map_meansim_fuse_quant"]


def select(q, k, topk=0.25, smooth=True, threshold=-0.1):
    if q.device.type != "cpu" or k.device.type != "cpu":
        raise ValueError("This is a CPU audit, not an integration backend")
    return upstream_selector()(q, k, km=k.mean(-2, keepdim=True) if smooth else None,
                               is_causal=True, topk=topk, cdfthreshd=None,
                               simthreshd1=threshold, return_lut=False)


def alibi(slopes, length, dtype=torch.float64):
    positions = torch.arange(length, dtype=dtype)
    return -slopes.to(dtype)[:, None, None] * (positions[:, None] - positions[None, :]).abs()


def dense_oracle(q, k, v, bias=None, block_map=None, causal=True):
    """Small dense mathematical oracle only; no performance or BF16 parity claim."""
    length, dim = q.shape[-2:]
    scores = q @ k.transpose(-1, -2) / dim
    if bias is not None:
        scores = scores + bias
    allowed = torch.ones_like(scores, dtype=torch.bool)
    if block_map is not None:
        allowed &= block_map.repeat_interleave(128, -2).repeat_interleave(64, -1)[..., :length, :k.shape[-2]]
    if causal:
        allowed &= torch.arange(length)[:, None] >= torch.arange(k.shape[-2])[None, :]
    return scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ v


def query_witness():
    q = torch.zeros(1, 1, 256, 64)
    q[..., 0] = 1
    changed = q.clone()
    changed[..., 129:, 0] = 0
    changed[..., 129:, 1] = 1
    k = torch.zeros_like(q)
    k[..., :64, 0], k[..., 64:128, 1] = 4, 4
    k[..., 128:192, 0], k[..., 192:, 1] = -4, -4
    v = torch.zeros_like(q)
    v[..., 64:128, :] = 1
    return q, changed, k, v


def findings():
    torch.set_num_threads(2)
    q, changed, k, v = query_witness()
    original_map, *_ = select(q, k)
    changed_map, *_ = select(changed, k)
    original = dense_oracle(q, k, v, block_map=original_map)
    modified = dense_oracle(changed, k, v, block_map=changed_map)
    # Extend only the future: budget changes from floor(.25*4)=1 to floor(.25*8)=2.
    length_k = torch.zeros_like(q)
    for i, value in enumerate((4, 3, 2, 1)):
        length_k[..., i * 64:(i + 1) * 64, 0] = value
    short_map, *_ = select(q, length_k)
    long_map, *_ = select(torch.cat((q, q), -2), torch.cat((length_k, length_k), -2))
    # Blockwise Q quantization changes an earlier row when only a later row changes.
    x = torch.ones(1, 1, 128, 64)
    x[..., 0, 0] = 0.31
    y = x.clone()
    y[..., -1, -1] = 100
    _, _, xi, xs = pool_quant_cpu(x, None, 128, torch.tensor([-0.1]))
    _, _, yi, ys = pool_quant_cpu(y, None, 128, torch.tensor([-0.1]))
    qdiff = float((xi[..., 0, :].float() * xs[..., 0, None]
                   - yi[..., 0, :].float() * ys[..., 0, None]).abs().max())
    # Global centering changes an old key block's similarity gate (no quantization needed).
    keys = torch.zeros(1, 1, 128, 64)
    keys[..., :32, 0], keys[..., 32:64, 0] = 1, -1
    keys[..., 64:96, 1], keys[..., 96:, 1] = 1, -1
    new_keys = keys.clone()
    new_keys[..., 64:, :] = 0
    new_keys[..., 64:, 1] = 20
    _, gate, *_ = pool_quant_cpu(keys, keys.mean(-2, keepdim=True), 64, torch.tensor([0.6]))
    _, new_gate, *_ = pool_quant_cpu(new_keys, new_keys.mean(-2, keepdim=True), 64, torch.tensor([0.6]))
    return dict(upstream_revision=PIN["revision"], decision="option_C_no_faithful_integration",
                evidence_scope="CPU mathematical probes plus unchanged upstream Python decision logic; not GPU execution",
                source=source_findings(),
                query_future_dependency=dict(unchanged_through_token=128, changed_tokens="129..255",
                    selected_before=original_map[0, 0, 1].nonzero().flatten().tolist(),
                    selected_after=changed_map[0, 0, 1].nonzero().flatten().tolist(),
                    token_128_output_max_abs_change=float((original[..., 128, :] - modified[..., 128, :]).abs().max()),
                    final_token_causal_mask_applied=True),
                length_dependency=dict(selected_at_length_256=int(short_map[0, 0, 1].sum()),
                                       selected_at_length_512=int(long_map[0, 0, 1].sum())),
                q_quantization_future_dependency=dict(earlier_row_dequantized_max_abs_change=qdiff),
                key_centering_future_dependency=dict(old_block_gate_before=bool(gate[0, 0, 0]),
                                                    old_block_gate_after=bool(new_gate[0, 0, 0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="optional NEW JSON file; never overwritten")
    args = parser.parse_args()
    report = findings()
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report:
        repo = HERE.parents[1]
        destination = args.report.resolve()
        allowed = (HERE, repo / "results/sparge_alibi_audit")
        if not any(destination.is_relative_to(p.resolve()) for p in allowed):
            parser.error("report must be inside research/sparge_alibi or results/sparge_alibi_audit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as handle:
            handle.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
