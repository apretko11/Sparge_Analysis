"""Two backends for ONE prefix-safe block policy; no upstream Sparge kernels.

The reference materializes dense scores and does NOT accelerate sparse attention.
Flex uses an actual BlockMask. Both retain the checkpoint's materialized ALiBi.
"""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SparseSettings:
    backend: str = "reference"
    keep_fraction: float = 0.5
    block_size: int = 128
    local_blocks: int = 1

    def __post_init__(self):
        if self.backend not in ("reference", "flex"):
            raise ValueError("backend must be reference or flex")
        if not 0 < self.keep_fraction <= 1:
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.block_size != 128:
            raise ValueError("Only 128-token blocks are validated")
        if self.local_blocks < 1:
            raise ValueError("The current causal block must always be retained")


def select_blocks(q, k, slopes, settings):
    """[B,H,Qblocks,Kblocks], using no information after a block's first Q.

    Rank COMPLETED historical blocks by Q[first] dot mean(K[block])/d
    plus mean ALiBi. Retain ceil(fraction * eligible blocks), plus local
    blocks. Counts depend only on prefix length. No smoothing/quantization.
    """
    b, h, length, d = q.shape
    if k.shape != q.shape or slopes.numel() != h:
        raise ValueError("Expected equal-length MHA Q/K and one slope per head")
    bs = settings.block_size
    n = math.ceil(length / bs)
    block = torch.arange(n, device=q.device)
    qi, ki = block[:, None], block[None, :]
    local = (ki <= qi) & (ki > qi - settings.local_blocks)
    historical = ki <= qi - settings.local_blocks
    if settings.keep_fraction == 1:
        return (ki <= qi)[None, None].expand(b, h, n, n).contiguous()

    # Padding is never selected as historical: only k_block < q_block qualifies.
    pooled = F.pad(k.float(), (0, 0, 0, n * bs - length))
    pooled = pooled.reshape(b, h, n, bs, d).mean(dim=-2)
    representative = q[:, :, ::bs, :].float()
    scores = representative @ pooled.transpose(-1, -2) / d
    distance = qi * bs - (ki * bs + (bs - 1) / 2)
    scores = scores - slopes.reshape(1, h, 1, 1).float() * distance
    scores = scores.masked_fill(~historical, -torch.inf)
    # Stable tie-breaking by key block index; no data-dependent keep count.
    order = scores.argsort(dim=-1, descending=True, stable=True)
    rank = order.argsort(dim=-1)
    count = torch.ceil(historical.sum(-1) * settings.keep_fraction).to(torch.long)
    return local | (historical & (rank < count[None, None, :, None]))


def token_mask(blocks, length, block_size=128):
    pos = torch.arange(length, device=blocks.device)
    return blocks[:, :, pos[:, None] // block_size, pos[None, :] // block_size] & (
        pos[:, None] >= pos[None, :]
    )


def make_block_mask(blocks):
    """Build public Flex metadata directly, avoiding an H*L*L mask expansion.

    All retained blocks are classified partial, so causal mask_mod ALWAYS
    executes, including diagonal blocks. No full-block causal shortcut.
    Compatible with the PyTorch 2.5.1 from_kv_blocks signature.
    """
    from torch.nn.attention.flex_attention import BlockMask

    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    counts = blocks.sum(dim=-1).to(torch.int32).contiguous()
    indices = blocks.to(torch.int32).argsort(
        dim=-1, descending=True, stable=True
    ).to(torch.int32).contiguous()
    return BlockMask.from_kv_blocks(
        counts, indices, BLOCK_SIZE=128, mask_mod=causal
    )


def reference_attention(q, k, v, bias, attention_mask, blocks):
    """Match Celerity eager operation order, adding only the sparse mask."""
    scores = q @ k.transpose(-1, -2) / q.shape[-1]
    if attention_mask is not None:
        scores = scores + attention_mask[..., :k.shape[-2]]
    scores += bias.to(scores.dtype).unsqueeze(0)
    scores = scores.masked_fill(~token_mask(blocks, q.shape[-2]), -torch.inf)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return probs @ v


def flex_attention_call(q, k, v, bias, blocks, compiled):
    if q.device.type != "cuda":
        raise RuntimeError("Flex backend requires CUDA; no reference/dense fallback")
    if q.dtype != torch.bfloat16 or q.shape[-1] not in (64, 128):
        raise ValueError("Flex path initially supports BF16, head dimensions 64/128")
    if torch.cuda.get_device_capability(q.device)[0] < 8:
        raise RuntimeError("Flex validation requires an Ampere-or-newer NVIDIA GPU")
    position_bias = bias.to(q.dtype)
    # Compute concrete bounds in eager Python: PyTorch 2.5.1 Flex lowering
    # can fail on symbolic shape lookups inside a captured score_mod.
    q_max = int(position_bias.shape[-2]) - 1
    kv_max = int(position_bias.shape[-1]) - 1

    def score_mod(score, b, h, q_idx, kv_idx):
        # Clamp only for out-of-bounds lanes in the final partial block.
        # Flex masks padded lanes; valid lanes read the ORIGINAL ALiBi tensor.
        return score + position_bias[h, q_idx.clamp(max=q_max),
                                     kv_idx.clamp(max=kv_max)].float()

    return compiled(q, k, v, score_mod=score_mod,
                    block_mask=make_block_mask(blocks), scale=1.0 / q.shape[-1])


def block_statistics(blocks, length):
    """Sparsity relative to causal dense attention, excluding future/padded keys."""
    n = blocks.shape[-1]
    pos = torch.arange(n, device=blocks.device)
    sizes = (length - pos * 128).clamp(min=0, max=128)
    pairs = sizes[:, None] * sizes[None, :]
    pairs = torch.tril(pairs, diagonal=-1) + torch.diag(sizes * (sizes + 1) // 2)
    retained_blocks = int(blocks.sum().item())
    retained_pairs = int((blocks * pairs).sum().item())
    heads = blocks.shape[0] * blocks.shape[1]
    eligible_blocks = heads * n * (n + 1) // 2
    eligible_pairs = heads * length * (length + 1) // 2
    return dict(retained_blocks=retained_blocks, eligible_blocks=eligible_blocks,
                retained_causal_pairs=retained_pairs, eligible_causal_pairs=eligible_pairs)
