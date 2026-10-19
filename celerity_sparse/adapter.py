"""Instance-local, reversible adapter for the audited adpretko Celerity source."""
from collections import Counter
import hashlib
import inspect
from pathlib import Path
from types import MethodType

import torch

from .attention import (SparseSettings, select_blocks, reference_attention,
                        flex_attention_call, block_statistics)

MODEL_SOURCE_SHA256 = "09ba09638f0b832efffd09951895455e043b7dfd152d344ff048bada8dce235c"
FAMILIES = {(640, 10, 64), (896, 14, 64), (1152, 9, 128)}


class CeleritySparseAdapter:
    """Only fresh-cache, unpadded, inference-mode eager ALiBi MHA is supported.

    Disabled calls delegate to the saved bound method without altering kwargs.
    Unsupported sparse requests raise; there is no dense fallback.
    """

    def __init__(self, model, settings=SparseSettings()):
        source = Path(inspect.getfile(type(model)))
        if hashlib.sha256(source.read_bytes()).hexdigest() != MODEL_SOURCE_SHA256:
            raise ValueError("Unknown Celerity source hash; inspect before enabling adapter")
        cfg = model.config
        if (cfg.model_type != "celerity" or cfg._attn_implementation != "eager"
                or cfg.position_embedding_type != "alibi"
                or not cfg.scale_qk_dot_by_d or cfg.pretraining_tp != 1):
            raise ValueError("Requires audited eager ALiBi Celerity, 1/d scale, TP=1")
        if (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim) not in FAMILIES:
            raise ValueError("Only the 271M/504M/906M head layouts are supported")
        if cfg.num_key_value_heads != cfg.num_attention_heads:
            raise ValueError("GQA/MQA not supported")
        if getattr(model, "_celerity_sparse_adapter", None) is not None:
            raise ValueError("An adapter is already installed")
        modules = [layer.self_attn for layer in model.model.layers]
        if not modules or any(type(m).__name__ != "CelerityAttention" for m in modules):
            raise ValueError("Every layer must use original CelerityAttention (eager)")
        self.model, self.settings = model, settings
        self.enabled = False
        self.slopes = model.model.alibi_emb.slopes
        if self.slopes.numel() != cfg.num_attention_heads:
            raise ValueError("Unexpected checkpoint ALiBi slope shape")
        self.calls = []
        self.failures = Counter()
        self._saved = []
        self._compiled = None
        self._signature = inspect.signature(model.forward)
        self._removed = False
        # Validate everything above BEFORE touching the model.
        for module in modules:
            had_instance_forward = "forward" in module.__dict__
            old_instance_forward = module.__dict__.get("forward")
            original = module.forward

            def replacement(attn, *args, _original=original, **kwargs):
                if not self.enabled:
                    result = _original(*args, **kwargs)
                    self.calls.append(dict(backend="original", layer=attn.layer_idx))
                    return result
                try:
                    return self._forward(attn, *args, **kwargs)
                except Exception as exc:
                    self.failures[type(exc).__name__ + ": " + str(exc)] += 1
                    raise

            self._saved.append((module, had_instance_forward, old_instance_forward))
            module.forward = MethodType(replacement, module)
        self._guard = model.register_forward_pre_hook(self._check_model_input, with_kwargs=True)
        model._celerity_sparse_adapter = self

    def _check_model_input(self, model, args, kwargs):
        if not self.enabled:
            return
        call = self._signature.bind_partial(*args, **kwargs).arguments
        if model.training or torch.is_grad_enabled():
            raise ValueError("Sparse adapter requires eval() and torch.no_grad()/inference_mode()")
        for name in ("attention_mask", "past_key_values", "inputs_embeds",
                     "position_ids", "cache_position"):
            if call.get(name) is not None:
                raise ValueError(f"Sparse teacher-forced path does not accept {name}")
        ids = call.get("input_ids")
        if ids is None or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1:
            raise ValueError("Expected one unpadded nonempty input_ids sequence")
        if call.get("output_attentions", model.config.output_attentions):
            raise ValueError("output_attentions is unsupported")

    def _forward(self, attn, hidden_states, attention_mask=None, position_ids=None,
                 past_key_value=None, output_attentions=False, use_cache=False,
                 cache_position=None, position_embeddings=None, position_bias=None,
                 **kwargs):
        if attn.training or torch.is_grad_enabled() or output_attentions or kwargs:
            raise ValueError("Unsupported training/attention-output/extra arguments")
        b, length, _ = hidden_states.shape
        if b != 1 or attn.use_rope or attn.attn_scale_power != 1.0:
            raise ValueError("Unsupported attention layout or scaling")
        if position_bias is None or position_bias.shape != (attn.num_heads, length, length):
            raise ValueError("Missing or unexpected original ALiBi tensor")
        # Top-level guard ensures the audited model constructed a standard causal
        # mask. No user padding/custom masks or nonzero cache offsets are allowed.
        if (attention_mask is None or attention_mask.shape[:3] != (1, 1, length)
                or attention_mask.shape[-1] < length):
            raise ValueError("Expected original eager causal mask")
        if past_key_value is not None:
            from transformers.cache_utils import DynamicCache
            if type(past_key_value) is not DynamicCache:
                raise ValueError("Only a fresh DynamicCache is supported")
            if past_key_value.get_seq_length(attn.layer_idx) != 0:
                raise ValueError("Cached decoding is unsupported; no dense fallback")
        q, k, v = [
            projection(hidden_states).view(b, length, attn.num_heads, attn.head_dim).transpose(1, 2)
            for projection in (attn.q_proj, attn.k_proj, attn.v_proj)
        ]
        if past_key_value is not None:
            k, v = past_key_value.update(k, v, attn.layer_idx, {"cache_position": cache_position})
        blocks = select_blocks(q, k, self.slopes, self.settings)
        if self.settings.backend == "reference":
            out = reference_attention(q, k, v, position_bias, attention_mask, blocks)
        else:
            if self._compiled is None:
                from torch.nn.attention.flex_attention import flex_attention
                self._compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)
            out = flex_attention_call(q, k, v, position_bias, blocks, self._compiled)
        out = attn.o_proj(out.transpose(1, 2).contiguous().reshape(b, length, -1))
        # Count COMPLETED backend calls, not merely requests or model forwards.
        self.calls.append(dict(backend=self.settings.backend, layer=attn.layer_idx,
                               q_len=length, kv_len=length, dtype=str(q.dtype),
                               **block_statistics(blocks, length)))
        return out, None, past_key_value

    def reset_stats(self):
        self.calls.clear()
        self.failures.clear()

    def statistics(self):
        counts = Counter(c["backend"] for c in self.calls)
        totals = {key: sum(c.get(key, 0) for c in self.calls) for key in (
            "retained_blocks", "eligible_blocks", "retained_causal_pairs", "eligible_causal_pairs")}
        eligible = totals["eligible_causal_pairs"]
        sparse = sum(c.get("retained_blocks", 0) < c.get("eligible_blocks", 0)
                     for c in self.calls)
        return dict(calls_by_backend=dict(counts), failures=dict(self.failures),
                    dense_fallback_calls=0, completed_attention_calls=len(self.calls),
                    calls_omitting_blocks=sparse,
                    flex_execution_coverage=counts["flex"] / max(1, len(self.calls)),
                    causal_pair_sparsity=1 - totals["retained_causal_pairs"] / eligible if eligible else None,
                    block_sparsity=1 - totals["retained_blocks"] / totals["eligible_blocks"]
                    if totals["eligible_blocks"] else None, **totals)

    def remove(self):
        if self._removed:
            return
        for module, had_forward, old_forward in self._saved:
            if had_forward:
                module.forward = old_forward
            else:
                del module.forward
        self._guard.remove()
        del self.model._celerity_sparse_adapter
        self._removed = True
