"""Small random-weight models using the UNMODIFIED, hash-pinned HF Python source.

No pretrained weights are used. Source files are separately cached under results/
and are not vendored as an alternative modeling implementation.
"""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import types

from celerity_sparse.adapter import MODEL_SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CELERITY_SOURCE_DIR", ROOT / "results/sparse_source"))
CONFIG_SHA = "83c3a27beedc8fe51a07619835d4e94d1777286a25263d24b3a7a3f89c320481"


def classes():
    package = "_sparse_validation_celerity"
    if package + ".modeling_celerity" in sys.modules:
        mod = sys.modules[package + ".modeling_celerity"]
        return mod.CelerityForCausalLM, mod.CelerityConfig
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(SOURCE)]
    sys.modules[package] = pkg
    for name in ("configuration_celerity", "modeling_celerity"):
        path = SOURCE / (name + ".py")
        if not path.exists():
            raise RuntimeError(f"Missing CPU source fixture {path}; see docs/sparse_longbench.md")
        expected = MODEL_SOURCE_SHA256 if name == "modeling_celerity" else CONFIG_SHA
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        spec = importlib.util.spec_from_file_location(package + "." + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
    return module.CelerityForCausalLM, module.CelerityConfig


def model_for(heads=10, dim=64, layers=2, dtype=None):
    import torch
    Model, Config = classes()
    torch.manual_seed(710)
    cfg = Config(vocab_size=97, hidden_size=heads * dim, intermediate_size=128,
                 num_hidden_layers=layers, num_attention_heads=heads,
                 num_key_value_heads=heads, head_dim=dim, max_position_embeddings=4096,
                 norm_type="layernorm", attention_bias=True, mlp_bias=True,
                 position_embedding_type="alibi", scale_qk_dot_by_d=True,
                 attention_dropout=0.2, use_cache=True)
    return Model(cfg).to(dtype=dtype or torch.float32).eval()
