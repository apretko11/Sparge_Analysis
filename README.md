SpargeAttention / Celerity prefill timing experiment
====================================================

Date: 2026-06-18
Machine: Hyperstack L40
CUDA image: Server 22.04 LTS R550 CUDA 12.4
GPU: NVIDIA L40
Conda env: sparge_celerity
Python: 3.10
Torch: 2.6.0+cu124
Transformers: 4.46.3
SpargeAttention install: ~/SpargeAttn, installed with `python -m pip install -e . --no-build-isolation`

Models
------

Baseline model repos downloaded from Hugging Face:

1. melhoushi/gpt_cp_h640_d13_gbs48_tpp20.0_lp0.0_null
2. melhoushi/gpt_cp_h896_d17_gbs66_tpp20.0_lp0.0_null
3. melhoushi/gpt_cp_h1152_d23_gbs76_tpp20.0_lp0.0_null

Directory structure:

models/baseline/h640
models/baseline/h896
models/baseline/h1152

models/sparge/h640
models/sparge/h896
models/sparge/h1152

Only the models/sparge/*/modeling_celerity.py files were patched.

Patch notes
-----------

The SpargeAttention README suggests replacing:

    torch.nn.functional.scaled_dot_product_attention(...)

with:

    spas_sage2_attn_meansim_topk_cuda(...)

However, these Celerity models do not call scaled_dot_product_attention. They manually implement attention in CelerityAttention._attn using:

    torch.matmul(query, key.transpose(-1, -2))
    softmax
    torch.matmul(attn_weights, value)

Therefore the patch hooks into CelerityAttention._attn.

The patch is environment-gated:

    CELERITY_USE_SPARGE_ATTN=1

This is not required by SpargeAttention; it is a safety switch added for this experiment. With this variable unset, the patched model runs the original attention path.

Important semantic caveat
-------------------------

The models use ALiBi position bias. The original Celerity attention code adds position_bias before softmax.

The plug-and-play SpargeAttention API used here accepts q/k/v, topk, and is_causal, but does not accept arbitrary additive ALiBi bias or additive attention masks.

For the Sparge timing runs, we explicitly used:

    CELERITY_USE_SPARGE_ATTN=1
    CELERITY_SPARGE_IGNORE_BIAS=1
    CELERITY_SPARGE_IGNORE_MASK=1
    CELERITY_SPARGE_TOPK=0.5

So these results should be described as timing-focused prefill experiments, not exact-logit-equivalent inference.

Sanity checks
-------------

1. h640 baseline vs h640 patched with Sparge OFF:
   max abs diff: 0.0
   mean abs diff: 0.0

2. h640 patched with Sparge ON:
   logits shape was correct
   logits were finite

Benchmark types
---------------

Three benchmark sets were run:

1. random_synthetic_tokens_nocache
   - synthetic random input_ids
   - use_cache=False
   - 3 model sizes x 3 seq lens x baseline/Sparge = 18 results

2. random_synthetic_tokens_withcache
   - synthetic random input_ids
   - use_cache=True
   - 3 model sizes x 3 seq lens x baseline/Sparge = 18 results

3. text_prompt_withcache
   - repeated text prompt: "The quick brown fox jumps over the lazy dog. "
   - use_cache=True
   - 3 model sizes x 3 seq lens x baseline/Sparge = 18 results

Total result JSONs: 54

Scripts
-------

scripts/bench_prefill.py
    Synthetic random token prefill timing.

scripts/bench_prefill_text_prompt.py
    Repeated text prompt prefill timing.

scripts/plot_prefill_results_from_dir.py
    Directory-based plotting script.

Results
-------

JSON files:

results/random_synthetic_tokens_nocache/
results/random_synthetic_tokens_withcache/
results/text_prompt_withcache/

Plots:

plots/random_synthetic_tokens_nocache/
plots/random_synthetic_tokens_withcache/
plots/text_prompt_withcache/

Text prompt with cache summary
------------------------------

h640 seq1024:
    baseline = 11.03 ms
    sparge   = 22.83 ms
    speedup  = 0.48x

h640 seq4096:
    baseline = 101.10 ms
    sparge   = 26.94 ms
    speedup  = 3.75x

h640 seq8192:
    baseline = 368.82 ms
    sparge   = 53.21 ms
    speedup  = 6.93x

h896 seq1024:
    baseline = 14.57 ms
    sparge   = 28.89 ms
    speedup  = 0.50x

h896 seq4096:
    baseline = 186.59 ms
    sparge   = 42.52 ms
    speedup  = 4.39x

h896 seq8192:
    baseline = 675.33 ms
    sparge   = 96.53 ms
    speedup  = 7.00x

h1152 seq1024:
    baseline = 22.43 ms
    sparge   = 39.60 ms
    speedup  = 0.57x

h1152 seq4096:
    baseline = 193.72 ms
    sparge   = 73.20 ms
    speedup  = 2.65x

h1152 seq8192:
    baseline = 652.68 ms
    sparge   = 159.50 ms
    speedup  = 4.09x

High-level takeaway
-------------------

At short context length 1024, SpargeAttention is slower due to overhead.

At long context lengths 4096 and 8192, SpargeAttention is substantially faster in these timing-focused prefill tests.

The strongest text-prompt-with-cache speedups were around 7x for h640/h896 at 8192 tokens.
