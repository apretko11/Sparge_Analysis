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
Generation smoke-test follow-up
-------------------------------

A follow-up generation script was added after the original prefill benchmark:

    scripts/generate_text.py

This script follows Mostafa's requested format: it is an argparse-based script that takes a Hugging Face model name or local model path plus a prompt, then generates text using `model.generate(...)`.

Example:

    python scripts/generate_text.py \
      --model models/baseline/h640 \
      --prompt "In one paragraph, explain why sparse attention can help long-context inference." \
      --max-new-tokens 64 \
      --dtype bf16 \
      --device-map auto \
      --out results/generation_h640_baseline.json

Generation command wrappers were also added:

    commands/07_run_generation_smoke_tests.sh
    commands/08_plot_generation_results.sh

Generation results
------------------

Generation smoke-test JSON files are stored directly under:

    results/generation_*.json

Generation plots are stored under:

    plots/generation/

The generation tests cover:

    h640, h896, h1152
    baseline and Sparge
    short prompt and long prompt

This gives 12 generation smoke-test results total.

Important generation caveat
---------------------------

These generation results are smoke tests, not the main performance benchmark.

The original Sparge prefill patch worked for long-context prefill timing, but failed during autoregressive generation because `generate()` performs token-by-token decode. During decode, query length is usually 1, while SpargeAttention requires query sequence length at least 128.

The Sparge patch was updated with a decode fallback:

    if query.size(-2) < 128:
        return None

This means:

    long prefill can use SpargeAttention when the query length is large enough;
    short-query decode falls back to the original Celerity attention path.

The previous pre-decode-fallback patch state is archived under:

    patches/archive/

Generation smoke-test summary
-----------------------------

| Model | Variant | Prompt | Prompt tokens | Generated tokens | Time | Generated tok/s |
|---|---|---|---:|---:|---:|---:|
| h640 | baseline | short | 15 | 64 | 0.9932 s | 64.44 |
| h640 | Sparge | short | 15 | 64 | 1.0114 s | 63.28 |
| h640 | baseline | long | 1122 | 64 | 0.9394 s | 68.13 |
| h640 | Sparge | long | 1122 | 64 | 3.1615 s | 20.24 |
| h896 | baseline | short | 15 | 64 | 1.0319 s | 62.02 |
| h896 | Sparge | short | 15 | 64 | 1.0656 s | 60.06 |
| h896 | baseline | long | 1122 | 64 | 1.0411 s | 61.47 |
| h896 | Sparge | long | 1122 | 64 | 1.3889 s | 46.08 |
| h1152 | baseline | short | 15 | 64 | 1.3259 s | 48.27 |
| h1152 | Sparge | short | 15 | 64 | 1.2065 s | 53.04 |
| h1152 | baseline | long | 1122 | 64 | 1.1587 s | 55.23 |
| h1152 | Sparge | long | 1122 | 64 | 2.7930 s | 22.91 |

The main performance result remains the prefill benchmark. The generation smoke tests mainly confirm that baseline generation works, Sparge-patched generation works, and long-prompt Sparge-prefill plus decode fallback no longer crashes.
