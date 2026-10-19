# Celerity sparse LongBench diagnostic

This is **prefix-safe, ALiBi-aware Sparge-inspired block selection**, not stock
SpargeAttention and not an evaluation of its published CUDA/quantized algorithm.
The reference backend simulates sparsity with dense PyTorch operations. The Flex
backend requests compiled block-sparse execution; GPU validation is mandatory.
No upstream SpargeAttention installation is needed.

## Method and scope

Each 128-token query block uses its **first** query to rank **completed**
historical key blocks by Q[first] dot mean(K[block])/head_dim plus mean ALiBi.
Keep ceil(keep_fraction * eligible_history), plus the current causal block and
local_blocks-1 preceding blocks. Counts depend only on the prefix, never on the
total future content. Ties use stable key-block order. Current blocks have
token-level triangular masks. There is no global Q/K smoothing, INT8
quantization, query-block averaging, similarity gate, or online-softmax pruning.
This differs materially from stock Sparge; report the adapted method by name.

The adapter reuses q_proj/k_proj/v_proj/o_proj and the ORIGINAL ALiBi tensor.
The checkpoint's slopes, attention scaling (1/d), weights, dropout configuration,
residual paths, tokenizer and model configuration remain untouched. Dropout is
inactive in eval mode; training is rejected. Full-reference control follows
Celerity's BF16 operation order exactly. Flex accumulates scores differently, so
rounding differences are expected even at keep_fraction=1.

Only the inspected adpretko source SHA is accepted. Head layouts 640/10/64,
896/14/64 and 1152/9/128 are supported. Eager MHA, eval/no_grad, one unpadded
input sequence, and a fresh DynamicCache are required. Custom masks, nondefault
positions, existing caches, decoding, GQA, RoPE and attention outputs are rejected.
Disabled adapter calls the original bound methods. remove() restores them.

Celerity still constructs its quadratic ALiBi and causal-mask tensors. We have
deliberately not changed this model behavior in the initial implementation.
Consequently this is NOT a claim of linear-memory attention; start at 4K.
Selection uses an O(heads * blocks^2) score matrix. Instrumentation counts actual
completed backend calls and synchronizes small GPU counters, adding overhead
that is included in reported model-forward time.

## CPU tests (no checkpoint weights)

Use the existing CPU environment; do not change the validated GPU environment.
Tests use the unmodified, hash-checked HF model source with small random-weight
models. Their layers/vocabulary are reduced for tests, not for evaluation.

The two source-only fixtures are already in results/sparse_source on this checkout.
To recreate them on another machine:

```bash
cd ~/Sparge_Analysis
mkdir -p results/sparse_source
curl -L --fail https://huggingface.co/adpretko/celerity-906m-8k-ld0p8-ild/resolve/1126d8b0496724c755f1c960db5dd5283878d340/modeling_celerity.py -o results/sparse_source/modeling_celerity.py
curl -L --fail https://huggingface.co/adpretko/celerity-906m-8k-ld0p8-ild/resolve/1126d8b0496724c755f1c960db5dd5283878d340/configuration_celerity.py -o results/sparse_source/configuration_celerity.py
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/lb_ppl_cpu/bin/python -B -m unittest discover -s tests_sparse -v
```

These tests do not validate CUDA code generation or a pretrained checkpoint.
The local environment is PyTorch 2.14.1+cpu / Transformers 4.44.2.

## A6000 gate, before any PPL

Resume the instance manually. Use its existing BF16 LongBench environment with
PyTorch 2.5.1 and Transformers 4.44.2, compatible bundled Triton, and its existing
LongBench dataset-loader versions (the validated loader used datasets 2.19.2).
Do NOT run historical Sparge setup/restore scripts or upgrade dependencies.
Flex is in PyTorch; no Sparge CUDA extension and no TORCH_CUDA_ARCH_LIST=8.9.
The 2.5.1 source has Ampere >=8.0 BF16 configurations for head sizes 64/128;
actual SM86 execution remains a required test, not an assumed compatibility claim.
torch.compile will generate/cache GPU code during these explicitly launched tests.
Use a separate codegen cache so existing environments/results remain undisturbed.

The PyTorch 2.5.1 compilation workaround uses fullgraph=True, dynamic=False
in both adapter and validator. score_mod captures concrete Python integer
clamp bounds instead of reading captured tensor shapes inside the subgraph;
ALiBi values, conversions, indexing, selection, causality, scale and thresholds
are unchanged. This targets the matching sym_size assertion in
[PyTorch issue #136914](https://github.com/pytorch/pytorch/issues/136914).
It is not GPU-verified until rerun on the A6000.

Static compilation specializes kernels for different sequence lengths/shapes,
increasing compilation time/cache usage. The kernel validator resets compiler
state between head dimensions to isolate its two five-shape batches without
raising cache limits, removing tests, or suppressing errors. Larger LongBench
diagnostics can still hit the compiler specialization limit; start with two
cases and stop on any compiler error. No automatic dense fallback is added.
Use fresh report names, e.g. a6000_kernels_static_02.json and
271m_static_02.json, to preserve earlier failures.

If this targeted workaround still fails, test PyTorch 2.6 only in a SEPARATE
environment and rerun all correctness gates and matched controls there. The
upstream issue reports success on 2.6.0.dev20241017+cu124; that is a reason
to test 2.6, not proof that this integration works. Do not upgrade the
validated LongBench environment.

From the existing GPU environment, with both repositories available:

```bash
cd ~/Sparge_Analysis
export PYTHONDONTWRITEBYTECODE=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/results/sparse_codegen_cache"
export TRITON_CACHE_DIR="$PWD/results/sparse_triton_cache"
python -B -c 'import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__, torch.version.cuda); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())'
python -B scripts/validate_sparse_gpu.py --kernel-only --report results/sparse_validation/a6000_kernels_01.json
python -B scripts/validate_sparse_gpu.py --model adpretko/celerity-271m-4k-ad0p0 --report results/sparse_validation/271m_01.json
```

Every gate must exit zero. Reports cannot be overwritten; choose a new suffix
after any failure. Checkpoint validation loads ONLY the requested manifest-pinned
checkpoint on the GPU instance. The kernel-only test downloads no weights.
Full gate includes the kernel checks again, then original/disabled exact equality,
full-reference exact equality, full-Flex comparison, sparse suffix invariance, and
a 4K model forward. Kernel profiler must show a CUDA Flex kernel.

Predefined bounds: BF16 attention max_abs <=0.03 and relative RMS <=0.02;
full-Flex versus original model max_abs <=0.5 and relative RMS <=0.02;
prefix invariance atol=rtol=1e-5. These are diagnostic gates, not a guarantee of
negligible PPL error. Do not loosen failing tolerances without investigating.
A same-backend full-attention NLL control is always measured.

## Matched 4K PPL

```bash
python -B scripts/run_sparse_longbench.py --model adpretko/celerity-271m-4k-ad0p0 --validation-report results/sparse_validation/271m_01.json --limit-cases 2
```

Default: two explicitly labeled diagnostic cases. It loads all frozen samples
to validate identities but scores only the selected cases. The launcher imports
v2 load_model/select_cases/score_one; score_one calls the UNCHANGED canonical
answer_conditional_ppl. Frozen 16-task order, indices, first references, prompt
construction, truncation, stride 128, BF16 and checkpoint revisions are reused.
The source manifest and exact prompt/answer cases are copied into new results.

Next validate 906M, then 504M if both earlier families passed:

```bash
python -B scripts/validate_sparse_gpu.py --model adpretko/celerity-906m-4k-ad0p0 --report results/sparse_validation/906m_01.json
python -B scripts/run_sparse_longbench.py --model adpretko/celerity-906m-4k-ad0p0 --validation-report results/sparse_validation/906m_01.json --limit-cases 2
python -B scripts/validate_sparse_gpu.py --model adpretko/celerity-504m-4k-ad0p0 --report results/sparse_validation/504m_01.json
python -B scripts/run_sparse_longbench.py --model adpretko/celerity-504m-4k-ad0p0 --validation-report results/sparse_validation/504m_01.json --limit-cases 2
```

Run all 80 frozen examples only after reviewing those small diagnostics:

```bash
python -B scripts/run_sparse_longbench.py --model adpretko/celerity-271m-4k-ad0p0 --validation-report results/sparse_validation/271m_01.json --limit-cases 0
```

No sweeps, 8K or 16K in this launcher.
Changing keep fraction/local blocks requires a fresh matching validation report.

Three arms run on the same loaded checkpoint: original, full backend, sparse
backend. Each arm warms ALL selected scoring windows before measurement, so
warmup/compilation wall time is reported separately. Cold compilation and useful
warmup computation are combined in that field, not misrepresented as pure
compiler time. Model-forward timing excludes loss/tokenization, includes adapter
selection/metadata/instrumentation, and is CUDA-synchronized wall time. Allocated
and reserved GPU peaks cover the measured scorer pass, including logits/loss.
No speedup is promised; compare measured results in this environment only.

Results include per-example NLL/PPL, global token-weighted NLL/PPL, sparse-original
and sparse-full delta NLL/token, per-window token-ID hashes, per-layer block counts,
pair-weighted sparsity relative to CAUSAL dense attention, backend coverage,
environment/source hashes and frozen inputs. Full-backend NLL difference >0.02
nat/token is a stop condition. Report both full-control and sparse changes.

All output directories are unique under results/sparse_longbench. Existing
directories/files are rejected; interrupted partial runs retain failure records
and completed arm/example records. Reruns create a NEW attempt, not an in-place
resume, and never overwrite validated results. A summary.json with status
complete is required for success; partial arm summaries are not campaign success.

The archived dataset script references a data archive outside its Git revision.
Existing sample identity checks are reused; exact evaluated prompt/answer strings
are additionally saved and hashed. This establishes pairing for the new run,
without claiming the dataset revision alone pins every archive byte.

## Reference fallback and decision

Use --backend reference to run the same three-arm diagnostic with dense masked
PyTorch attention. It is a correctness/reference experiment, NOT accelerated
SpargeAttention. Flex failures are raised, never silently redirected here.

Current decision: CPU reference correctness demonstrated; GO for the small A6000
validation gate. NO GO for paper claims or full LongBench runs until the GPU
gate and matched full-attention NLL control pass. These experiments can support a
paper about a documented adapted sparse method; they cannot be labeled a
reproduction of stock SpargeAttention.
