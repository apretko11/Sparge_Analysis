# Original SpargeAttention with Celerity ALiBi: source audit and decision

**Decision: Option C. No accelerated integration is implemented or qualified.**
The reviewed full-sequence Sparge selector is intrinsically dependent on future
queries and sequence length. Its pooling, quantization, and filtering rules need
changes to satisfy strict causal prefix independence. A score-mask fix cannot
resolve this, and changing these rules would require a documented Sparge-derived
method rather than a claim of unchanged original selection.

The running 63-evaluation Hyperstack campaign was not accessed. All work is local
and isolated here. No existing evaluator, attention implementation, tests,
experiment archive, or Perplexity_LongBench file was modified. No environment
changes, checkpoint downloads, or GPU execution were performed.

## Revision and evidence

Upstream: `https://github.com/thu-ml/SpargeAttn`.
Audited commit: **`ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a`**,
dated 2026-02-25, resolved from upstream main during this investigation.
Local Sparge_Analysis baseline: `5c882e3584acc4c1215531aa457f0a56d432a690`.

[source_pin.json](source_pin.json) records SHA256 hashes of 13 unmodified source
files and the local protected implementation/tests. `upstream/` is a 176 KB
inspection subset with the original Apache-2.0 license, not an installed package
or a replacement backend. The originals remain unchanged.

Primary references, pinned to that commit:

- [Recommended API and dispatch](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/core.py#L105-L172).
- [Pooling, similarity, and quantization](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L163-L222).
- [Selector and causal block map](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/utils.py#L282-L421).
- [CUDA API signature](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/attn_cuda.h#L32-L45),
  [SM80/86 dispatch](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cu#L188-L368),
  [CUDA tile loop](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/csrc/qattn/qk_int_sv_f16_cuda_sm80.cuh#L210-L558).
- [Triton example](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/Triton_SpargeAttn/triton_kernel_example.py),
  [Triton quantization scale](https://github.com/thu-ml/SpargeAttn/blob/ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a/spas_sage_attn/quant_per_block.py#L51-L98).
- [Paper](https://arxiv.org/html/2502.18137v1), sections 3.2–3.5, describes the
  two filtering stages. The recommended implementation is the authority for
  this audit; its current top-k rule is not interchangeable with the paper's
  cumulative-mass selection.

Celerity arithmetic was inspected in the already cached, unmodified
`results/sparse_source/modeling_celerity.py`, SHA256
`09ba09638f0b832efffd09951895455e043b7dfd152d344ff048bada8dce235c`:
ALiBi at lines 329–352, QKV layout at 458–460, eager attention at 484–496.
The historical files under `patches/` use a different model-source implementation;
they are not substituted for this validated checkpoint source.

## What the recommended implementation actually does

On A6000/SM86, `spas_sage2_attn_meansim_topk_cuda`:

1. Uses `[batch, heads, tokens, head_dim]` (HND); optionally rearranges NHD.
   It requires query length at least 128 and head dimension 64 or 128.
2. Computes a per-head mean over the entire K sequence. The fused primitive
   centers K, pools full Q blocks of 128 and K blocks of 64, and computes
   within-block mean pairwise cosine similarity, including the diagonal.
3. Scores pooled Q/K with `1/sqrt(d)`, masks disallowed key blocks, softmaxes
   and sorts. Top-k takes `floor(topk * total_K_blocks)` (minimum one), rather
   than a fraction of historical blocks. Low-similarity Q/K blocks are retained
   conservatively. Optional attention sink keeps block zero.
4. Fuses pooling with per-block symmetric INT8 Q/K quantization. Scales are
   block maximum absolute value divided by 127 plus `1e-7`, with signed rounding.
   For BF16 inputs, Q/K remain BF16 before quantization; V is converted to FP16.
   Ampere uses INT8 QK and FP16 PV, not the FP8-V branch used on newer hardware.
5. Converts selected blocks to an ascending delta-coded LUT. CUDA follows the
   LUT and selected-block count: omitted QK tiles are actually skipped. For
   interior selected tiles, a warp-level maximum test can additionally skip
   exponentiation/denominator/PV work. Final boundary iterations have separate
   mask handling. Source support for skipping is not a measurement of speedup.

The default similarity threshold is `-0.1`. In exact arithmetic the average
Gram entry of normalized vectors is the squared norm of their mean, hence
nonnegative. Thus the default effectively passes complete, nonzero blocks;
it should not be described as a strong active similarity filter. Degenerate
vectors, padding, finite precision, or different thresholds require separate care.

## Confirmed correctness problems

| Concern | Finding and consequence |
| --- | --- |
| `attn_mask` | Present in the API signature but never read in the recommended function. Neither additive ALiBi nor padding masks reach the selector or CUDA kernel. |
| Hard-coded `False` | It is exactly the low-level `is_causal` parameter, not a layout or instrumentation flag. The CUDA dispatcher therefore selects `MaskMode::kNone`. |
| Coarse causality | Selection allows K blocks starting before the END of each Q block. For the first 128 queries, both first 64-token K blocks can be present. Without token-level masking, token 0 can directly attend tokens 1–127. |
| Low-level causal support | CUDA has a causal template and masks its last two selected tiles, consistent with sorted LUTs and the 128/64 geometry. Passing `True` is necessary but not sufficient for prefix safety; arbitrary LUTs, tails, empty rows, and cached offsets still need validation. |
| Full Q-block pooling | A shared selection decision for an early row sees later Q rows in its block. A correct final triangular score mask does not remove this dependency. |
| Total-length top-k | Appending future K blocks changes the budget for earlier Q blocks. This is a second independent violation. |
| Global K smoothing | Before quantization, subtracting a common mean is a row-constant logit shift and leaves softmax unchanged. However, it changes similarity gates and INT8 scales; those effects depend on future K. |
| Block INT8 scales | Future Q rows can change the quantization of an earlier Q row. The partly visible current K block has the analogous problem. These remain even at `topk=1`. |
| Online PV gate | Its maximum reduction couples multiple query rows within a warp. A strict prefix audit must also address this second filter. With threshold 50, the affected contribution may be tiny; that does not justify ignoring the dependency. |
| Disabling smoothing | `smooth_k=False` leaves `km` uninitialized in the recommended wrapper at this revision; it is not a working escape hatch. |

Older `spas_sage_attn_meansim_*` wrappers pass a computed causal flag. That does
not repair their selector, smoothing, or quantization dependencies; this audit
does not silently switch to them.

This is particularly consequential for teacher-forced answer perplexity: later
answer tokens are present in the input, and selection can expose information
from them to earlier scored predictions. Attention over retained past values
can still leak future information through the choice of those values.

## CPU witnesses and what they establish

[cpu_findings.json](cpu_findings.json) contains deterministic results from
[cpu_audit.py](cpu_audit.py). It executes the **unchanged upstream Python
selection function**, loaded from its pinned AST, with explicit CPU substitutes
for pooling/quantization and map-fill primitives. It is not a GPU emulator and
does not import the installed Sparge package, Triton, or `celerity_sparse`.

- At length 256, Q tokens 0–128, all K, and all V are fixed. Changing only
  Q[129:256] switches the historical block for query 128 from block 0 to block 1.
  Applying a correct final causal mask still changes its output by **1.0**.
- Extending an unchanged 256-token prefix to length 512 changes a previous
  query block's selection count from **one to two**, at top-k 0.25.
- Changing a future row's magnitude changes an earlier row's dequantized Q.
- Changing only future keys changes an old key block's similarity gate at
  threshold 0.6, before any quantization or sparse attention computation.
- Mathematical tests establish causal score masking, `1/d` scaling, ALiBi
  equivalences, changed historical rankings, heterogeneous-block retention,
  and nonzero quantization error with full selection.

The query-pooling and length witnesses use the upstream default similarity
threshold, valid full blocks, and well-separated rankings. They do not depend
on ties, partial-block bugs, approximate GPU rounding, or our previous simplified
selector. Dense PyTorch appears only as a small correctness oracle; it is not
presented as an accelerated implementation.

## ALiBi: computation and selection are separate changes

For allowed causal positions `j <= i`, Celerity uses

`score(i,j) = dot(q_i,k_j)/d - m_h*(i-j)`.

Since `-m_h*i` is constant across a row, exact-arithmetic softmax equivalently
uses `dot(q_i,k_j)/d + m_h*j`, with the same causal and padding exclusions.
Thus a kernel can consume just per-head slopes and absolute positions instead
of an entire quadratic additive tensor. This equivalence does **not** promise
bitwise equality to Celerity's BF16 addition order. Quantization adds further
expected approximation error, distinct from ignored bias or information leaks.

CUDA changes would thread slopes/offsets through Python, bindings/declarations,
dispatch, and the attention kernel. Add the bias to correctly dequantized and
`1/d`-scaled logits **before** maxima, online softmax, and the PV filter. The
kernel operates in base-2 exponent units, so the natural-log bias needs the
`log2(e)` conversion. Interior and boundary paths currently apply dequantization
at different points; inserting an unscaled bias into raw INT8 dot products is
incorrect. Padding must exclude keys before normalization, not merely zero V.

For a fully historical rectangular pair of blocks, the natural pooled estimate
is `mean(Q) dot mean(K)/d + m_h*(mean(key_positions)-mean(query_positions))`.
For example, content scores 1.0 and 0.5 favor the older of two historical blocks;
with slope 0.02 and a 64-position separation, the newer block wins. ALiBi must
therefore enter selection if selection is intended to estimate the actual
attention importance. The change can retain pooling, similarity tests, ranking,
and the LUT representation, but necessarily changes selected blocks.

For a partially causal/diagonal tile, averaging the absolute-distance bias over
the full rectangle is not the same as averaging only valid causal pairs. A
prefix-safe design must define both the valid summary and selection budget.
Simply adding a bias to upstream pooled scores leaves the future-query problem.
Also, upstream's `scale` reaches the CUDA attention computation but not its
pooled score: with ALiBi, the content/bias balance must use `1/d` in both places.
Without ALiBi, positive scaling alone preserves strict top-k ordering in exact
arithmetic, although it changes probabilities/CDF selection and rounding.

Appending positional channels to Q/K is another exact-arithmetic construction:
`q'=[q,1]`, `k'=[k,d*m_h*j]`, scale `1/d`. It is unsuitable as a drop-in repair:
64 becomes 65 (requiring padding to 128), 128 becomes 129 (outside supported
dimensions), and the new coordinates change similarity and quantization scales.

## Option assessment and estimated scope

**A — modified upstream CUDA:** closest to original optimized execution, but
requires the mask/scale/bias plumbing above plus decisions about pooling,
prefix-local budgets, centering, quantization, and warp-level filtering. These
are algorithmic choices, not just integration fixes. Using one full-block
mean to reproduce all original selections while requiring every row to ignore
future rows is incompatible in general, as the counterexample demonstrates.
Running the original rule separately for every prefix is conceivable, but loses
the shared tile decision and needs a new execution strategy; no acceleration
should be assumed.

**B — adapted selector plus Triton:** more practical for a later experimental
ALiBi kernel because per-element positions and masks are explicit. The supplied
Triton example is not a complete Sage2 replacement: its PV-threshold conditional
is commented out, its wrapper does not forward `scale` to quantization, and its
relative imports assume files not present in that example directory. It does
contain genuine conditional block skipping and token-level causal logic.
Preserving the old selector would still fail the prefix witnesses; replacing it
with our existing custom selector would not satisfy this task.

Tail review is mandatory in either path. In `utils.py`, masked loads omit an
explicit `other`, padded/zero vectors can divide by zero during normalization,
and the unfused-mean branch does not explicitly zero masked Q lanes. The Triton
attention example does not explicitly set invalid K logits to `-inf`, and its
causal diagonal loop can exceed the final K-block-map extent for short tails.
CUDA has predicated tail loads and a final key-bound mask, but this does not fix
upstream pooling. These are source-level risks, not claims of measured CUDA
failures. CPU probes intentionally use valid-row tail handling and cannot
certify these kernels.

**C — chosen:** preserve the originals and stop before speculative kernel work.
No new Celerity adapter, production attention backend, or decode dispatch is
installed. A future adapter must explicitly route query lengths below 128 and
cached decoding to original Celerity, including cache offsets and its ALiBi;
it must not retry a failed prefill kernel through a silent fallback. Unsupported
padding/layouts should fail preflight until tested.

Engineering estimate, not a delivery promise: score-only ALiBi/causal plumbing
and tests on one architecture could take roughly 2–4 focused GPU-access days.
Resolving prefix semantics, quantization, tails, decode behavior, and full-model
validation is more plausibly **1–3 weeks**, with acceleration unproven. Triton
is the more readable research path; CUDA offers greater implementation fidelity
and higher register-layout/dispatch risk. Neither is a small defensible patch
for a ten-day paper deadline without narrowing the scientific claims.

## Commands and later GPU validation

CPU checks, using the existing environment only:

```bash
cd ~/Sparge_Analysis
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/lb_ppl_cpu/bin/python -B \
  -m unittest discover -s research/sparge_alibi/tests -v
PYTHONDONTWRITEBYTECODE=1 ~/miniconda3/envs/lb_ppl_cpu/bin/python -B \
  research/sparge_alibi/cpu_audit.py
```

`--report PATH` optionally writes an exclusive-create JSON within this research
directory or `results/sparge_alibi_audit/`. The saved `cpu_findings.json` is never
overwritten. The existing sparse suite was also run unchanged.

Results: **15 new audit tests passed; all 37 existing sparse/batch tests passed**
(52 total). The GPU probe's help/preflight paths and Python syntax were checked;
its CUDA execution remains untested. Protected-file SHA256 checks and repository
diffs confirmed that the validated code, tests, and archives were unchanged.

**After the current Hyperstack campaign has finished**, in its existing GPU
environment, this is the minimal opt-in upstream diagnostic:

```bash
cd ~/Sparge_Analysis
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -B research/sparge_alibi/gpu_probe.py
```

It requires an already installed source checkout at the exact audited Sparge
revision and cached weights for `adpretko/celerity-271m-4k-ad0p0` revision
`be032b277561058f3215d0862ba282c164e73d35`. If either is absent or different, it
stops without installing, updating, or downloading anything. Do not change the
running campaign's environment to meet these requirements. Existing extension
binary hashes are recorded; matching source does not prove binary build inputs.

The probe captures original layer-0 QKV at lengths 128, 256, and 257, confirms
the dense test oracle against unmodified Celerity, measures full-selection error,
checks whether masks affect output and future V affects token zero, records
sparse CUDA profiler events, and requests the upstream PV execution counters.
It preserves the existing 0.03 max-absolute / 0.02 relative-RMS kernel tolerances
as reported numerical gates. Correctness failures remain disqualifying. This
is a **falsification diagnostic, not validation of a working integration**:
completed audits exit 1 with `not_qualified`; incomplete probes exit 2. Reports
are saved in fresh directories under `results/sparge_alibi_audit/`, separate
from the running campaign. No end-to-end speedup is reported.

A later real integration still requires: device tests at block boundaries and
odd lengths, suffix perturbation and prefix-extension invariance, safe empty
rows, padding and decode/cache checks, isolated quantization/full-selection
controls, verified block skipping, all-layer 271M/logit checks, and then the
unchanged LongBench scorer/control gates. Only after these pass should repeated,
warmup-separated end-to-end timings be compared to original Celerity, including
selection/quantization overhead. 504M/906M remain separate validation targets.

## Deliverables and limitations

- Added only this isolated directory: assessment, source pin, original upstream
  source subset/license, CPU audit/report, independent tests, and later-only GPU
  diagnostic. No accelerated integration or custom replacement algorithm.
- Original selection is preserved in the source snapshot and the extracted
  Python decision body used by the CPU audit. CPU pooling/tail primitives are
  explicitly idealized, not certified GPU numerics.
- A faithful strict-prefix integration is blocked by original semantics, not
  merely by the absence of local GPU access. GPU tests can quantify the pinned
  implementation's failures but cannot remove those structural dependencies.
- The ongoing reference campaign remains evidence about its documented
  Sparge-inspired method. Historical timings that ignored bias/masks cannot be
  paired with those PPL results to claim faithful accelerated SpargeAttention.
