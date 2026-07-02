#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SHORT_PROMPT="In one paragraph, explain why sparse attention can help long-context inference."

LONG_PROMPT=$(python - <<'PY'
text = (
    "Sparse attention can help long-context inference by reducing the amount of attention work "
    "performed over long sequences while trying to preserve the most important token interactions. "
)
print(text * 40)
PY
)

for size in h640 h896 h1152; do
  echo "============================================================"
  echo "Running ${size} baseline short prompt generation"
  echo "============================================================"

  unset CELERITY_USE_SPARGE_ATTN
  unset CELERITY_SPARGE_IGNORE_BIAS
  unset CELERITY_SPARGE_IGNORE_MASK
  unset CELERITY_SPARGE_TOPK

  python scripts/generate_text.py \
    --model models/baseline/${size} \
    --prompt "$SHORT_PROMPT" \
    --max-new-tokens 64 \
    --dtype bf16 \
    --device-map auto \
    --out results/generation_${size}_baseline.json

  echo "============================================================"
  echo "Running ${size} Sparge short prompt generation"
  echo "============================================================"

  export CELERITY_USE_SPARGE_ATTN=1
  export CELERITY_SPARGE_IGNORE_BIAS=1
  export CELERITY_SPARGE_IGNORE_MASK=1
  export CELERITY_SPARGE_TOPK=0.5

  python scripts/generate_text.py \
    --model models/sparge/${size} \
    --prompt "$SHORT_PROMPT" \
    --max-new-tokens 64 \
    --dtype bf16 \
    --device-map auto \
    --out results/generation_${size}_sparge.json

  echo "============================================================"
  echo "Running ${size} baseline long prompt generation"
  echo "============================================================"

  unset CELERITY_USE_SPARGE_ATTN
  unset CELERITY_SPARGE_IGNORE_BIAS
  unset CELERITY_SPARGE_IGNORE_MASK
  unset CELERITY_SPARGE_TOPK

  python scripts/generate_text.py \
    --model models/baseline/${size} \
    --prompt "$LONG_PROMPT" \
    --max-new-tokens 64 \
    --dtype bf16 \
    --device-map auto \
    --out results/generation_${size}_baseline_long_prompt.json

  echo "============================================================"
  echo "Running ${size} Sparge long prompt generation"
  echo "============================================================"

  export CELERITY_USE_SPARGE_ATTN=1
  export CELERITY_SPARGE_IGNORE_BIAS=1
  export CELERITY_SPARGE_IGNORE_MASK=1
  export CELERITY_SPARGE_TOPK=0.5

  python scripts/generate_text.py \
    --model models/sparge/${size} \
    --prompt "$LONG_PROMPT" \
    --max-new-tokens 64 \
    --dtype bf16 \
    --device-map auto \
    --out results/generation_${size}_sparge_long_prompt.json
done
