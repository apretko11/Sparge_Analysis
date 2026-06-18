#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p results/random_synthetic_tokens_withcache

for size in h640 h896 h1152; do
  for seq in 1024 4096 8192; do
    echo "Running ${size} baseline synthetic seq${seq} WITH CACHE"

    unset CELERITY_USE_SPARGE_ATTN
    unset CELERITY_SPARGE_IGNORE_BIAS
    unset CELERITY_SPARGE_IGNORE_MASK
    unset CELERITY_SPARGE_TOPK

    python scripts/bench_prefill.py \
      --model-dir models/baseline/${size} \
      --label ${size}_baseline_seq${seq}_cache \
      --seq-len ${seq} \
      --batch-size 1 \
      --warmup 2 \
      --repeats 5 \
      --dtype bf16 \
      --use-cache \
      --out results/random_synthetic_tokens_withcache/${size}_baseline_seq${seq}_cache.json

    echo "Running ${size} Sparge synthetic seq${seq} WITH CACHE"

    export CELERITY_USE_SPARGE_ATTN=1
    export CELERITY_SPARGE_IGNORE_BIAS=1
    export CELERITY_SPARGE_IGNORE_MASK=1
    export CELERITY_SPARGE_TOPK=0.5

    python scripts/bench_prefill.py \
      --model-dir models/sparge/${size} \
      --label ${size}_sparge_seq${seq}_topk0.5_cache \
      --seq-len ${seq} \
      --batch-size 1 \
      --warmup 2 \
      --repeats 5 \
      --dtype bf16 \
      --use-cache \
      --out results/random_synthetic_tokens_withcache/${size}_sparge_seq${seq}_topk0.5_cache.json
  done
done
