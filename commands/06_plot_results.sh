#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/plot_prefill_results_from_dir.py \
  --results-dir results/random_synthetic_tokens_nocache \
  --plots-dir plots/random_synthetic_tokens_nocache \
  --title-suffix "synthetic tokens, use_cache=False"

python scripts/plot_prefill_results_from_dir.py \
  --results-dir results/random_synthetic_tokens_withcache \
  --plots-dir plots/random_synthetic_tokens_withcache \
  --title-suffix "synthetic tokens, use_cache=True"

python scripts/plot_prefill_results_from_dir.py \
  --results-dir results/text_prompt_withcache \
  --plots-dir plots/text_prompt_withcache \
  --title-suffix "repeated text prompt, use_cache=True"
