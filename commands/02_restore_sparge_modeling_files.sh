#!/usr/bin/env bash
set -euo pipefail

# This script assumes 01_download_models.sh has already created:
#   models/baseline/{h640,h896,h1152}
#   models/sparge/{h640,h896,h1152}
#
# Since models/ is gitignored, the patched modeling_celerity.py files are stored
# under patches/<size>/ and copied back into models/sparge/<size>/ here.

cd "$(dirname "$0")/.."

cp patches/h640/modeling_celerity_sparge.py \
   models/sparge/h640/modeling_celerity.py

cp patches/h896/modeling_celerity_sparge.py \
   models/sparge/h896/modeling_celerity.py

cp patches/h1152/modeling_celerity_sparge.py \
   models/sparge/h1152/modeling_celerity.py

for size in h640 h896 h1152; do
  echo "============================================================"
  echo "Checking Sparge patch markers for ${size}"
  echo "============================================================"

  grep -n "def _sparge_attn\|CELERITY_USE_SPARGE_ATTN\|spas_sage2_attn_meansim_topk_cuda\|sparge_result" \
    models/sparge/${size}/modeling_celerity.py
done
