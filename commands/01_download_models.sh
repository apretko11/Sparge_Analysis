#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p models/baseline models/sparge

hf download melhoushi/gpt_cp_h640_d13_gbs48_tpp20.0_lp0.0_null \
  --local-dir models/baseline/h640

hf download melhoushi/gpt_cp_h896_d17_gbs66_tpp20.0_lp0.0_null \
  --local-dir models/baseline/h896

hf download melhoushi/gpt_cp_h1152_d23_gbs76_tpp20.0_lp0.0_null \
  --local-dir models/baseline/h1152

rm -rf models/sparge/h640 models/sparge/h896 models/sparge/h1152

cp -a models/baseline/h640 models/sparge/h640
cp -a models/baseline/h896 models/sparge/h896
cp -a models/baseline/h1152 models/sparge/h1152

find models -maxdepth 3 -type f -name "modeling_celerity.py" -print
