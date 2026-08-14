#!/usr/bin/env bash
# 把 Hunyuan-MT-7B 导出成 OpenVINO INT4 IR。只需跑一次，产物落在 $OUT。
# 容器内: docker compose run --rm hunyuan-mt bash export_int4.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-tencent/Hunyuan-MT-7B}"
OUT="${OUT:-/models/Hunyuan-MT-7B-int4-ov}"
export HF_HOME="${HF_HOME:-/models/.hf}"

if [ -f "$OUT/openvino_model.xml" ]; then
  echo "[skip] $OUT 已存在"
  exit 0
fi

optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task text-generation-with-past \
  --weight-format int4 \
  --group-size 128 \
  --ratio 1.0 \
  --sym \
  --trust-remote-code \
  "$OUT"

echo "[done] -> $OUT"
du -sh "$OUT"
