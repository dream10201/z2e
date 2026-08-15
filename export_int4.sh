#!/usr/bin/env bash
# 把 Hunyuan-MT-7B 导出成 OpenVINO INT4 IR。只需跑一次，产物落在 $OUT。
# 一般不用手动调，容器 entrypoint 发现模型缺失会自动跑。
set -euo pipefail

MODEL_ID="${MODEL_ID:-tencent/Hunyuan-MT-7B}"
OUT="${OUT:-/models/Hunyuan-MT-7B-int4-ov}"
export HF_HOME="${HF_HOME:-/models/.hf}"
# 国内可以设 HF_ENDPOINT=https://hf-mirror.com
[ -n "${HF_ENDPOINT:-}" ] && echo "[export] 走镜像站 $HF_ENDPOINT"

if [ -f "$OUT/openvino_model.xml" ]; then
  echo "[skip] $OUT 已存在"
  exit 0
fi

# 先导到临时目录再整体挪过去：中途被 kill 不会留下一个看着像成功的半成品
TMP="${OUT}.tmp"
rm -rf "$TMP"
mkdir -p "$(dirname "$OUT")"

echo "[export] $MODEL_ID -> $OUT (int4 sym, group_size=128)"
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task text-generation-with-past \
  --weight-format int4 \
  --group-size "${GROUP_SIZE:-128}" \
  --ratio "${RATIO:-1.0}" \
  --sym \
  --trust-remote-code \
  "$TMP"

test -f "$TMP/openvino_model.xml" || { echo "[export] 产物缺 openvino_model.xml" >&2; exit 1; }
mv "$TMP" "$OUT"

echo "[done] -> $OUT"
du -sh "$OUT"
echo "[hint] HF 原始权重缓存在 $HF_HOME，确认没问题后可以删掉腾 ~15 GB"
