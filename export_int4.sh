#!/usr/bin/env bash
# 把一个 HF decoder-only 因果语言模型导出成 OpenVINO INT4 IR。
# 产物目录由 MODEL_ID 派生：tencent/Hy-MT2-7B -> $MODELS_ROOT/Hy-MT2-7B-int4-ov
# 一般不用手动调，容器 entrypoint 发现模型缺失会自动跑。
#
#   MODEL_ID=Qwen/Qwen3-8B bash export_int4.sh
set -euo pipefail

MODEL_ID="${MODEL_ID:-tencent/Hy-MT2-7B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
WEIGHT_FORMAT="${WEIGHT_FORMAT:-int4}"
export HF_HOME="${HF_HOME:-$MODELS_ROOT/.hf}"
# 国内可以设 HF_ENDPOINT=https://hf-mirror.com
[ -n "${HF_ENDPOINT:-}" ] && echo "[export] 走镜像站 $HF_ENDPOINT"

# 目录名派生规则要和 modelmgr.dir_name_for() 保持一致
derive_out() {
  local base="${MODEL_ID%/}"
  base="${base##*/}"
  base="$(printf '%s' "$base" | tr -c 'A-Za-z0-9._-' '-')"
  printf '%s/%s-%s-ov' "$MODELS_ROOT" "$base" "$WEIGHT_FORMAT"
}
OUT="${OUT:-$(derive_out)}"

if [ -f "$OUT/openvino_model.xml" ]; then
  echo "[skip] $OUT 已存在"
  exit 0
fi

# 先导到临时目录再整体挪过去：中途被 kill 不会留下一个看着像成功的半成品
TMP="${OUT}.tmp"
rm -rf "$TMP"
mkdir -p "$(dirname "$OUT")"

QUANT_ARGS=(--weight-format "$WEIGHT_FORMAT")
if [ "$WEIGHT_FORMAT" = "int4" ]; then
  QUANT_ARGS+=(--group-size "${GROUP_SIZE:-128}" --ratio "${RATIO:-1.0}" --sym)
fi

echo "[export] $MODEL_ID -> $OUT (${QUANT_ARGS[*]})"
optimum-cli export openvino \
  --model "$MODEL_ID" \
  --task "${EXPORT_TASK:-text-generation-with-past}" \
  "${QUANT_ARGS[@]}" \
  --trust-remote-code \
  "$TMP"

test -f "$TMP/openvino_model.xml" || { echo "[export] 产物缺 openvino_model.xml" >&2; exit 1; }
printf '{"model_id": "%s", "weight_format": "%s"}\n' "$MODEL_ID" "$WEIGHT_FORMAT" > "$TMP/.z2e.json"
mv "$TMP" "$OUT"

echo "[done] -> $OUT"
du -sh "$OUT"
echo "[hint] HF 原始权重缓存在 $HF_HOME，确认没问题后可以删掉腾空间"
