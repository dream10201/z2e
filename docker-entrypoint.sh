#!/usr/bin/env bash
# 启动前确保模型就位：不在就自动导出。关掉自动导出用 AUTO_EXPORT=0。
set -euo pipefail

MODEL_ID="${MODEL_ID:-tencent/Hy-MT2-7B}"
MODELS_ROOT="${MODELS_ROOT:-/models}"
WEIGHT_FORMAT="${WEIGHT_FORMAT:-int4}"
AUTO_EXPORT="${AUTO_EXPORT:-1}"

# 目录名派生规则要和 modelmgr.dir_name_for() 保持一致
if [ -z "${MODEL_DIR:-}" ]; then
  _base="${MODEL_ID%/}"; _base="${_base##*/}"
  _base="$(printf '%s' "$_base" | tr -c 'A-Za-z0-9._-' '-')"
  MODEL_DIR="$MODELS_ROOT/${_base}-${WEIGHT_FORMAT}-ov"
fi
export MODEL_DIR

if [ ! -f "$MODEL_DIR/openvino_model.xml" ] && [ ! -f "$MODEL_DIR/openvino_language_model.xml" ]; then
  if [ "$AUTO_EXPORT" != "1" ]; then
    cat >&2 <<EOF
[entrypoint] 找不到模型: $MODEL_DIR  (MODEL_ID=$MODEL_ID)
AUTO_EXPORT=0，不会自动导出。去掉这个变量，或手动跑:
    MODEL_ID=$MODEL_ID bash /app/export_int4.sh
EOF
    exit 1
  fi
  echo "[entrypoint] $MODEL_DIR 不存在，开始导出 $MODEL_ID"
  echo "[entrypoint] 7B 首次要下 ~15 GB 权重再量化，N305 上约 30-60 分钟；导完会自动继续启动"
  OUT="$MODEL_DIR" bash /app/export_int4.sh
fi

exec "$@"
