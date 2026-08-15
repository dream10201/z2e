#!/usr/bin/env bash
# 启动前确保 INT4 模型就位：不在就自动导出（仅 :export 镜像有导出依赖）。
# 关掉自动导出: AUTO_EXPORT=0
set -euo pipefail

MODEL_ID="${MODEL_ID:-tencent/Hunyuan-MT-7B}"
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

if [ ! -f "$MODEL_DIR/openvino_model.xml" ]; then
  if [ "$AUTO_EXPORT" != "1" ]; then
    echo "[entrypoint] 找不到模型 $MODEL_DIR，且 AUTO_EXPORT=0" >&2
    exit 1
  elif [ -f /app/export_int4.sh ] && python -c "import optimum.intel" 2>/dev/null; then
    echo "[entrypoint] $MODEL_DIR 不存在，开始导出 $MODEL_ID（7B 首次要下 ~15 GB，N305 上约 30-60 分钟）"
    OUT="$MODEL_DIR" bash /app/export_int4.sh
  else
    cat >&2 <<EOF
[entrypoint] 找不到模型: $MODEL_DIR  (MODEL_ID=$MODEL_ID)

当前镜像没有导出依赖（torch/optimum/nncf），装不下自动导出。三选一：

  1) 用 compose，一条命令搞定（api 会等 export 跑完再起）:
       MODEL_ID=$MODEL_ID docker compose up -d api

  2) 用 :export 镜像，它会先导出再执行你给的命令:
       docker run -d --name llm-api --device /dev/dri \\
         --group-add "\$(stat -c %g /dev/dri/renderD128)" \\
         -p 8000:8000 -v /your/path/models:/models \\
         -e MODEL_ID=$MODEL_ID \\
         ghcr.io/dream10201/z2e:export \\
         python -m uvicorn server:app --host 0.0.0.0 --port 8000

  3) 单独跑一次导出，之后继续用这个小镜像:
       docker run --rm -v /your/path/models:/models -e MODEL_ID=$MODEL_ID \\
         ghcr.io/dream10201/z2e:export

模型目录要挂到 $MODELS_ROOT。也可以先起服务再用 POST /admin/pull 后台导出。
EOF
    exit 1
  fi
fi

exec "$@"
