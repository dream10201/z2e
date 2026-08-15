#!/usr/bin/env bash
# 启动前确保 INT4 模型就位：不在就自动导出（仅 :export 镜像有导出依赖）。
# 关掉自动导出: AUTO_EXPORT=0
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models/Hunyuan-MT-7B-int4-ov}"
AUTO_EXPORT="${AUTO_EXPORT:-1}"

if [ ! -f "$MODEL_DIR/openvino_model.xml" ]; then
  if [ "$AUTO_EXPORT" = "1" ] && [ -f /app/export_int4.sh ] && python -c "import optimum.intel" 2>/dev/null; then
    echo "[entrypoint] $MODEL_DIR 不存在，开始导出（首次要下 ~15 GB 权重，N305 上约 30-60 分钟）"
    OUT="$MODEL_DIR" bash /app/export_int4.sh
  else
    cat >&2 <<EOF
[entrypoint] 找不到模型: $MODEL_DIR

当前镜像没有导出依赖（torch/optimum/nncf），装不下自动导出。三选一：

  1) 用 compose，一条命令搞定（api 会等 export 跑完再起）:
       docker compose up -d api

  2) 用 :export 镜像，它会先导出再执行你给的命令:
       docker run -d --name hunyuan-api --device /dev/dri \\
         --group-add "\$(stat -c %g /dev/dri/renderD128)" \\
         -p 8000:8000 -v /your/path/models:/models \\
         ghcr.io/dream10201/z2e:export \\
         python -m uvicorn server:app --host 0.0.0.0 --port 8000

  3) 单独跑一次导出，之后继续用这个小镜像:
       docker run --rm -v /your/path/models:/models ghcr.io/dream10201/z2e:export

模型目录要挂到 /models，导出产物约 4.5 GB。
EOF
    exit 1
  fi
fi

exec "$@"
