# syntax=docker/dockerfile:1
#
# 单镜像：Intel iGPU 驱动 + OpenVINO GenAI 运行时 + 模型导出工具链。
# 早前拆过 runtime/export 两个 target，实测导出依赖只多 ~340 MB（压缩），
# 换来的却是两个 tag、entrypoint 降级分支和一套 compose 编排，不划算。
FROM ubuntu:24.04

ARG NEO_VER=26.27.39122.11
ARG GMM_VER=22.10.0
ARG IGC_VER=2.38.2
ARG IGC_BUILD=22051

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
        ca-certificates curl \
        ocl-icd-libopencl1 libze1 clinfo \
    && rm -rf /var/lib/apt/lists/*

# Intel iGPU 用户态运行时（NEO OpenCL + Level Zero）。
# Debian 13 已经不提供 intel-opencl-icd，Ubuntu 自带版本又偏旧，所以直接用 upstream deb。
RUN set -eux; \
    cd /tmp; \
    NEO=https://github.com/intel/compute-runtime/releases/download/${NEO_VER}; \
    IGC=https://github.com/intel/intel-graphics-compiler/releases/download/v${IGC_VER}; \
    curl -sSLO "${IGC}/intel-igc-core-2_${IGC_VER}%2B${IGC_BUILD}_amd64.deb"; \
    curl -sSLO "${IGC}/intel-igc-opencl-2_${IGC_VER}%2B${IGC_BUILD}_amd64.deb"; \
    curl -sSLO "${NEO}/libigdgmm12_${GMM_VER}_amd64.deb"; \
    curl -sSLO "${NEO}/intel-opencl-icd_${NEO_VER}-0_amd64.deb"; \
    curl -sSLO "${NEO}/libze-intel-gpu1_${NEO_VER}-0_amd64.deb"; \
    curl -sSLO "${NEO}/intel-ocloc_${NEO_VER}-0_amd64.deb"; \
    dpkg -i ./*.deb; \
    rm -f ./*.deb

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH=$VIRTUAL_ENV/bin:$PATH

# 运行时依赖和导出依赖分成两层：改代码时前一层能命中缓存
COPY requirements-runtime.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && pip install -r /tmp/requirements-runtime.txt
COPY requirements-export.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-export.txt

ENV HF_HOME=/models/.hf \
    OV_CACHE=/models/.ovcache \
    MODELS_ROOT=/models \
    PYTHONUNBUFFERED=1
WORKDIR /app

COPY translate.py bench.py server.py modelmgr.py export_int4.sh docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh /app/export_int4.sh
EXPOSE 8000

# entrypoint 会先确认模型在不在，不在就自动导出
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
