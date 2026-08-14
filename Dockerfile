# syntax=docker/dockerfile:1
#
# 两个 target:
#   runtime — 只有 openvino-genai，日常翻译用，镜像小
#   export  — 额外装 torch/optimum-intel/nncf，首次把 HF 模型转成 INT4 IR 用
FROM ubuntu:24.04 AS base

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

ENV HF_HOME=/models/.hf \
    OV_CACHE=/models/.ovcache \
    PYTHONUNBUFFERED=1
WORKDIR /app


FROM base AS runtime
COPY requirements-runtime.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip && pip install -r /tmp/requirements-runtime.txt
COPY translate.py bench.py /app/
CMD ["python", "translate.py", "--device", "GPU"]


FROM runtime AS export
COPY requirements-export.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements-export.txt
COPY export_int4.sh /app/
CMD ["bash", "export_int4.sh"]
