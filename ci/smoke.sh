#!/usr/bin/env bash
# 镜像冒烟测试。CI runner 上没有 iGPU，所以只验证：
#   1) Intel 驱动库和 OpenCL ICD 装齐了
#   2) OpenVINO / openvino-genai 能 import，CPU 设备可用
# GPU 路径要在真机（N305）上跑 clinfo -l 和 bench.py 确认。
set -euo pipefail

echo "--- Intel GPU 用户态库 ---"
ls -1 /usr/lib/x86_64-linux-gnu/ | grep -E "libze_intel_gpu|libigdrcl|libigdgmm|libze_loader"
echo "--- OpenCL ICD ---"
ls -1 /etc/OpenCL/vendors/
# ICD 文件里写的路径必须真的存在，否则 clinfo 会静默认不到设备
while read -r icd; do
  echo "$icd -> $(cat "$icd")"
  test -e "$(cat "$icd")" || { echo "ICD 指向的库不存在"; exit 1; }
done < <(find /etc/OpenCL/vendors -name '*.icd')
echo "--- ocloc ---"
command -v ocloc >/dev/null && echo "ocloc ok"

echo "--- OpenVINO ---"
python - <<'PY'
import openvino as ov
import openvino_genai

core = ov.Core()
devs = core.available_devices
print("OpenVINO", ov.__version__)
print("openvino_genai", openvino_genai.__version__)
print("devices:", devs)
assert "CPU" in devs, "CPU 插件没起来"
PY

echo "--- 导出工具链 ---"
python - <<'PY'
import nncf, optimum.intel, torch
print("torch", torch.__version__, "/ nncf", nncf.__version__, "-> 自动导出可用")
PY
command -v optimum-cli >/dev/null && echo "optimum-cli ok"

echo "--- HTTP API（stub 掉模型）---"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd /app 2>/dev/null || cd "$HERE/.."
python "$HERE/test_api.py"

echo "--- 并发一致性 ---"
python "$HERE/test_concurrency.py"

echo "--- 冒烟通过 ---"
