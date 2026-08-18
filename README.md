# Zero to Endpoint

一条命令，从零到一个能调的 OpenAI 兼容端点：自动下载模型、量化成 INT4、
起好 HTTP 服务。基于 OpenVINO GenAI，面向 Intel iGPU / CPU 的边缘小机器
（主力目标是 i3-N305：8 核 Gracemont + UHD 32EU Xe-LP）。
默认模型是腾讯 Hy-MT2-7B，换 `MODEL_ID` 即可换成任何 decoder-only 或多模态模型。

**支持范围**：
- decoder-only 因果语言模型（`openvino_genai.LLMPipeline`）——Qwen / Llama /
  Mistral / Hunyuan / Seed-X 等；
- 多模态 VLM（`openvino_genai.VLMPipeline`）——Qwen-VL / Qwen3.5 这类
  `model_type` 只能按 image-text-to-text 导出的架构。既能当纯文本模型聊，
  也支持 OpenAI 多模态消息格式发图（见「HTTP API」）。导出 task 由 optimum
  自动按架构推断（`EXPORT_TASK=auto`）。

**不支持 seq2seq 翻译模型**（NLLB、M2M100、Opus-MT、T5），GenAI 没有对应的
text2text pipeline，这类模型在导出阶段就会失败。

镜像由 GitHub Actions 构建并推到 `ghcr.io/dream10201/z2e:latest`，
压缩后约 **0.6 GB**，包含 Intel iGPU 驱动、OpenVINO GenAI 运行时和模型导出工具链。

## 快速开始

```bash
git clone https://github.com/dream10201/z2e && cd z2e

# render 组 id 要和宿主机一致，否则容器里打不开 /dev/dri/renderD128
echo "RENDER_GID=$(stat -c %g /dev/dri/renderD128)" > .env

# 起服务。模型不在会自动先导出（下 ~15 GB 权重再量化，N305 上约 30-60 分钟），
# 导完自动接着启动 API。
docker compose up -d api
docker compose logs -f            # 看导出进度
```

## 换模型

`MODEL_ID` 收 HF repo id，目录名自动派生（`Qwen/Qwen3-8B` → `Qwen3-8B-int4-ov`）：

```bash
echo "MODEL_ID=Qwen/Qwen3-8B" >> .env
docker compose up -d api
```

多个模型可以共存在 `./models` 下，服务启动时扫描，API 里按需切换：

```bash
curl localhost:8000/v1/models                      # 列出已导出的
curl localhost:8000/admin/load -H 'Content-Type: application/json' \
     -d '{"model": "Qwen/Qwen3-8B"}'               # 显式预热

# 或者直接在请求里指定，服务会自动切过去
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
     -d '{"model": "Qwen3-8B-int4-ov", "messages": [{"role":"user","content":"你好"}]}'
```

`model` 字段认三种写法：目录名 `Qwen3-8B-int4-ov`、HF repo id `Qwen/Qwen3-8B`、裸名字 `Qwen3-8B`。

**一次只驻留一个模型**——N305 上也塞不下两个 7B。切换要卸载旧的再加载新的，
GPU 上首次编译 kernel 一两分钟（之后走 `OV_CACHE`，几秒）。所以别在生产流量里频繁切。

**请求没导出的模型会自动触发导出**：只要它在允许列表里（见下），服务就启动
后台导出并回 503 + `Retry-After`，进度看 `GET /admin/pull`，导完重发请求即可。
也可以显式触发：

```bash
curl localhost:8000/admin/pull -H 'Content-Type: application/json' \
     -d '{"model": "Qwen/Qwen3-8B"}'
curl localhost:8000/admin/pull                     # 轮询进度，带日志尾巴
```

**允许列表**：`MODEL_ALLOWLIST` 不设时，本地已导出的模型随便切，但 API 触发
导出只放行内置的 N305 友好清单（Qwen3 全系、Qwen2.5-7B、Phi-4-mini、
DeepSeek-R1-Distill-Qwen-7B、Hy-MT2-7B 等 7B 级以下开放模型，
见 `modelmgr.N305_SAFE_MODELS`）。设了就严格按列表来——切换和导出都只认
列表里的（逗号分隔 repo id，`*` 完全放开）。

量化参数也是环境变量（`WEIGHT_FORMAT` / `GROUP_SIZE` / `RATIO`），完整列表见下面「环境变量」一节。


## HTTP API

```bash
docker compose up -d api      # 默认 8000 端口，改 API_PORT
curl localhost:8000/health
```

OpenAPI 文档在 http://localhost:8000/docs ，schema 在 `/openapi.json`。

| 端点 | 作用 |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI 兼容对话，`stream=true` 走 SSE |
| `GET /v1/models` | 列出 `/models` 下已导出的模型 |
| `GET /health` | 当前模型、设备、可用模型列表、加载耗时 |
| `POST /admin/load` | 显式预热/切换模型 |
| `POST /admin/pull` | 后台导出一个 HF 模型 |
| `GET /admin/pull` | 查导出进度（带日志尾巴） |

`/v1/chat/completions` 请求参数：`model`（可空，认目录名 / repo id / 裸名字，
给了就运行时切换）、`messages`、`max_completion_tokens` / `max_tokens`
（新旧两个名字都认，前者优先，默认 2048、上限 32768）、`temperature`（默认 0，走贪心）、
`top_p`、`stop`（停止序列，字符串或数组）、`repetition_penalty`（默认 1.05）、`stream`、
`stream_options.include_usage`（流式结尾补一个带 usage 的 chunk）、
`tools` / `tool_choice`（见下）。响应里 `finish_reason` 会如实报
`length`（打满 max_tokens）/ `stop` / `tool_calls`，`usage` 带真实的
prompt/completion token 数。`n>1`、`logprobs`、`response_format`、
`presence/frequency_penalty` 这些不支持，传了会被忽略。

**OpenAI 兼容**，现成的客户端直接指过来就行：

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Hy-MT2-7B-int4-ov",
  "messages": [{"role": "user", "content": "把下面的文本翻译成中文，不要额外解释。\n\nEdge inference cuts latency."}],
  "stream": false
}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = c.chat.completions.create(
    model="Hy-MT2-7B-int4-ov",
    messages=[{"role": "user", "content": "把下面的文本翻译成中文，不要额外解释。\n\nEdge inference cuts latency."}],
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
```

`stream=true` 走标准 SSE。服务不做 prompt 包装，消息原样过模型自带的
chat template，怎么用模型由客户端决定。Cline 这类 agent 客户端发的 content
分段数组和 `tool` 角色也能收。

**发图（VLM 模型）**：content 分段数组里的 `image_url` 按标准 OpenAI 格式收，
`data:` base64 和 `http/https` 都认，图片在消息里的位置会保留：

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "这张图里有什么？"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
  ]}]
}'
```

当前模型是纯文本模型时图片段照旧丢弃、只留文本（Cline 这类客户端会给文本模型
发截图，硬拒会把它们打断）；当前模型是不是 VLM 看 `GET /health` 的
`multimodal` 字段。图片下载上限 20 MB（`IMAGE_MAX_BYTES`）、超时 15s
（`IMAGE_FETCH_TIMEOUT`）。

**工具调用（tools）**：标准 OpenAI 协议——请求带 `tools` 函数签名，
响应给 `tool_calls` + `finish_reason: "tool_calls"`，流式下 `tool_calls` 作为
delta 整块给出。`tool_choice` 支持 `"auto"`（默认）/ `"none"` / `"required"` /
指定函数。实现方式是文本层的：函数签名以 Hermes/Qwen 风格注入 system prompt，
模型输出的 `<tool_call>{...}</tool_call>` 解析回标准格式，历史里的
`tool_calls` / `tool` 轮按同一格式渲染回去。**前提是模型按这个格式训练过**
（Qwen、Hermes 系都是）；没练过工具调用的模型（比如 Hy-MT2 这种翻译特化模型）
给了 tools 也不会用。

**并发是串行的**：`LLMPipeline` 不是线程安全的，而且 N305 上并行解码只会互相拖慢，
所以服务内部加了一把全局锁，请求排队处理。锁的范围是**从解析模型一直到生成结束**——
否则并发请求指定不同 `model` 时，会出现用 B 模型生成却按 A 模型上报的情况
（`ci/test_concurrency.py` 覆盖了这个）。排队超过 `GEN_WAIT_SECONDS`（默认 300）
直接回 503 + `Retry-After`，别让客户端无限悬着。客户端中途断连会立刻叫停生成、
释放锁。要吞吐就上批处理，别靠并发。

**前缀缓存**（可选）：agent 类客户端每轮都带完整历史，prompt 前缀递增，
`OV_PREFIX_CACHING=1` 走 continuous-batching 后端，每轮只 prefill 新增部分，
长对话 TTFT 明显降。代价是 KV cache 常驻内存，紧张的机器上可能装不下，默认关。

**鉴权**：`/v1/*` 无鉴权；设了 `ADMIN_TOKEN` 后 `/admin/*` 要带
`Authorization: Bearer <token>`。别直接暴露到公网，要么绑
`127.0.0.1:8000:8000`，要么前面挡一层反代。

不用 compose 也是一条命令，entrypoint 会先补上模型再启动服务：

```bash
docker run -d --name z2e --device /dev/dri \
  -p 8000:8000 -v /your/path/models:/models \
  -e MODEL_ID=Qwen/Qwen3-8B \
  ghcr.io/dream10201/z2e
```

已经知道模型在、不想让它自动下载，加 `-e AUTO_EXPORT=0`——这时模型缺失会直接拒绝启动。

## 关键点

**iGPU 驱动不能靠发行版装。** Debian 13 已经把 `intel-opencl-icd` 移出仓库，
Ubuntu 24.04 自带的版本也偏旧。镜像里直接装 `intel/compute-runtime` 和
`intel/intel-graphics-compiler` 的 upstream deb（NEO 26.27 + IGC 2.38 + gmmlib），
再配 Ubuntu 自带的 `ocl-icd-libopencl1` + `libze1`。验证：

```bash
docker run --rm --device /dev/dri --entrypoint clinfo \
  ghcr.io/dream10201/z2e -l
# Platform #0: Intel(R) OpenCL Graphics
#  `-- Device #0: Intel(R) UHD Graphics
```

**`--group-add` 多数情况下并不必需。** 只有「容器以非 root 用户运行」**且**
「宿主机 `renderD128` 是 0660」两个条件同时成立时才需要它。容器默认以 root 跑，
root 有 `CAP_DAC_OVERRIDE`，直接绕过权限位；另外不少系统上 `renderD128` 是 0666，
谁都能开。留着它是为了将来给镜像加 `USER`、或换到权限更严的机器时不会挂。

**iGPU 没有独立显存**，走系统内存（N305 上 OpenVINO 报可用约 28.8 GB），
7B INT4（~4.5 GB 权重）完全放得下，不需要 offload。

**首次在 GPU 上加载要编译 kernel**，慢一两分钟。`OV_CACHE=/models/.ovcache`
已经在 compose 里配好，第二次启动直接读缓存。

**CPU 侧的线程数**：N305 是 8 个物理核、无超线程。默认
`INFERENCE_NUM_THREADS=4`（用 `OV_THREADS` 覆盖），过度并行在这颗 U 上反而掉速。

**GPU 侧额外开了两项**：`DYNAMIC_QUANTIZATION_GROUP_SIZE=32`（激活动态量化）和
`KV_CACHE_PRECISION=u8`（KV cache 压到 u8，省带宽）。32EU 的瓶颈基本在内存带宽，
这两项影响明显。

## 环境变量

服务运行时：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `MODEL_ID` | `tencent/Hy-MT2-7B` | 启动时加载（缺失则先导出）的 HF repo id |
| `MODEL_DIR` | 由 `MODEL_ID` 派生 | 直接指定模型目录，优先于 `MODEL_ID` |
| `MODELS_ROOT` | `/models` | 模型仓库根目录 |
| `OV_DEVICE` | `GPU` | 推理设备：`GPU` / `CPU` / `AUTO` |
| `OV_CACHE` | 空（compose 里是 `/models/.ovcache`） | GPU kernel 编译缓存目录，二次启动秒开 |
| `OV_THREADS` | `4` | CPU 推理线程数（只在 `OV_DEVICE=CPU` 时生效） |
| `OV_PREFIX_CACHING` | `0` | `1` 开前缀缓存（continuous-batching 后端），agent 长对话 TTFT 明显降；KV cache 常驻内存，紧张的机器可能装不下 |
| `GEN_WAIT_SECONDS` | `300` | 生成锁排队上限，超时回 503 + `Retry-After` |
| `IMAGE_MAX_BYTES` | `20971520` | VLM 图片输入的单张大小上限（字节） |
| `IMAGE_FETCH_TIMEOUT` | `15` | 图片 URL 下载超时（秒） |
| `ADMIN_TOKEN` | 空（不鉴权） | 设置后 `/admin/*` 需要 `Authorization: Bearer <token>` |
| `MODEL_ALLOWLIST` | 空 | 限制能切换/自动导出的模型，逗号分隔 repo id，`*` 放开；不设时本地模型随便切、自动导出只认内置 N305 清单 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 直接 `python server.py` 时的监听地址（compose 里用 `API_PORT` 改宿主机端口） |

导出（entrypoint 自动导出和 `/admin/pull`、手动 `export_int4.sh` 都认）：

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `AUTO_EXPORT` | `1` | 容器启动发现模型缺失时自动导出；设 `0` 则直接拒绝启动 |
| `WEIGHT_FORMAT` | `int4` | 量化格式：`int4` / `int8` |
| `GROUP_SIZE` | `128` | INT4 量化分组大小 |
| `RATIO` | `1.0` | INT4 层占比，`0.8` 表示 20% 层留 INT8 |
| `EXPORT_TASK` | `auto` | optimum-cli 的导出 task，auto 按模型架构自动推断（decoder-only 出 text-generation-with-past，VLM 出 image-text-to-text） |
| `EXPORT_TIMEOUT` | `21600` | `/admin/pull` 后台导出的超时秒数 |
| `HF_HOME` | `$MODELS_ROOT/.hf` | HF 原始权重缓存目录 |
| `` | 空 | 国内下载慢设 `https://hf-mirror.com` |

compose 专用：`RENDER_GID`（宿主机 `stat -c %g /dev/dri/renderD128`，见快速开始）、
`API_PORT`（宿主机端口，默认 8000）。

## 量化配置

`export_int4.sh` 用 data-free 的 INT4 对称量化，`group_size=128`、`ratio=1.0`：

```
--weight-format int4 --group-size 128 --ratio 1.0 --sym
```

输出质量掉得厉害的话，按这个顺序放宽：`--ratio 0.8`（20% 层留 INT8）→
`--group-size 64` → `--weight-format int8`。想再压一点质量损失可以加
`--awq --dataset wikitext2 --scale-estimation`，但在 N305 上校准要跑很久。

## CI

`.github/workflows/build.yml` 在 push 到 `main` 或打 `v*` tag 时构建镜像
并推到 GHCR，然后跑 `ci/smoke.sh`：

- Intel 驱动库装齐，且 `/etc/OpenCL/vendors/*.icd` 指向的 `libigdrcl.so` 真实存在
  （ICD 指向不存在的库时 `clinfo` 只会静默认不到设备）
- OpenVINO / openvino-genai 能 import，`CPU` 设备可用
- 导出工具链（torch / nncf / optimum-cli）在镜像里可用
- `ci/test_concurrency.py`：16 个并发请求交替指定两个模型，校验每条响应上报的
  model 和实际生成用的模型一致（去掉锁会立刻失败，反向验证过）
- `ci/test_api.py`：造两个假模型目录 + stub 掉 `LLMPipeline`，起真 uvicorn 打真 HTTP，
  验证路由、OpenAPI schema、注册表扫描（跳过 `.tmp` 半成品和没 xml 的目录）、
  运行时切换三种写法、未知模型 404、SSE 分片能拼回完整文本、空 `messages` 报 400、
  Cline 风格入参（content 分段数组、`tool` 角色）能正常收、tools 协议
  （非流式/流式解析 `<tool_call>`、标签跨分片缓冲、多轮工具轨迹渲染、未调用时正常 stop）、
  `max_completion_tokens` 优先级、`stop` 两种写法、`stream_options.include_usage`、
  请求未导出模型自动触发后台导出（503 + Retry-After）、`MODEL_ALLOWLIST` 拦截切换

还有一步验证 `AUTO_EXPORT=0` 时模型缺失会拒绝启动并说清原因——
免得哪天改坏了变成在没人预期的时候闷头下 15 GB。

runner 上没有 iGPU，所以 **GPU 路径和实际 tok/s CI 验不了**，
要在 N305 上跑 `clinfo -l` 并实际打一发 `/v1/chat/completions` 确认。
