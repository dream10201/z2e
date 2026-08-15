# LLM on OpenVINO GenAI (Intel N305 iGPU, INT4)

在 Intel i3-N305（8 核 Gracemont + UHD 32EU Xe-LP）上用 OpenVINO GenAI 跑
decoder-only 因果语言模型，权重 INT4，OpenAI 兼容 API，可切 CPU / iGPU 对比。
默认模型是腾讯 Hunyuan-MT-7B（WMT25 翻译赛道多语向第一），换 `MODEL_ID` 即可换模型。

**支持范围**：`openvino_genai.LLMPipeline` 覆盖的 decoder-only 因果语言模型——
Qwen / Llama / Mistral / Hunyuan / Seed-X 等。**不支持 seq2seq 翻译模型**
（NLLB、M2M100、Opus-MT、T5），GenAI 没有对应的 text2text pipeline，
这类模型在导出阶段就会失败。

镜像由 GitHub Actions 构建并推到 `ghcr.io/dream10201/z2e:latest`，
压缩后约 **0.6 GB**，包含 Intel iGPU 驱动、OpenVINO GenAI 运行时和模型导出工具链。

> 早前拆过 `:runtime` / `:export` 两个 tag，实测导出依赖只多约 340 MB（压缩），
> 不值得为此多一套概念，已经合并。那两个 tag 仍作为别名指向同一个镜像。

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

模型只导一次，之后 `up` 会秒过。国内下载慢就加 `HF_ENDPOINT=https://hf-mirror.com` 到 `.env`。

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

不想 shell 进容器也能加新模型：

```bash
curl localhost:8000/admin/pull -H 'Content-Type: application/json' \
     -d '{"model": "Qwen/Qwen3-8B"}'
curl localhost:8000/admin/pull                     # 轮询进度，带日志尾巴
```

量化参数也是环境变量：`WEIGHT_FORMAT`（int4/int8）、`GROUP_SIZE`、`RATIO`。

命令行翻译和跑分：

```bash
docker compose run --rm cli                              # 交互式，默认译成中文
echo "Edge inference cuts latency." | \
  docker compose run --rm -T cli python translate.py --device GPU --to 中文
docker compose run --rm cli python translate.py --device GPU --to English -f in.txt -o out.txt
docker compose run --rm cli python bench.py --devices CPU GPU
```

（`cli` 在 `cli` profile 下，`docker compose up` 不会连带起它。）

## HTTP API

```bash
docker compose up -d api      # 默认 8000 端口，改 API_PORT
curl localhost:8000/health
```

OpenAPI 文档在 http://localhost:8000/docs ，schema 在 `/openapi.json`。

**OpenAI 兼容**，现成的客户端直接指过来就行：

```bash
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Hunyuan-MT-7B-int4-ov",
  "messages": [{"role": "user", "content": "把下面的文本翻译成中文，不要额外解释。\n\nEdge inference cuts latency."}],
  "stream": false
}'
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = c.chat.completions.create(
    model="Hunyuan-MT-7B-int4-ov",
    messages=[{"role": "user", "content": "把下面的文本翻译成中文，不要额外解释。\n\nEdge inference cuts latency."}],
    stream=True,
)
for chunk in r:
    print(chunk.choices[0].delta.content or "", end="")
```

`stream=true` 走标准 SSE。不想自己拼 prompt 的话用更直白的 `/translate`，
它会套上模型卡给的翻译模板并回报吞吐：

```bash
curl localhost:8000/translate -H 'Content-Type: application/json' \
  -d '{"text": "Edge inference cuts latency.", "to": "中文"}'
# {"translation":"边缘推理降低了时延。","tokens":9,"seconds":3.1,"tokens_per_second":2.9,"device":"GPU"}
```

**并发是串行的**：`LLMPipeline` 不是线程安全的，而且 N305 上并行解码只会互相拖慢，
所以服务内部加了一把全局锁，请求排队处理。锁的范围是**从解析模型一直到生成结束**——
否则并发请求指定不同 `model` 时，会出现用 B 模型生成却按 A 模型上报的情况
（`ci/test_concurrency.py` 覆盖了这个）。要吞吐就上批处理，别靠并发。

**翻译模板按模型选**：翻译专用模型自带指定的 prompt 格式，用错了质量会掉。
`/translate` 按模型名匹配 preset（Hunyuan-MT、Seed-X），认不出就退化成通用指令。
可以用 `TRANSLATE_TEMPLATE` / `TRANSLATE_TEMPLATE_ZH` 覆盖。
`/v1/chat/completions` 不受影响——它走模型自带的 chat template。

**没有鉴权**，只监听容器内 0.0.0.0。别直接暴露到公网，要么绑
`127.0.0.1:8000:8000`，要么前面挡一层反代。

不用 compose 也是一条命令，entrypoint 会先补上模型再启动服务：

```bash
docker run -d --name z2e --device /dev/dri \
  -p 8000:8000 -v /your/path/models:/models \
  -e MODEL_ID=Qwen/Qwen3-8B \
  ghcr.io/dream10201/z2e
```

想跑别的命令就附在后面，模型检查照样先做：

```bash
docker run --rm -it --device /dev/dri -v /your/path/models:/models \
  ghcr.io/dream10201/z2e python translate.py --device GPU
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

**CPU 侧的线程数**：N305 是 8 个物理核、无超线程。`translate.py` 默认
`INFERENCE_NUM_THREADS=4`（用 `OV_THREADS` 覆盖），过度并行在这颗 U 上反而掉速。

**GPU 侧额外开了两项**：`DYNAMIC_QUANTIZATION_GROUP_SIZE=32`（激活动态量化）和
`KV_CACHE_PRECISION=u8`（KV cache 压到 u8，省带宽）。32EU 的瓶颈基本在内存带宽，
这两项影响明显。

## 量化配置

`export_int4.sh` 用 data-free 的 INT4 对称量化，`group_size=128`、`ratio=1.0`：

```
--weight-format int4 --group-size 128 --ratio 1.0 --sym
```

译文质量掉得厉害的话，按这个顺序放宽：`--ratio 0.8`（20% 层留 INT8）→
`--group-size 64` → `--weight-format int8`。想再压一点质量损失可以加
`--awq --dataset wikitext2 --scale-estimation`，但在 N305 上校准要跑很久。

## CI

`.github/workflows/build.yml` 在 push 到 `main` 或打 `v*` tag 时构建两个 target
并推到 GHCR，然后跑 `ci/smoke.sh`：

- Intel 驱动库装齐，且 `/etc/OpenCL/vendors/*.icd` 指向的 `libigdrcl.so` 真实存在
  （ICD 指向不存在的库时 `clinfo` 只会静默认不到设备）
- OpenVINO / openvino-genai 能 import，`CPU` 设备可用
- 导出工具链（torch / nncf / optimum-cli）在镜像里可用
- `ci/test_concurrency.py`：16 个并发请求交替指定两个模型，校验每条响应上报的
  model 和实际生成用的模型一致（去掉锁会立刻失败，反向验证过）
- `ci/test_api.py`：造两个假模型目录 + stub 掉 `LLMPipeline`，起真 uvicorn 打真 HTTP，
  验证路由、OpenAPI schema、注册表扫描（跳过 `.tmp` 半成品和没 xml 的目录）、
  运行时切换三种写法、未知模型 404、SSE 分片能拼回完整文本、空 `messages` 报 400

还有一步验证 `AUTO_EXPORT=0` 时模型缺失会拒绝启动并说清原因——
免得哪天改坏了变成在没人预期的时候闷头下 15 GB。

runner 上没有 iGPU，所以 **GPU 路径和实际 tok/s CI 验不了**，
要在 N305 上跑 `clinfo -l` 和 `bench.py` 确认。
