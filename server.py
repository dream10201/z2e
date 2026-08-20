#!/usr/bin/env python
"""OpenVINO GenAI 的 OpenAI 兼容 HTTP 服务。

decoder-only 因果语言模型（LLMPipeline）和多模态 VLM（VLMPipeline，如
Qwen-VL / Qwen3.5 系）都能跑。VLM 模型支持 OpenAI 多模态消息格式：content
分段数组里的 image_url（data: base64 或 http/https）会解码成图片喂给模型。
请求里的 model 字段可以触发运行时切换——一次只驻留一个模型。
消息原样过 chat template，怎么用模型由客户端决定；唯一的例外是 tools——
函数签名以 Hermes/Qwen 风格注入 system prompt，输出解析回标准 tool_calls。

  POST /v1/chat/completions   OpenAI 兼容，支持 stream=true（SSE）和 tools
  GET  /v1/models             列出 /models 下已导出的模型
  POST /admin/load            显式预热某个模型
  POST /admin/pull            后台导出一个 HF 模型（需要 :export 镜像）
  GET  /admin/pull            查导出进度
  GET  /health                当前加载的模型、设备
  GET  /docs, /openapi.json   自动生成的 OpenAPI 文档

环境变量: MODEL_ID / MODELS_ROOT / OV_DEVICE / OV_CACHE / OV_THREADS / WEIGHT_FORMAT
          OV_PREFIX_CACHING=1 开前缀缓存 / GEN_WAIT_SECONDS 排队上限
          ADMIN_TOKEN 设了之后 /admin/* 要带 Authorization: Bearer <token>
          MODEL_ALLOWLIST 限制能切/能自动导出的模型（不设=本地随便切+N305 内置清单）
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import queue
import re
import threading
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal

import openvino_genai as ov_genai
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

import modelmgr

DEVICE = os.environ.get("OV_DEVICE", "GPU")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
# 所有生成共用一把锁，忙的时候请求最多排这么久，超了直接 503 让客户端重试
GEN_WAIT_SECONDS = float(os.environ.get("GEN_WAIT_SECONDS", "300"))

_mgr = modelmgr.PipelineManager(DEVICE, os.environ.get("OV_CACHE"))
_exporter = modelmgr.Exporter()

# 旧版 openvino_genai 没有 CANCEL（丢弃已生成内容），退回 STOP
_CANCEL = getattr(ov_genai.StreamingStatus, "CANCEL", ov_genai.StreamingStatus.STOP)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时按 MODEL_ID 预热；没导出就先不加载，等请求指定或调 /admin/pull
    entry = modelmgr.resolve(modelmgr.DEFAULT_MODEL)
    if entry is None:
        reg = modelmgr.scan()
        entry = next(iter(reg.values()), None)
    if entry is not None:
        await asyncio.to_thread(_mgr.load, entry)
    yield
    _mgr.unload()


app = FastAPI(
    title="z2e — Zero to Endpoint",
    version="2.1.0",
    description="decoder-only 因果语言模型 / 多模态 VLM 服务，INT4 权重，跑在 Intel iGPU / CPU 上。",
    lifespan=lifespan,
)


# ---------- 模型定义 ----------

# content 分段数组里图片段的占位符，_render_messages 会按全局顺序替换成
# openvino_genai 认识的 <ov_genai_image_i> 通用标签
_IMG_MARK = "\x00z2e_image\x00"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    images: list[str] = Field(default_factory=list)  # 本消息里图片段的 url/data URI，按出现顺序
    tool_calls: list[dict[str, Any]] | None = None   # assistant 历史里的工具调用
    tool_call_id: str | None = None                  # tool 消息对应哪次调用

    @model_validator(mode="before")
    @classmethod
    def split_content(cls, data: Any) -> Any:
        # OpenAI 多模态格式：content 可以是分段数组。文本段拼接；图片段收进
        # images，并在文本里留占位符记住位置
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            return data
        texts: list[str] = []
        images: list[str] = []
        for p in data["content"]:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                texts.append(p.get("text", ""))
            elif p.get("type") in ("image_url", "input_image"):
                url = p.get("image_url") or p.get("url") or ""
                if isinstance(url, dict):
                    url = url.get("url", "")
                if url:
                    texts.append(_IMG_MARK)
                    images.append(url)
        return {**data, "content": "\n".join(texts), "images": images}

    @field_validator("content", mode="before")
    @classmethod
    def none_to_empty(cls, v: Any) -> Any:
        return "" if v is None else v


class ToolFunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: ToolFunctionDef


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default=None, description="留空用当前模型；给了就切过去")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(
        default=None, ge=1, le=32768, description="新版字段，优先于 max_tokens")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="0 走贪心")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, description="不设=不限制；仅采样时生效")
    stop: str | list[str] | None = Field(default=None, description="停止序列，命中即停且不进输出")
    stream: bool = False
    stream_options: StreamOptions | None = None
    repetition_penalty: float = 1.05
    tools: list[ToolDef] | None = None
    tool_choice: str | dict[str, Any] | None = Field(
        default=None, description='"auto"（默认）/ "none" / "required" / {"function":{"name":...}}')

    @property
    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 2048


class HealthResponse(BaseModel):
    status: str
    model: str | None
    multimodal: bool = False   # 当前模型是不是 VLM（能吃图片）
    device: str
    models_root: str
    available: list[str]
    load_seconds: float


class LoadRequest(BaseModel):
    model: str = Field(description="目录名 / HF repo id")


class PullRequest(BaseModel):
    model: str = Field(description="HF repo id，例如 Qwen/Qwen3-8B")


# ---------- 内部工具 ----------

def _require_admin(authorization: str | None = Header(default=None)):
    if ADMIN_TOKEN and authorization != f"Bearer {ADMIN_TOKEN}":
        raise HTTPException(401, "需要 Authorization: Bearer $ADMIN_TOKEN")


def _acquire_gen_lock() -> None:
    """拿生成锁，排队超时给 503 而不是无限悬着。"""
    if not _mgr.gen_lock.acquire(timeout=GEN_WAIT_SECONDS):
        raise HTTPException(
            503, f"服务正忙，排队超过 {GEN_WAIT_SECONDS:.0f}s，稍后重试",
            headers={"Retry-After": "5"},
        )


def _need(model_ref: str | None) -> modelmgr.ModelEntry:
    """解析并（必要时）切换到目标模型，返回当前条目。

    没导出但在允许列表里的模型，自动触发后台导出并回 503 + Retry-After——
    导出要几十分钟，不能让这个 HTTP 请求挂着等。
    """
    if model_ref:
        entry = modelmgr.resolve(model_ref)
        if entry is None:
            _auto_pull_or_raise(model_ref)
        if not modelmgr.serve_allowed(entry):
            raise HTTPException(403, f"模型 {entry.name!r} 不在 MODEL_ALLOWLIST 里")
        _mgr.load(entry)          # 已经是当前模型时是空操作
        return entry
    if _mgr.current is None:
        raise HTTPException(503, "还没有加载任何模型，先调 POST /admin/load 或在请求里给 model")
    return _mgr.current


def _auto_pull_or_raise(model_ref: str) -> None:
    """请求了一个没导出的模型：能自动导就启动后台导出，不能就说清楚为什么。"""
    avail = sorted(modelmgr.scan())
    if not modelmgr.pull_allowed(model_ref):
        raise HTTPException(
            404,
            f"模型 {model_ref!r} 没导出，也不在允许自动导出的列表里。已有: {avail or '无'}。"
            f"允许列表用 MODEL_ALLOWLIST 配置（逗号分隔，* 放开）",
        )
    if not modelmgr.Exporter.available():
        raise HTTPException(
            404,
            f"模型 {model_ref!r} 没导出，且当前镜像没有导出依赖（torch/optimum/nncf）。"
            f"已有: {avail or '无'}",
        )
    try:
        _exporter.start(model_ref)
    except RuntimeError as e:
        raise HTTPException(503, f"导出队列忙：{e}。进度 GET /admin/pull",
                            headers={"Retry-After": "120"})
    raise HTTPException(
        503,
        f"模型 {model_ref!r} 已开始后台导出（7B 约 30-60 分钟），"
        f"进度 GET /admin/pull，导完重试本请求即可",
        headers={"Retry-After": "120"},
    )


def _gen_config(max_tokens: int, temperature: float, top_p: float, rp: float,
                stop: str | list[str] | None = None, top_k: int | None = None):
    cfg = ov_genai.GenerationConfig()
    # prompt 在 _apply_template 里已经过了 chat template，别让 pipeline 再包一层
    # （新版 GenerationConfig 默认 apply_chat_template=True，VLMPipeline 尤其会踩）
    if hasattr(cfg, "apply_chat_template"):
        cfg.apply_chat_template = False
    cfg.max_new_tokens = max_tokens
    cfg.repetition_penalty = rp
    if stop:
        cfg.stop_strings = {stop} if isinstance(stop, str) else set(stop)
        cfg.include_stop_str_in_output = False
    if temperature > 0:
        cfg.do_sample = True
        cfg.temperature = temperature
        cfg.top_p = top_p
        if top_k is not None:
            cfg.top_k = top_k
    else:
        cfg.do_sample = False
    return cfg


# ---------- tools 协议 ----------
#
# ov_genai 的 apply_chat_template 不收 tools 参数，所以走文本层：函数签名以
# Hermes/Qwen 风格注入 system prompt，模型输出里的 <tool_call>{...}</tool_call>
# 解析回标准 tool_calls。这要求模型本身按这个格式训练过（Qwen / Hermes 系都是）；
# 没练过工具调用的模型（比如纯翻译模型）给了 tools 也不会用。

_TC_OPEN, _TC_CLOSE = "<tool_call>", "</tool_call>"
_TC_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _tools_enabled(req: ChatCompletionRequest) -> bool:
    return bool(req.tools) and req.tool_choice != "none"


def _tools_system_text(req: ChatCompletionRequest) -> str:
    sigs = "\n".join(
        json.dumps(t.model_dump(exclude_none=True), ensure_ascii=False) for t in req.tools or [])
    txt = (
        "# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        f"<tools>\n{sigs}\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        '<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'
    )
    if req.tool_choice == "required":
        txt += "\n\nYou must call at least one function before responding."
    elif isinstance(req.tool_choice, dict):
        name = (req.tool_choice.get("function") or {}).get("name")
        if name:
            txt += f"\n\nYou must call the function {name!r}."
    return txt


def _make_call(raw: str) -> dict[str, Any] | None:
    try:
        d = json.loads(raw)
        name = d["name"]
    except Exception:
        return None
    args = d.get("arguments", {})
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return {"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
            "function": {"name": name, "arguments": args}}


def _parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """把正文里的 <tool_call> 块摘出来。解析不了的块原样留在正文里。"""
    calls: list[dict[str, Any]] = []

    def repl(m: re.Match) -> str:
        c = _make_call(m.group(1))
        if c is None:
            return m.group(0)
        calls.append(c)
        return ""

    return _TC_RE.sub(repl, text).strip(), calls


def _render_messages(req: ChatCompletionRequest) -> tuple[list[dict[str, str]], list[str]]:
    """OpenAI 消息 -> (chat template 能吃的纯文本消息, 全部图片 url 按出现顺序)。

    当前模型是 VLM 时，图片占位符替换成 <ov_genai_image_i> 通用标签（i 是整个
    请求里的全局序号，和返回的 url 列表一一对应），VLMPipeline 生成时会把标签
    换成对应图片的嵌入。纯文本模型则把图片段丢弃（Cline 这类客户端会给文本模型
    发截图，硬拒会把它们打断）。

    assistant 的 tool_calls 渲染回 <tool_call> 块，tool 结果包成 <tool_response>
    塞进 user 轮（多数 chat template 不认识 tool 角色），和注入 prompt 的格式对齐，
    模型看到的多轮工具轨迹才是自洽的。
    """
    vlm = _mgr.is_vlm
    msgs: list[dict[str, str]] = []
    image_urls: list[str] = []
    for m in req.messages:
        content = m.content
        if vlm:
            for url in m.images:
                content = content.replace(_IMG_MARK, f"<ov_genai_image_{len(image_urls)}>", 1)
                image_urls.append(url)
        else:
            content = content.replace(_IMG_MARK, "")
        m = m.model_copy(update={"content": content})
        if m.role == "assistant" and m.tool_calls:
            blocks = []
            for c in m.tool_calls:
                fn = c.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        pass
                blocks.append(
                    "<tool_call>\n"
                    + json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False)
                    + "\n</tool_call>")
            content = "\n".join(([m.content] if m.content else []) + blocks)
            msgs.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            msgs.append({"role": "user",
                         "content": f"<tool_response>\n{m.content}\n</tool_response>"})
        else:
            msgs.append({"role": m.role, "content": m.content})
    if _tools_enabled(req):
        sys_txt = _tools_system_text(req)
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] += "\n\n" + sys_txt
        else:
            msgs.insert(0, {"role": "system", "content": sys_txt})
    return msgs, image_urls


# ---------- 图片输入 ----------

IMAGE_MAX_BYTES = int(os.environ.get("IMAGE_MAX_BYTES", str(20 * 1024 * 1024)))
IMAGE_FETCH_TIMEOUT = float(os.environ.get("IMAGE_FETCH_TIMEOUT", "15"))


def _decode_image(url: str):
    """data: base64 或 http/https 的图片 -> ov.Tensor（HWC uint8 RGB）。

    pillow/numpy 只在真用到图片时才 import：纯文本路径不背这个依赖。
    """
    if url.startswith("data:"):
        try:
            raw = base64.b64decode(url.split(",", 1)[1], validate=False)
        except Exception:
            raise HTTPException(400, "图片 data URI 不是合法的 base64")
    elif url.startswith(("http://", "https://")):
        try:
            r = urllib.request.Request(url, headers={"User-Agent": "z2e"})
            with urllib.request.urlopen(r, timeout=IMAGE_FETCH_TIMEOUT) as resp:
                raw = resp.read(IMAGE_MAX_BYTES + 1)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, f"图片下载失败 {url!r}: {e}")
    else:
        raise HTTPException(400, f"不支持的图片来源 {url[:64]!r}，只认 data: base64 和 http/https")
    if len(raw) > IMAGE_MAX_BYTES:
        raise HTTPException(400, f"图片超过 {IMAGE_MAX_BYTES} 字节上限")
    try:
        import numpy as np
        import openvino as ov
        from PIL import Image
        pic = Image.open(io.BytesIO(raw)).convert("RGB")
        return ov.Tensor(np.array(pic))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"图片解码失败: {e}")


def _load_images(urls: list[str]) -> list:
    """_render_messages 只在当前模型是 VLM 时才会收集 url，这里只管解码。"""
    return [_decode_image(u) for u in urls]


def _stream_events(pieces: Iterator[str], parse_tools: bool) -> Iterator[tuple[str, Any]]:
    """把原始分片流变成 ("content", str) 事件流，结尾一条 ("tool_calls", list)。

    开着 tools 时要缓冲可能是 <tool_call> 标签开头的尾巴，避免把半个标签
    当正文吐给客户端；标签内的内容攒完整了再解析。
    """
    if not parse_tools:
        for p in pieces:
            yield "content", p
        yield "tool_calls", []
        return

    buf, in_call = "", False
    calls: list[dict[str, Any]] = []
    for p in pieces:
        buf += p
        while True:
            if in_call:
                i = buf.find(_TC_CLOSE)
                if i < 0:
                    break
                c = _make_call(buf[:i].strip())
                if c:
                    calls.append(c)
                buf = buf[i + len(_TC_CLOSE):]
                in_call = False
            else:
                i = buf.find(_TC_OPEN)
                if i >= 0:
                    if buf[:i]:
                        yield "content", buf[:i]
                    buf = buf[i + len(_TC_OPEN):]
                    in_call = True
                    continue
                # 留下可能是标签前缀的尾巴，其余的放行
                keep = 0
                for k in range(min(len(buf), len(_TC_OPEN) - 1), 0, -1):
                    if buf.endswith(_TC_OPEN[:k]):
                        keep = k
                        break
                if keep < len(buf):
                    yield "content", buf[:len(buf) - keep]
                buf = buf[len(buf) - keep:]
                break
    if in_call:
        # 没闭合就到头了（多半是打满 max_tokens），能解析就收下
        c = _make_call(buf.strip())
        if c:
            calls.append(c)
    elif buf:
        yield "content", buf
    yield "tool_calls", calls


def _apply_template(messages: list[dict[str, str]]) -> str:
    try:
        return _mgr.pipe().get_tokenizer().apply_chat_template(messages, True)
    except Exception:
        # 模型没带 chat template 就退化成拼接
        return "\n".join(m["content"] for m in messages)


def _token_counts(res) -> tuple[int, int]:
    """(completion_tokens, prompt_tokens)，拿不到就 0。"""
    n_out = n_in = 0
    try:
        n_out = res.perf_metrics.get_num_generated_tokens()
    except Exception:
        pass
    try:
        n_in = res.perf_metrics.get_num_input_tokens()
    except Exception:
        pass
    return n_out, n_in


def _res_text(res) -> str:
    """DecodedResults / VLMDecodedResults 都有 texts；stub 之类没有的退回 str()。"""
    texts = getattr(res, "texts", None)
    return texts[0] if texts else str(res)


def _generate(prompt: str, cfg, images: list) -> tuple[str, int, int]:
    """调用方必须已持有 _mgr.gen_lock。返回 (文本, completion_tokens, prompt_tokens)。"""
    if images:
        res = _mgr.pipe().generate(prompt, images=images, generation_config=cfg)
    else:
        res = _mgr.pipe().generate(prompt, generation_config=cfg)
    n_out, n_in = _token_counts(res)
    return _res_text(res), n_out, n_in


def _generate_stream(prompt: str, cfg, stats: dict[str, int], images: list) -> Iterator[str]:
    """在后台线程跑 generate，把 streamer 回调的分片透过队列吐出来。

    调用方必须已持有 _mgr.gen_lock（工作线程不再自己去拿，否则跨线程会死锁）。
    生成器被关掉（客户端断连）时让回调返回 CANCEL 叫停 pipeline，并等工作线程
    真正退出——否则孤儿线程还占着 pipeline，放锁后下一个请求就并发进来了。
    结束后 stats 里有 tokens / prompt_tokens。
    """
    q: queue.Queue[str | None | BaseException] = queue.Queue()
    cancelled = threading.Event()

    def cb(chunk: str):
        if cancelled.is_set():
            return _CANCEL
        q.put(chunk)
        return ov_genai.StreamingStatus.RUNNING

    def work():
        try:
            if images:
                res = _mgr.pipe().generate(
                    prompt, images=images, generation_config=cfg, streamer=cb)
            else:
                res = _mgr.pipe().generate(prompt, generation_config=cfg, streamer=cb)
            stats["tokens"], stats["prompt_tokens"] = _token_counts(res)
        except BaseException as e:  # 把异常带回主线程，别让请求悬住
            q.put(e)
        finally:
            q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        cancelled.set()
        # CANCEL 在下一个 token 生效；pipeline 真挂死的话服务本来也没法继续
        t.join()


# ---------- 端点 ----------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    cur = _mgr.current
    return HealthResponse(
        status="ok" if cur is not None else "no-model",
        model=cur.name if cur else None,
        multimodal=_mgr.is_vlm,
        device=DEVICE,
        models_root=str(modelmgr.MODELS_ROOT),
        available=sorted(modelmgr.scan()),
        load_seconds=round(_mgr.load_seconds, 2),
    )


@app.get("/v1/models", tags=["openai"])
def list_models():
    return {"object": "list", "data": [e.as_openai() for e in modelmgr.scan().values()]}


@app.post("/admin/load", tags=["admin"], dependencies=[Depends(_require_admin)])
def load_model(req: LoadRequest):
    _acquire_gen_lock()
    try:
        entry = _need(req.model)
    finally:
        _mgr.gen_lock.release()
    return {"loaded": entry.name, "device": DEVICE, "load_seconds": round(_mgr.load_seconds, 2)}


@app.post("/admin/pull", tags=["admin"], dependencies=[Depends(_require_admin)])
def pull_model(req: PullRequest):
    if not modelmgr.Exporter.available():
        raise HTTPException(
            501, "当前镜像没有导出依赖（torch/optimum/nncf），用 ghcr.io/dream10201/z2e:export"
        )
    if not modelmgr.pull_allowed(req.model):
        raise HTTPException(
            403, f"模型 {req.model!r} 不在允许导出的列表里，用 MODEL_ALLOWLIST 配置")
    if modelmgr.resolve(req.model) is not None:
        return {"status": "done", "message": "已经导出过了"}
    try:
        job = _exporter.start(req.model)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"status": job.status, "model": job.model_id, "target": job.target,
            "hint": "7B 大约 30-60 分钟，GET /admin/pull 查进度"}


@app.get("/admin/pull", tags=["admin"], dependencies=[Depends(_require_admin)])
def pull_status():
    job = _exporter.job
    if job is None:
        return {"status": "idle", "export_available": modelmgr.Exporter.available()}
    return {
        "status": job.status,
        "model": job.model_id,
        "target": job.target,
        "elapsed_seconds": round((job.finished or time.time()) - job.started, 1),
        # 只报结果（失败原因）；进度是运维信息，看容器日志（docker/podman logs）
        "message": job.message,
    }


@app.post("/v1/chat/completions", tags=["openai"])
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(400, "messages 不能为空")

    max_tokens = req.effective_max_tokens
    cfg = _gen_config(max_tokens, req.temperature, req.top_p, req.repetition_penalty,
                      req.stop, req.top_k)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        _acquire_gen_lock()
        try:
            entry = _need(req.model)
            msgs, image_urls = _render_messages(req)
            prompt = _apply_template(msgs)
            images = _load_images(image_urls)
            out, n_out, n_in = _generate(prompt, cfg, images)
        finally:
            _mgr.gen_lock.release()
        tool_calls: list[dict[str, Any]] = []
        if _tools_enabled(req):
            out, tool_calls = _parse_tool_calls(out)
        message: dict[str, Any] = {"role": "assistant", "content": out}
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        else:
            finish = "length" if n_out >= max_tokens else "stop"
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": entry.name,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish,
            }],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        }

    # 流式：锁在这里拿，一直持有到 SSE 生成器结束才还（可能在另一个线程里还，
    # 所以 gen_lock 是普通 Lock）
    _acquire_gen_lock()
    try:
        entry = _need(req.model)
        msgs, image_urls = _render_messages(req)
        prompt = _apply_template(msgs)
        images = _load_images(image_urls)
    except BaseException:
        _mgr.gen_lock.release()
        raise

    include_usage = bool(req.stream_options and req.stream_options.include_usage)

    def sse() -> Iterator[str]:
        def chunk(delta: dict[str, Any], finish: str | None) -> str:
            payload: dict[str, Any] = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": entry.name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if include_usage:
                # 按 OpenAI 规范：开了 include_usage 后每个 chunk 带 usage: null，
                # 最后单发一个 choices 为空、usage 有值的 chunk
                payload["usage"] = None
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        stats: dict[str, int] = {}
        gen = _generate_stream(prompt, cfg, stats, images)
        try:
            yield chunk({"role": "assistant", "content": ""}, None)
            tool_calls: list[dict[str, Any]] = []
            try:
                for kind, data in _stream_events(gen, _tools_enabled(req)):
                    if kind == "content":
                        yield chunk({"content": data}, None)
                    else:
                        tool_calls = data
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if tool_calls:
                yield chunk({"tool_calls": [
                    {"index": i, "id": c["id"], "type": "function", "function": c["function"]}
                    for i, c in enumerate(tool_calls)
                ]}, None)
                finish = "tool_calls"
            else:
                finish = "length" if stats.get("tokens", 0) >= max_tokens else "stop"
            yield chunk({}, finish)
            if include_usage:
                n_out = stats.get("tokens", 0)
                n_in = stats.get("prompt_tokens", 0)
                payload = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": entry.name, "choices": [],
                    "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                              "total_tokens": n_in + n_out},
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # 客户端提前断开时 Starlette 会关生成器，这里同样会走到。
            # 先显式关内层生成器（叫停并等工作线程退出），再放锁
            gen.close()
            _mgr.gen_lock.release()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))
