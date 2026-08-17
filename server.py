#!/usr/bin/env python
"""OpenVINO GenAI 的 HTTP 服务：OpenAI 兼容 + 一个更直白的翻译端点。

任何 decoder-only 因果语言模型都能跑（LLMPipeline 的适用范围）。
请求里的 model 字段可以触发运行时切换——一次只驻留一个模型。

  POST /v1/chat/completions   OpenAI 兼容，支持 stream=true（SSE）
  GET  /v1/models             列出 /models 下已导出的模型
  POST /translate             {"text": "...", "to": "中文"} -> {"translation": "..."}
  POST /admin/load            显式预热某个模型
  POST /admin/pull            后台导出一个 HF 模型（需要 :export 镜像）
  GET  /admin/pull            查导出进度
  GET  /health                当前加载的模型、设备
  GET  /docs, /openapi.json   自动生成的 OpenAPI 文档

环境变量: MODEL_ID / MODELS_ROOT / OV_DEVICE / OV_CACHE / OV_THREADS / WEIGHT_FORMAT
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal

import openvino_genai as ov_genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

import modelmgr
from translate import DEFAULT_MODEL, build_prompt

DEVICE = os.environ.get("OV_DEVICE", "GPU")

_mgr = modelmgr.PipelineManager(DEVICE, os.environ.get("OV_CACHE"))
_exporter = modelmgr.Exporter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时按 MODEL_ID 预热；没导出就先不加载，等请求指定或调 /admin/pull
    entry = modelmgr.resolve(DEFAULT_MODEL)
    if entry is None:
        reg = modelmgr.scan()
        entry = next(iter(reg.values()), None)
    if entry is not None:
        await asyncio.to_thread(_mgr.load, entry)
    yield
    _mgr.unload()


app = FastAPI(
    title="LLM on OpenVINO",
    version="2.0.0",
    description="decoder-only 因果语言模型服务，INT4 权重，跑在 Intel iGPU / CPU 上。",
    lifespan=lifespan,
)


# ---------- 模型定义 ----------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: Any) -> Any:
        # tool 结果消息按 user 输入处理，多数 chat template 不认识 tool 角色
        return "user" if v == "tool" else v

    @field_validator("content", mode="before")
    @classmethod
    def flatten_content(cls, v: Any) -> Any:
        # OpenAI 多模态格式：content 可以是分段数组，取出其中的文本段拼接
        if isinstance(v, list):
            return "\n".join(
                p.get("text", "") for p in v
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if v is None:
            return ""
        return v


class ChatCompletionRequest(BaseModel):
    model: str | None = Field(default=None, description="留空用当前模型；给了就切过去")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="0 走贪心")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    repetition_penalty: float = 1.05


class TranslateRequest(BaseModel):
    text: str = Field(description="待翻译文本")
    to: str = Field(default="中文", description="目标语言，如 中文 / English / 日本語")
    model: str | None = Field(default=None, description="留空用当前模型")
    max_tokens: int = Field(default=512, ge=1, le=8192)

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Edge inference cuts latency.", "to": "中文"}]
        }
    }


class TranslateResponse(BaseModel):
    translation: str
    model: str
    tokens: int
    seconds: float
    tokens_per_second: float
    device: str


class HealthResponse(BaseModel):
    status: str
    model: str | None
    device: str
    models_root: str
    available: list[str]
    load_seconds: float


class LoadRequest(BaseModel):
    model: str = Field(description="目录名 / HF repo id")


class PullRequest(BaseModel):
    model: str = Field(description="HF repo id，例如 Qwen/Qwen3-8B")


# ---------- 内部工具 ----------

def _need(model_ref: str | None) -> modelmgr.ModelEntry:
    """解析并（必要时）切换到目标模型，返回当前条目。"""
    if model_ref:
        entry = modelmgr.resolve(model_ref)
        if entry is None:
            avail = sorted(modelmgr.scan())
            raise HTTPException(
                404,
                f"模型 {model_ref!r} 没导出。已有: {avail or '无'}。"
                f"可以调 POST /admin/pull 后台导出（需要 :export 镜像）",
            )
        _mgr.load(entry)          # 已经是当前模型时是空操作
        return entry
    if _mgr.current is None:
        raise HTTPException(503, "还没有加载任何模型，先调 POST /admin/load 或在请求里给 model")
    return _mgr.current


def _gen_config(max_tokens: int, temperature: float, top_p: float, rp: float):
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_tokens
    cfg.repetition_penalty = rp
    if temperature > 0:
        cfg.do_sample = True
        cfg.temperature = temperature
        cfg.top_p = top_p
    else:
        cfg.do_sample = False
    return cfg


def _apply_template(messages: list[dict[str, str]]) -> str:
    try:
        return _mgr.pipe().get_tokenizer().apply_chat_template(messages, True)
    except Exception:
        # 模型没带 chat template 就退化成拼接
        return "\n".join(m["content"] for m in messages)


def _generate(prompt: str, cfg) -> tuple[str, int, float]:
    """调用方必须已持有 _mgr.gen_lock。"""
    t0 = time.perf_counter()
    res = _mgr.pipe().generate(prompt, cfg)
    dt = time.perf_counter() - t0
    try:
        n = res.perf_metrics.get_num_generated_tokens()
    except Exception:
        n = 0
    return str(res), n, dt


def _generate_stream(prompt: str, cfg) -> Iterator[str]:
    """在后台线程跑 generate，把 streamer 回调的分片透过队列吐出来。

    调用方必须已持有 _mgr.gen_lock（工作线程不再自己去拿，否则跨线程会死锁）。
    """
    q: queue.Queue[str | None | BaseException] = queue.Queue()

    def cb(chunk: str):
        q.put(chunk)
        return ov_genai.StreamingStatus.RUNNING

    def work():
        try:
            _mgr.pipe().generate(prompt, cfg, cb)
        except BaseException as e:  # 把异常带回主线程，别让请求悬住
            q.put(e)
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


# ---------- 端点 ----------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    cur = _mgr.current
    return HealthResponse(
        status="ok" if cur is not None else "no-model",
        model=cur.name if cur else None,
        device=DEVICE,
        models_root=str(modelmgr.MODELS_ROOT),
        available=sorted(modelmgr.scan()),
        load_seconds=round(_mgr.load_seconds, 2),
    )


@app.get("/v1/models", tags=["openai"])
def list_models():
    return {"object": "list", "data": [e.as_openai() for e in modelmgr.scan().values()]}


@app.post("/admin/load", tags=["admin"])
def load_model(req: LoadRequest):
    with _mgr.gen_lock:
        entry = _need(req.model)
    return {"loaded": entry.name, "device": DEVICE, "load_seconds": round(_mgr.load_seconds, 2)}


@app.post("/admin/pull", tags=["admin"])
def pull_model(req: PullRequest):
    if not modelmgr.Exporter.available():
        raise HTTPException(
            501, "当前镜像没有导出依赖（torch/optimum/nncf），用 ghcr.io/dream10201/z2e:export"
        )
    if modelmgr.resolve(req.model) is not None:
        return {"status": "done", "message": "已经导出过了"}
    try:
        job = _exporter.start(req.model)
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"status": job.status, "model": job.model_id, "target": job.target,
            "hint": "7B 大约 30-60 分钟，GET /admin/pull 查进度"}


@app.get("/admin/pull", tags=["admin"])
def pull_status():
    job = _exporter.job
    if job is None:
        return {"status": "idle", "export_available": modelmgr.Exporter.available()}
    return {
        "status": job.status,
        "model": job.model_id,
        "target": job.target,
        "elapsed_seconds": round((job.finished or time.time()) - job.started, 1),
        # 跑着的时候读日志尾巴，结束了读最终结果
        "message": job.message or job.tail(),
        "log": str(job.log_path),
    }


@app.post("/translate", response_model=TranslateResponse, tags=["translate"])
def translate_ep(req: TranslateRequest):
    # 从解析模型到生成结束整段持锁，避免中途被别的请求换掉模型
    with _mgr.gen_lock:
        entry = _need(req.model)
        hint = entry.source or entry.name
        prompt = _apply_template(
            [{"role": "user", "content": build_prompt(req.text, req.to, hint)}])
        out, n, dt = _generate(prompt, _gen_config(req.max_tokens, 0.0, 1.0, 1.05))
    return TranslateResponse(
        translation=out.strip(),
        model=entry.name,
        tokens=n,
        seconds=round(dt, 3),
        tokens_per_second=round(n / dt, 2) if dt > 0 else 0.0,
        device=DEVICE,
    )


@app.post("/v1/chat/completions", tags=["openai"])
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(400, "messages 不能为空")

    cfg = _gen_config(req.max_tokens or 512, req.temperature, req.top_p, req.repetition_penalty)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        with _mgr.gen_lock:
            entry = _need(req.model)
            prompt = _apply_template([m.model_dump() for m in req.messages])
            out, n, _ = _generate(prompt, cfg)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": entry.name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
        }

    # 流式：锁在这里拿，一直持有到 SSE 生成器结束才还（可能在另一个线程里还，
    # 所以 gen_lock 是普通 Lock）
    _mgr.gen_lock.acquire()
    try:
        entry = _need(req.model)
        prompt = _apply_template([m.model_dump() for m in req.messages])
    except BaseException:
        _mgr.gen_lock.release()
        raise

    def sse() -> Iterator[str]:
        def chunk(delta: dict[str, Any], finish: str | None) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": entry.name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            yield chunk({"role": "assistant", "content": ""}, None)
            try:
                for piece in _generate_stream(prompt, cfg):
                    yield chunk({"content": piece}, None)
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            yield chunk({}, "stop")
            yield "data: [DONE]\n\n"
        finally:
            # 客户端提前断开时 Starlette 会关生成器，这里同样会走到
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
