#!/usr/bin/env python
"""Hunyuan-MT-7B 的 HTTP 服务，OpenAI 兼容 + 一个更直白的翻译端点。

  POST /v1/chat/completions   OpenAI 兼容，支持 stream=true（SSE）
  GET  /v1/models             OpenAI 兼容
  POST /translate             {"text": "...", "to": "中文"} -> {"translation": "..."}
  GET  /health                模型加载状态、设备
  GET  /docs, /openapi.json   自动生成的 OpenAPI 文档

启动: uvicorn server:app --host 0.0.0.0 --port 8000
环境变量: MODEL_DIR / OV_DEVICE / OV_CACHE / OV_THREADS / MODEL_NAME
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
from pydantic import BaseModel, Field

from translate import DEFAULT_MODEL, build_prompt, make_pipe

MODEL_DIR = os.environ.get("MODEL_DIR", DEFAULT_MODEL)
DEVICE = os.environ.get("OV_DEVICE", "GPU")
MODEL_NAME = os.environ.get("MODEL_NAME", "Hunyuan-MT-7B-int4-ov")

# LLMPipeline 不是线程安全的，而且 N305 上并发解码只会互相拖慢。
# 所以全局一把锁，请求串行处理。
_pipe: ov_genai.LLMPipeline | None = None
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipe
    _pipe = await asyncio.to_thread(make_pipe, MODEL_DIR, DEVICE, os.environ.get("OV_CACHE"))
    yield
    _pipe = None


app = FastAPI(
    title="Hunyuan-MT-7B on OpenVINO",
    version="1.0.0",
    description="Hunyuan-MT-7B 翻译服务，INT4 权重，跑在 Intel iGPU / CPU 上。",
    lifespan=lifespan,
)


# ---------- 模型定义 ----------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME, description="忽略，仅为兼容 OpenAI 客户端")
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="0 走贪心，翻译建议保持 0")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    repetition_penalty: float = 1.05


class TranslateRequest(BaseModel):
    text: str = Field(description="待翻译文本")
    to: str = Field(default="中文", description="目标语言，如 中文 / English / 日本語")
    max_tokens: int = Field(default=512, ge=1, le=8192)

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Edge inference cuts latency.", "to": "中文"}]
        }
    }


class TranslateResponse(BaseModel):
    translation: str
    tokens: int
    seconds: float
    tokens_per_second: float
    device: str


class HealthResponse(BaseModel):
    status: str
    model: str
    model_dir: str
    device: str


# ---------- 推理 ----------

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
    assert _pipe is not None
    try:
        return _pipe.get_tokenizer().apply_chat_template(messages, True)
    except Exception:
        # 模板不可用就退化成拼接
        return "\n".join(m["content"] for m in messages)


def _generate(prompt: str, cfg) -> tuple[str, int, float]:
    assert _pipe is not None
    with _lock:
        t0 = time.perf_counter()
        res = _pipe.generate(prompt, cfg)
        dt = time.perf_counter() - t0
    try:
        n = res.perf_metrics.get_num_generated_tokens()
    except Exception:
        n = 0
    return str(res), n, dt


def _generate_stream(prompt: str, cfg) -> Iterator[str]:
    """在后台线程跑 generate，把 streamer 回调的分片透过队列吐出来。"""
    assert _pipe is not None
    q: queue.Queue[str | None | BaseException] = queue.Queue()

    def cb(chunk: str):
        q.put(chunk)
        return ov_genai.StreamingStatus.RUNNING

    def work():
        try:
            with _lock:
                _pipe.generate(prompt, cfg, cb)
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
    return HealthResponse(
        status="ok" if _pipe is not None else "loading",
        model=MODEL_NAME,
        model_dir=MODEL_DIR,
        device=DEVICE,
    )


@app.get("/v1/models", tags=["openai"])
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "created": 0, "owned_by": "tencent"}],
    }


@app.post("/translate", response_model=TranslateResponse, tags=["translate"])
def translate_ep(req: TranslateRequest):
    if _pipe is None:
        raise HTTPException(503, "模型还在加载")
    prompt = _apply_template([{"role": "user", "content": build_prompt(req.text, req.to)}])
    cfg = _gen_config(req.max_tokens, 0.0, 1.0, 1.05)
    out, n, dt = _generate(prompt, cfg)
    return TranslateResponse(
        translation=out.strip(),
        tokens=n,
        seconds=round(dt, 3),
        tokens_per_second=round(n / dt, 2) if dt > 0 else 0.0,
        device=DEVICE,
    )


@app.post("/v1/chat/completions", tags=["openai"])
def chat_completions(req: ChatCompletionRequest):
    if _pipe is None:
        raise HTTPException(503, "模型还在加载")
    if not req.messages:
        raise HTTPException(400, "messages 不能为空")

    prompt = _apply_template([m.model_dump() for m in req.messages])
    cfg = _gen_config(req.max_tokens or 512, req.temperature, req.top_p, req.repetition_penalty)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        out, n, _ = _generate(prompt, cfg)
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": req.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": n, "total_tokens": n},
        }

    def sse() -> Iterator[str]:
        def chunk(delta: dict[str, Any], finish: str | None) -> str:
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

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

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
