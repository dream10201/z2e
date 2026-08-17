#!/usr/bin/env python
"""OpenVINO GenAI 的 OpenAI 兼容 HTTP 服务。

任何 decoder-only 因果语言模型都能跑（LLMPipeline 的适用范围）。
请求里的 model 字段可以触发运行时切换——一次只驻留一个模型。
不做任何 prompt 包装，消息原样过 chat template，怎么用模型由客户端决定。

  POST /v1/chat/completions   OpenAI 兼容，支持 stream=true（SSE）
  GET  /v1/models             列出 /models 下已导出的模型
  POST /admin/load            显式预热某个模型
  POST /admin/pull            后台导出一个 HF 模型（需要 :export 镜像）
  GET  /admin/pull            查导出进度
  GET  /health                当前加载的模型、设备
  GET  /docs, /openapi.json   自动生成的 OpenAPI 文档

环境变量: MODEL_ID / MODELS_ROOT / OV_DEVICE / OV_CACHE / OV_THREADS / WEIGHT_FORMAT
          OV_PREFIX_CACHING=1 开前缀缓存 / GEN_WAIT_SECONDS 排队上限
          ADMIN_TOKEN 设了之后 /admin/* 要带 Authorization: Bearer <token>
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
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

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
    title="LLM on OpenVINO",
    version="2.1.0",
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
    max_tokens: int | None = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="0 走贪心")
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False
    repetition_penalty: float = 1.05


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


def _generate(prompt: str, cfg) -> tuple[str, int, int]:
    """调用方必须已持有 _mgr.gen_lock。返回 (文本, completion_tokens, prompt_tokens)。"""
    res = _mgr.pipe().generate(prompt, cfg)
    n_out, n_in = _token_counts(res)
    return str(res), n_out, n_in


def _generate_stream(prompt: str, cfg, stats: dict[str, int]) -> Iterator[str]:
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
            res = _mgr.pipe().generate(prompt, cfg, cb)
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
        # 跑着的时候读日志尾巴，结束了读最终结果
        "message": job.message or job.tail(),
        "log": str(job.log_path),
    }


@app.post("/v1/chat/completions", tags=["openai"])
def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(400, "messages 不能为空")

    max_tokens = req.max_tokens or 2048
    cfg = _gen_config(max_tokens, req.temperature, req.top_p, req.repetition_penalty)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if not req.stream:
        _acquire_gen_lock()
        try:
            entry = _need(req.model)
            prompt = _apply_template([m.model_dump() for m in req.messages])
            out, n_out, n_in = _generate(prompt, cfg)
        finally:
            _mgr.gen_lock.release()
        return {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": entry.name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": out},
                "finish_reason": "length" if n_out >= max_tokens else "stop",
            }],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        }

    # 流式：锁在这里拿，一直持有到 SSE 生成器结束才还（可能在另一个线程里还，
    # 所以 gen_lock 是普通 Lock）
    _acquire_gen_lock()
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

        stats: dict[str, int] = {}
        gen = _generate_stream(prompt, cfg, stats)
        try:
            yield chunk({"role": "assistant", "content": ""}, None)
            try:
                for piece in gen:
                    yield chunk({"content": piece}, None)
            except Exception as e:
                yield f"data: {json.dumps({'error': {'message': str(e)}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return
            finish = "length" if stats.get("tokens", 0) >= max_tokens else "stop"
            yield chunk({}, finish)
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
