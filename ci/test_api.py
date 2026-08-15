#!/usr/bin/env python
"""不需要真模型的 API 冒烟测试：把 LLMPipeline 换成 stub，起真 uvicorn 打真 HTTP。

验证路由、OpenAPI schema、SSE 分片格式这些容易写错又不依赖模型的部分。
真实译文质量和 tok/s 要在有 GPU 的机器上跑 bench.py。
"""
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

# 本地跑时 server.py 在上一级；镜像里 ci/ 是挂进来的，server.py 在 WORKDIR
sys.path[:0] = [str(Path(__file__).resolve().parent.parent), os.getcwd()]

import openvino_genai as ov_genai  # noqa: E402

FAKE_OUT = "边缘推理降低了时延。"


class _Tok:
    def apply_chat_template(self, messages, add_generation_prompt):
        return "|".join(m["content"] for m in messages)


class _Res(str):
    @property
    def perf_metrics(self):
        class M:
            def get_num_generated_tokens(self):
                return 7
        return M()


class StubPipe:
    def get_tokenizer(self):
        return _Tok()

    def generate(self, prompt, cfg, cb=None):
        if cb is None:
            return _Res(FAKE_OUT)
        for ch in (FAKE_OUT[:4], FAKE_OUT[4:]):
            if cb(ch) == ov_genai.StreamingStatus.STOP:
                break
        return _Res(FAKE_OUT)


os.environ.setdefault("OV_DEVICE", "CPU")
import server  # noqa: E402

server.make_pipe = lambda *a, **k: StubPipe()

PORT = int(os.environ.get("TEST_PORT", "8931"))
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, json.load(r)


def post(path, body, raw=False):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        if raw:
            return r.status, r.read().decode()
        return r.status, json.load(r)


def main():
    import uvicorn

    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()

    for _ in range(100):
        try:
            if get("/health")[1]["status"] == "ok":
                break
        except Exception:
            pass
        time.sleep(0.2)
    else:
        raise SystemExit("服务没起来")

    st, body = get("/health")
    assert st == 200 and body["status"] == "ok", body
    print("health ok:", body)

    st, body = get("/v1/models")
    assert st == 200 and body["data"][0]["object"] == "model", body
    print("v1/models ok:", body["data"][0]["id"])

    st, body = get("/openapi.json")
    paths = set(body["paths"])
    assert {"/health", "/v1/models", "/v1/chat/completions", "/translate"} <= paths, paths
    print("openapi ok:", sorted(paths))

    st, body = post("/translate", {"text": "Edge inference cuts latency.", "to": "中文"})
    assert st == 200 and body["translation"] == FAKE_OUT, body
    assert body["tokens"] == 7 and body["tokens_per_second"] >= 0, body
    print("translate ok:", body)

    st, body = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "翻译成中文：Edge inference cuts latency."}],
        "max_tokens": 64,
    })
    assert st == 200 and body["object"] == "chat.completion", body
    assert body["choices"][0]["message"]["content"] == FAKE_OUT, body
    assert body["usage"]["completion_tokens"] == 7, body
    print("chat/completions ok")

    st, text = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "x"}],
        "stream": True,
    }, raw=True)
    lines = [l[6:] for l in text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]", lines[-3:]
    chunks = [json.loads(l) for l in lines[:-1]]
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert joined == FAKE_OUT, joined
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1]
    print(f"SSE ok: {len(chunks)} 个 chunk, 拼回 {joined!r}")

    # 空 messages 要报 400 而不是 500
    try:
        post("/v1/chat/completions", {"messages": []})
        raise AssertionError("空 messages 应该被拒")
    except urllib.error.HTTPError as e:
        assert e.code == 400, e.code
    print("入参校验 ok")

    print("--- API 冒烟通过 ---")


if __name__ == "__main__":
    main()
