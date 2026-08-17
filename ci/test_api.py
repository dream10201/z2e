#!/usr/bin/env python
"""不需要真模型的 API 冒烟测试：造两个假模型目录 + stub 掉 LLMPipeline，起真 uvicorn 打真 HTTP。

验证路由、OpenAPI schema、模型注册表扫描、运行时切换、SSE 分片格式这些
容易写错又不依赖模型权重的部分。真实生成质量和 tok/s 要在有 GPU 的真机上验。
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), os.getcwd()]

import openvino_genai as ov_genai  # noqa: E402

FAKE_OUT = "边缘推理降低了时延。"

# 造两个假模型，验证注册表和运行时切换
ROOT = Path(tempfile.mkdtemp(prefix="z2e-models-"))
for name, src in [("Hunyuan-MT-7B-int4-ov", "tencent/Hunyuan-MT-7B"),
                  ("Qwen3-8B-int4-ov", "Qwen/Qwen3-8B")]:
    d = ROOT / name
    d.mkdir()
    (d / "openvino_model.xml").write_text("<net/>")
    (d / "openvino_model.bin").write_bytes(b"\0" * 16)
    (d / ".z2e.json").write_text(json.dumps({"model_id": src, "weight_format": "int4"}))
(ROOT / "half-baked.tmp").mkdir()          # 半成品目录，不该被列出来
(ROOT / "not-a-model").mkdir()             # 没有 xml，不该被列出来

os.environ["MODELS_ROOT"] = str(ROOT)
os.environ["OV_DEVICE"] = "CPU"
os.environ.setdefault("MODEL_ID", "tencent/Hunyuan-MT-7B")


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
    def __init__(self, path):
        self.path = path

    def get_tokenizer(self):
        return _Tok()

    def generate(self, prompt, cfg, cb=None):
        if cb is None:
            return _Res(FAKE_OUT)
        for ch in (FAKE_OUT[:4], FAKE_OUT[4:]):
            if cb(ch) == ov_genai.StreamingStatus.STOP:
                break
        return _Res(FAKE_OUT)


import modelmgr  # noqa: E402

modelmgr.make_pipe = lambda model, device, cache: StubPipe(model)

import server  # noqa: E402

PORT = int(os.environ.get("TEST_PORT", "8931"))
BASE = f"http://127.0.0.1:{PORT}"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.status, json.load(r)


def post(path, body, raw=False):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return (r.status, r.read().decode()) if raw else (r.status, json.load(r))


def expect_error(fn, code):
    try:
        fn()
    except urllib.error.HTTPError as e:
        assert e.code == code, f"期望 {code}，实际 {e.code}"
        return json.loads(e.read().decode())
    raise AssertionError(f"期望 HTTP {code}，但请求成功了")


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

    # 启动时按 MODEL_ID 预热
    st, body = get("/health")
    assert body["model"] == "Hunyuan-MT-7B-int4-ov", body
    assert body["available"] == ["Hunyuan-MT-7B-int4-ov", "Qwen3-8B-int4-ov"], body
    print("health ok:", body["model"], body["available"])

    # 注册表只列真模型，.tmp 和没 xml 的目录要被跳过
    st, body = get("/v1/models")
    ids = [m["id"] for m in body["data"]]
    assert ids == ["Hunyuan-MT-7B-int4-ov", "Qwen3-8B-int4-ov"], ids
    assert body["data"][1]["owned_by"] == "Qwen", body["data"][1]
    print("v1/models ok:", ids)

    st, body = get("/openapi.json")
    paths = set(body["paths"])
    assert {"/health", "/v1/models", "/v1/chat/completions",
            "/admin/load", "/admin/pull"} <= paths, paths
    assert "/translate" not in paths, paths
    print("openapi ok:", sorted(paths))

    # 运行时切换：按目录名、HF repo id、裸名字三种写法都要认
    for ref, want in [("Qwen3-8B-int4-ov", "Qwen3-8B-int4-ov"),
                      ("tencent/Hunyuan-MT-7B", "Hunyuan-MT-7B-int4-ov"),
                      ("Qwen3-8B", "Qwen3-8B-int4-ov")]:
        st, body = post("/v1/chat/completions",
                        {"model": ref, "messages": [{"role": "user", "content": "x"}]})
        assert body["model"] == want, (ref, body["model"])
        assert get("/health")[1]["model"] == want
    print("运行时切换 ok（目录名 / repo id / 裸名字都认）")

    # 认不出的模型要 404 并列出可用的
    err = expect_error(lambda: post("/v1/chat/completions",
                                    {"model": "不存在的模型",
                                     "messages": [{"role": "user", "content": "x"}]}), 404)
    assert "Hunyuan-MT-7B-int4-ov" in err["detail"], err
    print("未知模型 404 ok")

    st, body = post("/admin/load", {"model": "tencent/Hunyuan-MT-7B"})
    assert body["loaded"] == "Hunyuan-MT-7B-int4-ov", body
    print("admin/load ok")

    st, body = get("/admin/pull")
    assert body["status"] == "idle", body
    print("admin/pull 状态 ok:", body)

    st, body = post("/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "翻译"}], "max_tokens": 64})
    assert body["choices"][0]["message"]["content"] == FAKE_OUT, body
    assert body["usage"]["completion_tokens"] == 7, body
    print("chat/completions ok")

    st, text = post("/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "x"}], "stream": True}, raw=True)
    lines = [l[6:] for l in text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "[DONE]", lines[-3:]
    chunks = [json.loads(l) for l in lines[:-1]]
    joined = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert joined == FAKE_OUT, joined
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop", chunks[-1]
    print(f"SSE ok: {len(chunks)} 个 chunk, 拼回 {joined!r}")

    expect_error(lambda: post("/v1/chat/completions", {"messages": []}), 400)
    print("入参校验 ok")

    # OpenAI 多模态格式的 content 分段数组和 tool 角色都要能收
    st, body = post("/v1/chat/completions", {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "分段"},
                                     {"type": "image_url", "image_url": {"url": "data:x"}}]},
        {"role": "tool", "content": "工具结果"},
    ]})
    assert body["choices"][0]["message"]["content"] == FAKE_OUT, body
    assert body["choices"][0]["finish_reason"] == "stop", body
    print("Cline 风格入参 ok")

    print("--- API 冒烟通过 ---")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
