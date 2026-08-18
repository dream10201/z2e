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
for name, src in [("Hy-MT2-7B-int4-ov", "tencent/Hy-MT2-7B"),
                  ("Qwen3-8B-int4-ov", "Qwen/Qwen3-8B")]:
    d = ROOT / name
    d.mkdir()
    (d / "openvino_model.xml").write_text("<net/>")
    (d / "openvino_model.bin").write_bytes(b"\0" * 16)
    (d / ".z2e.json").write_text(json.dumps({"model_id": src, "weight_format": "int4"}))
# 假 VLM：多模态导出产物没有 openvino_model.xml，语言塔叫 openvino_language_model.xml
_vlm = ROOT / "FakeVLM-int4-ov"
_vlm.mkdir()
(_vlm / "openvino_language_model.xml").write_text("<net/>")
(_vlm / ".z2e.json").write_text(json.dumps({"model_id": "fake/FakeVLM", "weight_format": "int4"}))
(ROOT / "half-baked.tmp").mkdir()          # 半成品目录，不该被列出来
(ROOT / "not-a-model").mkdir()             # 没有 xml，不该被列出来

os.environ["MODELS_ROOT"] = str(ROOT)
os.environ["OV_DEVICE"] = "CPU"
os.environ.setdefault("MODEL_ID", "tencent/Hy-MT2-7B")


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


# 模拟 Qwen/Hermes 风格的工具调用输出；流式分片故意切在标签中间，
# 验证服务端不会把半个 <tool_call> 标签当正文吐出去
TOOL_TEXT = ('好的<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}'
             '\n</tool_call>')
TOOL_CHUNKS = ('好的<tool_', 'call>{"name": "get_weather", "arg',
               'uments": {"city": "北京"}}</tool', '_call>')


class StubPipe:
    def __init__(self, path):
        self.path = path

    def get_tokenizer(self):
        return _Tok()

    def generate(self, prompt, images=None, generation_config=None, streamer=None):
        text, chunks = FAKE_OUT, (FAKE_OUT[:4], FAKE_OUT[4:])
        if "帮我查天气" in prompt:
            text, chunks = TOOL_TEXT, TOOL_CHUNKS
        if images:
            # 回显收到几张图、prompt 里注入了几个通用标签，测试据此断言
            text = f"images={len(images)} tags={prompt.count('<ov_genai_image_')}"
            chunks = (text,)
        if streamer is None:
            return _Res(text)
        for ch in chunks:
            if streamer(ch) == ov_genai.StreamingStatus.STOP:
                break
        return _Res(text)


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
    assert body["model"] == "Hy-MT2-7B-int4-ov", body
    assert body["available"] == ["FakeVLM-int4-ov", "Hy-MT2-7B-int4-ov", "Qwen3-8B-int4-ov"], body
    print("health ok:", body["model"], body["available"])

    # 注册表只列真模型，.tmp 和没 xml 的目录要被跳过
    st, body = get("/v1/models")
    ids = [m["id"] for m in body["data"]]
    assert ids == ["FakeVLM-int4-ov", "Hy-MT2-7B-int4-ov", "Qwen3-8B-int4-ov"], ids
    assert body["data"][2]["owned_by"] == "Qwen", body["data"][2]
    print("v1/models ok:", ids)

    st, body = get("/openapi.json")
    paths = set(body["paths"])
    assert {"/health", "/v1/models", "/v1/chat/completions",
            "/admin/load", "/admin/pull"} <= paths, paths
    assert "/translate" not in paths, paths
    print("openapi ok:", sorted(paths))

    # 运行时切换：按目录名、HF repo id、裸名字三种写法都要认
    for ref, want in [("Qwen3-8B-int4-ov", "Qwen3-8B-int4-ov"),
                      ("tencent/Hy-MT2-7B", "Hy-MT2-7B-int4-ov"),
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
    assert "Hy-MT2-7B-int4-ov" in err["detail"], err
    print("未知模型 404 ok")

    st, body = post("/admin/load", {"model": "tencent/Hy-MT2-7B"})
    assert body["loaded"] == "Hy-MT2-7B-int4-ov", body
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
    print("Cline 风格入参 ok（文本模型丢弃图片段）")

    # VLM 模型：图片段解码成张量、prompt 里按位置注入 <ov_genai_image_i> 标签
    png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
           "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    st, body = post("/v1/chat/completions", {"model": "FakeVLM-int4-ov", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "这是什么"},
                                     {"type": "image_url", "image_url": {"url": png}}]},
    ]})
    assert body["choices"][0]["message"]["content"] == "images=1 tags=1", body
    st, h = get("/health")
    assert h["multimodal"] is True, h
    # 图片解码失败要给 400，不能 500
    expect_error(lambda: post("/v1/chat/completions", {"model": "FakeVLM-int4-ov", "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}}]},
    ]}), 400)
    # 切回文本模型，multimodal 要跟着变回去
    st, body = post("/v1/chat/completions", {
        "model": "Hy-MT2-7B-int4-ov", "messages": [{"role": "user", "content": "你好"}]})
    assert get("/health")[1]["multimodal"] is False
    print("VLM 图片入参 ok（标签注入 / 解码失败 400 / multimodal 上报）")

    # tools：非流式，模型输出 <tool_call> 要解析成标准 tool_calls
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "查天气",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
    st, body = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "帮我查天气"}], "tools": tools})
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls", choice
    assert choice["message"]["content"] == "好的", choice
    tc = choice["message"]["tool_calls"]
    assert tc[0]["function"]["name"] == "get_weather", tc
    assert json.loads(tc[0]["function"]["arguments"]) == {"city": "北京"}, tc
    assert tc[0]["id"].startswith("call_") and tc[0]["type"] == "function", tc
    print("tools 非流式 ok:", tc[0]["function"])

    # tools：流式，分片切在标签中间也要能拼出 tool_calls delta
    st, text = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "帮我查天气"}],
        "tools": tools, "stream": True}, raw=True)
    lines = [l[6:] for l in text.splitlines() if l.startswith("data: ") and l[6:] != "[DONE]"]
    chunks = [json.loads(l) for l in lines]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "好的", content
    tc_deltas = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert len(tc_deltas) == 1, chunks
    d = tc_deltas[0]["choices"][0]["delta"]["tool_calls"][0]
    assert d["function"]["name"] == "get_weather", d
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls", chunks[-1]
    print("tools 流式 ok（标签跨分片）")

    # 给了 tools 但模型没调用 -> 正常 stop；历史里的 tool_calls / tool 轮也要能收
    st, body = post("/v1/chat/completions", {
        "messages": [
            {"role": "user", "content": "北京天气如何"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "晴 30 度"},
        ], "tools": tools})
    assert body["choices"][0]["finish_reason"] == "stop", body
    assert body["choices"][0]["message"]["content"] == FAKE_OUT, body
    print("tools 多轮轨迹 + 未调用时正常 stop ok")

    # max_completion_tokens 优先于 max_tokens：stub 生成 7 个 token，
    # 上限给到 7 应该报 length，同时给的 max_tokens=64 要被忽略
    st, body = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "x"}],
        "max_tokens": 64, "max_completion_tokens": 7})
    assert body["choices"][0]["finish_reason"] == "length", body
    print("max_completion_tokens 优先 ok")

    # stop 收字符串和数组两种写法
    for stop in ("###", ["###", "\n\n"]):
        st, body = post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "x"}], "stop": stop})
        assert body["choices"][0]["message"]["content"] == FAKE_OUT, body
    print("stop 入参 ok")

    # stream_options.include_usage：结尾多一个 choices 为空、usage 有值的 chunk
    st, text = post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "x"}], "stream": True,
        "stream_options": {"include_usage": True}}, raw=True)
    lines = [l[6:] for l in text.splitlines() if l.startswith("data: ") and l[6:] != "[DONE]"]
    chunks = [json.loads(l) for l in lines]
    assert all("usage" in c for c in chunks), chunks[0]
    last = chunks[-1]
    assert last["choices"] == [] and last["usage"]["completion_tokens"] == 7, last
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop", chunks[-2]
    print("stream_options.include_usage ok")

    # 自动导出：请求 N305 白名单里但没导出的模型 -> 触发后台导出并回 503
    import types

    class FakeExporter:
        def __init__(self):
            self.started = []

        def start(self, mid):
            self.started.append(mid)
            return types.SimpleNamespace(status="running", model_id=mid, target="/tmp/x")

    fake, real = FakeExporter(), server._exporter
    server._exporter = fake
    try:
        err = expect_error(lambda: post("/v1/chat/completions", {
            "model": "Qwen/Qwen3-4B", "messages": [{"role": "user", "content": "x"}]}), 503)
        assert "导出" in err["detail"], err
        assert fake.started == ["Qwen/Qwen3-4B"], fake.started
    finally:
        server._exporter = real
    print("请求未导出模型自动触发导出 ok")

    # MODEL_ALLOWLIST：设了就只能切列表里的
    os.environ["MODEL_ALLOWLIST"] = "Qwen/Qwen3-8B"
    try:
        expect_error(lambda: post("/v1/chat/completions", {
            "model": "Hy-MT2-7B-int4-ov",
            "messages": [{"role": "user", "content": "x"}]}), 403)
        st, body = post("/v1/chat/completions", {
            "model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "x"}]})
        assert body["model"] == "Qwen3-8B-int4-ov", body
    finally:
        del os.environ["MODEL_ALLOWLIST"]
    print("MODEL_ALLOWLIST 限制切换 ok")

    print("--- API 冒烟通过 ---")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
