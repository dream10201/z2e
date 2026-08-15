#!/usr/bin/env python
"""并发下的模型一致性：两个请求指定不同 model 同时打进来，
每个响应上报的 model 必须和实际生成用的模型一致。

stub 的 generate 会返回当前加载模型的名字，所以只要有交错就能抓到。
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), os.getcwd()]

import openvino_genai as ov_genai  # noqa: E402

ROOT = Path(tempfile.mkdtemp(prefix="z2e-conc-"))
NAMES = ["A-int4-ov", "B-int4-ov"]
for n in NAMES:
    d = ROOT / n
    d.mkdir()
    (d / "openvino_model.xml").write_text("<net/>")

os.environ["MODELS_ROOT"] = str(ROOT)
os.environ["OV_DEVICE"] = "CPU"
os.environ["MODEL_ID"] = "A"


class _Tok:
    def apply_chat_template(self, messages, add_generation_prompt):
        return "x"


class _Res(str):
    @property
    def perf_metrics(self):
        class M:
            def get_num_generated_tokens(self):
                return 1
        return M()


class StubPipe:
    """generate 返回自己所属模型的名字，并故意慢一点放大交错窗口。"""

    def __init__(self, path):
        self.name = Path(path).name

    def get_tokenizer(self):
        return _Tok()

    def generate(self, prompt, cfg, cb=None):
        time.sleep(0.05)
        if cb is not None:
            for ch in self.name:
                cb(ch)
            time.sleep(0.05)
        return _Res(self.name)


import translate  # noqa: E402

translate.make_pipe = lambda model, device, cache: StubPipe(model)

import server  # noqa: E402

PORT = int(os.environ.get("TEST_PORT", "8932"))
BASE = f"http://127.0.0.1:{PORT}"


def chat(model, stream=False):
    body = {"model": model, "messages": [{"role": "user", "content": "x"}], "stream": stream}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
    if not stream:
        d = json.loads(raw)
        return d["model"], d["choices"][0]["message"]["content"]
    lines = [l[6:] for l in raw.splitlines() if l.startswith("data: ") and l[6:] != "[DONE]"]
    chunks = [json.loads(l) for l in lines]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    return chunks[0]["model"], content


def main():
    import uvicorn

    cfg = uvicorn.Config(server.app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=5)
            break
        except Exception:
            time.sleep(0.2)
    else:
        raise SystemExit("服务没起来")

    for label, stream in [("非流式", False), ("流式", True)]:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(chat, NAMES[i % 2], stream) for i in range(16)]
            results = [f.result() for f in futs]
        bad = [(rep, got) for rep, got in results if rep != got]
        assert not bad, f"{label} 上报模型与实际生成不一致: {bad[:3]}"
        print(f"{label} 16 并发请求交替切换模型: 全部一致（{len(results)} 条）")

    print("--- 并发一致性通过 ---")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(ROOT, ignore_errors=True)
