#!/usr/bin/env python
"""Hunyuan-MT-7B 翻译，走 OpenVINO GenAI（CPU 或 iGPU）。

用法:
  python translate.py --device GPU                     # 交互式
  echo "hello world" | python translate.py --to 中文    # 管道批处理
  python translate.py --to English -f in.txt -o out.txt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import openvino_genai as ov_genai

DEFAULT_MODEL = "/models/Hunyuan-MT-7B-int4-ov"

# 模型卡给的两套 prompt：中文互译 / 非中文互译
ZH_TMPL = "把下面的文本翻译成{tgt}，不要额外解释。\n\n{text}"
XX_TMPL = "Translate the following segment into {tgt}, without additional explanation.\n\n{text}"


def build_prompt(text: str, tgt: str) -> str:
    zh_like = any(k in tgt for k in ("中文", "Chinese", "汉语"))
    tmpl = ZH_TMPL if zh_like or _has_han(text) else XX_TMPL
    return tmpl.format(tgt=tgt, text=text.strip())


def _has_han(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


def make_pipe(model: str, device: str, cache: str | None):
    cfg = {}
    if cache:
        Path(cache).mkdir(parents=True, exist_ok=True)
        cfg["CACHE_DIR"] = cache
    if device.upper().startswith("CPU"):
        # N305 是 8 个物理核、无超线程，别超订
        cfg["INFERENCE_NUM_THREADS"] = int(os.environ.get("OV_THREADS", "4"))
        cfg["PERFORMANCE_HINT"] = "LATENCY"
    else:
        cfg["PERFORMANCE_HINT"] = "LATENCY"
        # iGPU 上 INT4 权重配合动态量化激活，通常还能再快一档
        cfg["DYNAMIC_QUANTIZATION_GROUP_SIZE"] = "32"
        cfg["KV_CACHE_PRECISION"] = "u8"
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(model, device, **cfg)
    print(f"[load] {device} 就绪，耗时 {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return pipe


def translate(pipe, text: str, tgt: str, max_new_tokens: int, stream: bool):
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False          # 翻译走贪心，稳定可复现
    cfg.repetition_penalty = 1.05

    prompt = build_prompt(text, tgt)
    tok = pipe.get_tokenizer()
    try:
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], True)
    except Exception:
        pass  # 模板不可用就直接喂原始 prompt

    t0 = time.perf_counter()
    if stream:
        out_parts = []

        def cb(chunk: str):
            out_parts.append(chunk)
            print(chunk, end="", flush=True)
            return ov_genai.StreamingStatus.RUNNING

        res = pipe.generate(prompt, cfg, cb)
        print()
        out = "".join(out_parts)
    else:
        res = pipe.generate(prompt, cfg)
        out = str(res)
    dt = time.perf_counter() - t0

    n = None
    try:
        pm = res.perf_metrics
        n = pm.get_num_generated_tokens()
        print(
            f"[perf] {n} tok / {dt:.1f}s = {n / dt:.2f} tok/s | "
            f"首字 {pm.get_ttft().mean / 1000:.2f}s | 每字 {pm.get_tpot().mean:.0f}ms",
            file=sys.stderr,
        )
    except Exception:
        print(f"[perf] {dt:.1f}s", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL))
    ap.add_argument("--device", default="GPU", help="GPU / CPU / AUTO")
    ap.add_argument("--to", dest="tgt", default="中文", help="目标语言，如 中文 / English / 日本語")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("-f", "--file", help="输入文件；缺省读 stdin / 交互")
    ap.add_argument("-o", "--out", help="输出文件")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--cache", default=os.environ.get("OV_CACHE"))
    args = ap.parse_args()

    pipe = make_pipe(args.model, args.device, args.cache)
    stream = not args.no_stream

    if args.file:
        src = Path(args.file).read_text(encoding="utf-8")
    elif not sys.stdin.isatty():
        src = sys.stdin.read()
    else:
        src = None

    if src is not None:
        # 按空行切段，逐段翻译，避免一次塞太长
        segs = [s for s in src.split("\n\n") if s.strip()]
        outs = [translate(pipe, s, args.tgt, args.max_new_tokens, stream) for s in segs]
        text = "\n\n".join(outs)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"[done] 写入 {args.out}", file=sys.stderr)
        return

    print(f"交互模式（目标语言 {args.tgt}），Ctrl-D 退出", file=sys.stderr)
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line.strip():
            translate(pipe, line, args.tgt, args.max_new_tokens, stream)


if __name__ == "__main__":
    main()
