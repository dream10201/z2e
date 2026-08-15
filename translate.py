#!/usr/bin/env python
"""翻译 CLI，走 OpenVINO GenAI（CPU 或 iGPU）。

任何 decoder-only 因果语言模型都能用；`--model` 收目录名、HF repo id 或绝对路径。

用法:
  python translate.py --device GPU                          # 交互式
  echo "hello world" | python translate.py --to 中文         # 管道批处理
  python translate.py --model Qwen/Qwen3-8B --to English -f in.txt -o out.txt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import openvino_genai as ov_genai

DEFAULT_MODEL = os.environ.get("MODEL_DIR") or os.environ.get("MODEL_ID", "tencent/Hunyuan-MT-7B")

# 翻译专用模型自带指定的 prompt 格式，用错了质量会掉。按模型名匹配 preset，
# 认不出就退化成通用指令（对 Qwen / Llama 这类通用 instruct 模型足够）。
TEMPLATE_PRESETS: dict[str, tuple[str, str]] = {
    # 模型名关键字: (中文向模板, 其他语向模板)
    "hunyuan-mt": (
        "把下面的文本翻译成{tgt}，不要额外解释。\n\n{text}",
        "Translate the following segment into {tgt}, without additional explanation.\n\n{text}",
    ),
    "seed-x": (
        "Translate the following text into {tgt}:\n{text} <{tgt}>",
        "Translate the following text into {tgt}:\n{text} <{tgt}>",
    ),
}
GENERIC = (
    "把下面的文本翻译成{tgt}，只输出译文，不要解释。\n\n{text}",
    "Translate the following text into {tgt}. Output only the translation.\n\n{text}",
)


def templates_for(model_hint: str | None) -> tuple[str, str]:
    """环境变量优先，其次按模型名匹配 preset，最后退化到通用模板。"""
    zh = os.environ.get("TRANSLATE_TEMPLATE_ZH")
    xx = os.environ.get("TRANSLATE_TEMPLATE")
    if zh or xx:
        return (zh or xx or GENERIC[0], xx or zh or GENERIC[1])
    key = (model_hint or "").lower()
    for name, tmpl in TEMPLATE_PRESETS.items():
        if name in key:
            return tmpl
    return GENERIC


def build_prompt(text: str, tgt: str, model_hint: str | None = None) -> str:
    zh_tmpl, xx_tmpl = templates_for(model_hint if model_hint is not None else DEFAULT_MODEL)
    zh_like = any(k in tgt for k in ("中文", "Chinese", "汉语"))
    tmpl = zh_tmpl if zh_like or _has_han(text) else xx_tmpl
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


def translate(pipe, text: str, tgt: str, max_new_tokens: int, stream: bool,
              model_hint: str | None = None):
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False          # 翻译走贪心，稳定可复现
    cfg.repetition_penalty = 1.05

    prompt = build_prompt(text, tgt, model_hint)
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
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="目录名 / HF repo id / 绝对路径")
    ap.add_argument("--device", default="GPU", help="GPU / CPU / AUTO")
    ap.add_argument("--to", dest="tgt", default="中文", help="目标语言，如 中文 / English / 日本語")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("-f", "--file", help="输入文件；缺省读 stdin / 交互")
    ap.add_argument("-o", "--out", help="输出文件")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--cache", default=os.environ.get("OV_CACHE"))
    args = ap.parse_args()

    import modelmgr

    path = modelmgr.dir_for(args.model)
    if not modelmgr.is_exported(path):
        print(f"[error] {path} 里没有 openvino_model.xml；先导出: "
              f"MODEL_ID={args.model} bash export_int4.sh", file=sys.stderr)
        sys.exit(1)
    pipe = make_pipe(str(path), args.device, args.cache)
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
        outs = [translate(pipe, s, args.tgt, args.max_new_tokens, stream, args.model)
                for s in segs]
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
            translate(pipe, line, args.tgt, args.max_new_tokens, stream, args.model)


if __name__ == "__main__":
    main()
