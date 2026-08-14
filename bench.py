#!/usr/bin/env python
"""对比 CPU / iGPU 上 Hunyuan-MT-7B INT4 的实际吞吐。

python bench.py --devices CPU GPU --runs 2
"""
import argparse
import os
import time

import openvino_genai as ov_genai

from translate import DEFAULT_MODEL, build_prompt, make_pipe

SAMPLES = [
    ("The quick brown fox jumps over the lazy dog. Neural machine translation has "
     "improved substantially over the past decade, driven by large pretrained models.", "中文"),
    ("我们计划在下个季度把推理服务迁移到边缘设备上，这样可以显著降低时延和带宽成本。", "English"),
]


def run(device: str, model: str, runs: int, max_new_tokens: int):
    pipe = make_pipe(model, device, os.environ.get("OV_CACHE"))
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False

    tok = pipe.get_tokenizer()
    rows = []
    for text, tgt in SAMPLES:
        p = build_prompt(text, tgt)
        try:
            p = tok.apply_chat_template([{"role": "user", "content": p}], True)
        except Exception:
            pass
        for i in range(runs):
            t0 = time.perf_counter()
            res = pipe.generate(p, cfg)
            dt = time.perf_counter() - t0
            pm = res.perf_metrics
            n = pm.get_num_generated_tokens()
            rows.append((n, dt, pm.get_ttft().mean / 1000, n / dt))
            if i == 0:
                print(f"  [{device}] {str(res)[:80]}")
    # 丢掉第一条预热
    warm = rows[1:] or rows
    avg = sum(r[3] for r in warm) / len(warm)
    ttft = sum(r[2] for r in warm) / len(warm)
    print(f"== {device}: {avg:.2f} tok/s, 首字 {ttft:.2f}s ==")
    return avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_DIR", DEFAULT_MODEL))
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU"])
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    res = {d: run(d, args.model, args.runs, args.max_new_tokens) for d in args.devices}
    print("\n--- 汇总 ---")
    for d, v in res.items():
        print(f"{d:>4}: {v:.2f} tok/s")
    if "CPU" in res and "GPU" in res:
        print(f"GPU/CPU = {res['GPU'] / res['CPU']:.2f}x")


if __name__ == "__main__":
    main()
