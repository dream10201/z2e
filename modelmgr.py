#!/usr/bin/env python
"""模型注册表 + pipeline 生命周期管理。

约定：每个模型是 $MODELS_ROOT 下的一个目录，里面有 openvino_model.xml
（decoder-only，LLMPipeline）或 openvino_language_model.xml（多模态 VLM，
VLMPipeline，例如 qwen3_5 / qwen3_vl 这类只能按 image-text-to-text 导出的架构）。
目录名由 HF repo id 派生，例如 tencent/Hy-MT2-7B -> Hy-MT2-7B-int4-ov。

seq2seq 翻译模型（NLLB / M2M100 / Opus-MT 等）走的是另一套 API，这里不支持，
导出阶段会直接报错。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import openvino_genai as ov_genai

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
WEIGHT_FORMAT = os.environ.get("WEIGHT_FORMAT", "int4")
DEFAULT_MODEL = os.environ.get("MODEL_DIR") or os.environ.get("MODEL_ID", "tencent/Hy-MT2-7B")

# 不设 MODEL_ALLOWLIST 时，允许 API 自动导出的模型：7B 级以下、不用签协议就能下、
# N305（~16 GB 内存 + 32EU iGPU）上 INT4 跑得动的
N305_SAFE_MODELS = [
    "tencent/Hy-MT2-7B",
    "tencent/Hy-MT2-1.8B",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]


def _allowlist() -> list[str] | None:
    """MODEL_ALLOWLIST 环境变量，逗号分隔；没设返回 None。每次读，测试/热改方便。"""
    raw = os.environ.get("MODEL_ALLOWLIST", "").strip()
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def pull_allowed(model_id: str) -> bool:
    """这个模型允许通过 API 触发导出吗？"""
    al = _allowlist()
    if al is None:
        return model_id in N305_SAFE_MODELS or model_id == DEFAULT_MODEL
    return "*" in al or model_id in al


def serve_allowed(entry: ModelEntry) -> bool:
    """这个已导出的模型允许被切换使用吗？不设 MODEL_ALLOWLIST 就全放行。"""
    al = _allowlist()
    if al is None or "*" in al:
        return True
    names = set(al) | {dir_name_for(a) for a in al}
    return entry.name in names or (entry.source in al if entry.source else False)


def dir_name_for(model_id: str, weight_format: str = WEIGHT_FORMAT) -> str:
    """tencent/Hy-MT2-7B -> Hy-MT2-7B-int4-ov"""
    base = model_id.rstrip("/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "-", base)
    return f"{base}-{weight_format}-ov"


def dir_for(model_id: str, weight_format: str = WEIGHT_FORMAT) -> Path:
    """已经是绝对路径就原样返回，否则按 repo id 派生。"""
    if "/" in model_id and Path(model_id).is_absolute():
        return Path(model_id)
    return MODELS_ROOT / dir_name_for(model_id, weight_format)


def is_exported(path: Path) -> bool:
    return (path / "openvino_model.xml").is_file() or is_vlm(path)


def is_vlm(path: Path) -> bool:
    """多模态导出产物：语言塔叫 openvino_language_model.xml，没有 openvino_model.xml。"""
    return (
        not (path / "openvino_model.xml").is_file()
        and (path / "openvino_language_model.xml").is_file()
    )


@dataclass
class ModelEntry:
    name: str                    # 目录名，也是 API 里的 model id
    path: Path
    source: str | None = None    # 导出时用的 HF repo id（记在 .z2e.json 里）
    size_bytes: int = 0

    def as_openai(self) -> dict:
        return {
            "id": self.name,
            "object": "model",
            "created": int(self.path.stat().st_mtime),
            "owned_by": (self.source or "").split("/")[0] or "local",
        }


# 每个带 model 字段的请求都会 resolve -> scan，短 TTL 缓存挡掉高频目录遍历，
# 又不至于让刚导完的模型迟迟不出现
_SCAN_TTL = 5.0
_scan_cache: tuple[float, dict[str, ModelEntry]] | None = None


def scan() -> dict[str, ModelEntry]:
    """扫 MODELS_ROOT，返回 {目录名: ModelEntry}。结果缓存几秒。"""
    global _scan_cache
    now = time.monotonic()
    if _scan_cache is not None and now - _scan_cache[0] < _SCAN_TTL:
        return _scan_cache[1]
    out: dict[str, ModelEntry] = {}
    if not MODELS_ROOT.is_dir():
        return out
    for p in sorted(MODELS_ROOT.iterdir()):
        if not p.is_dir() or p.name.startswith(".") or p.name.endswith(".tmp"):
            continue
        if not is_exported(p):
            continue
        source = None
        meta = p / ".z2e.json"
        if meta.is_file():
            try:
                source = json.loads(meta.read_text()).get("model_id")
            except Exception:
                pass
        size = sum(f.stat().st_size for f in p.glob("*.bin"))
        out[p.name] = ModelEntry(name=p.name, path=p, source=source, size_bytes=size)
    _scan_cache = (now, out)
    return out


def resolve(model_ref: str | None) -> ModelEntry | None:
    """把请求里的 model 字段解析成注册表条目。

    接受目录名（Hy-MT2-7B-int4-ov）、HF repo id（tencent/Hy-MT2-7B）、
    以及裸名字（Hy-MT2-7B）。
    """
    if not model_ref:
        return None
    # 绝对路径直接用，不要求它在 MODELS_ROOT 下
    p = Path(model_ref)
    if p.is_absolute():
        if not is_exported(p):
            return None
        source = None
        meta = p / ".z2e.json"
        if meta.is_file():
            try:
                source = json.loads(meta.read_text()).get("model_id")
            except Exception:
                pass
        return ModelEntry(name=p.name, path=p, source=source)
    reg = scan()
    if model_ref in reg:
        return reg[model_ref]
    derived = dir_name_for(model_ref)
    if derived in reg:
        return reg[derived]
    for e in reg.values():
        if e.source == model_ref:
            return e
    return None


# ---------- pipeline 生命周期 ----------

def make_pipe(
    model: str, device: str, cache: str | None
) -> ov_genai.LLMPipeline | ov_genai.VLMPipeline:
    cfg: dict[str, object] = {}
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
    if is_vlm(Path(model)):
        # 多模态产物用 VLMPipeline；纯文本对话它也能跑，图片是可选输入
        pipe = ov_genai.VLMPipeline(model, device, **cfg)
        print(f"[load] VLM {device} 就绪，耗时 {time.perf_counter() - t0:.1f}s", file=sys.stderr)
        return pipe
    pipe: ov_genai.LLMPipeline | None = None
    # agent 类客户端每轮都带完整历史，prompt 是前缀递增的；前缀缓存能让每轮
    # 只 prefill 新增部分。走的是 continuous-batching 后端，KV cache 常驻显存，
    # 在内存紧张的机器上可能装不下，所以默认关，OV_PREFIX_CACHING=1 打开。
    if os.environ.get("OV_PREFIX_CACHING", "0") == "1":
        try:
            sched = ov_genai.SchedulerConfig()
            sched.enable_prefix_caching = True
            pipe = ov_genai.LLMPipeline(model, device, scheduler_config=sched, **cfg)
        except Exception as e:
            print(f"[load] 前缀缓存后端不可用（{e}），退回普通 pipeline", file=sys.stderr)
    if pipe is None:
        pipe = ov_genai.LLMPipeline(model, device, **cfg)
    print(f"[load] {device} 就绪，耗时 {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return pipe

class PipelineManager:
    """一次只驻留一个模型。N305 上也塞不下两个 7B。

    generate 和切换共用一把锁：GenAI 的 pipeline 都不是线程安全的，而且这颗 U 上
    并发解码只会互相拖慢。
    """

    def __init__(self, device: str, cache: str | None = None):
        self.device = device
        self.cache = cache
        # lock 保护 pipeline 状态本身（可重入，同线程内 load 嵌套调用没问题）
        self.lock = threading.RLock()
        # gen_lock 是请求级互斥：调用方从"解析模型"一直持有到"生成结束"，
        # 否则并发请求指定不同 model 时，会出现用 B 模型生成却按 A 模型上报。
        # 用普通 Lock 而非 RLock，因为流式响应要在另一个线程里释放它。
        self.gen_lock = threading.Lock()
        self._pipe: ov_genai.LLMPipeline | ov_genai.VLMPipeline | None = None
        self._entry: ModelEntry | None = None
        self._is_vlm = False
        self._load_seconds: float = 0.0

    @property
    def current(self) -> ModelEntry | None:
        return self._entry

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    @property
    def is_vlm(self) -> bool:
        """当前加载的模型是不是多模态（能吃图片输入）。"""
        return self._is_vlm

    def pipe(self) -> ov_genai.LLMPipeline | ov_genai.VLMPipeline:
        if self._pipe is None:
            raise RuntimeError("还没有加载模型")
        return self._pipe

    def load(self, entry: ModelEntry) -> None:
        with self.lock:
            if self._entry is not None and self._entry.path == entry.path:
                return
            # 先释放旧的，别让两个模型同时占内存
            self._pipe = None
            self._entry = None
            t0 = time.perf_counter()
            self._pipe = make_pipe(str(entry.path), self.device, self.cache)
            self._load_seconds = time.perf_counter() - t0
            self._entry = entry
            self._is_vlm = is_vlm(entry.path)

    def unload(self) -> None:
        with self.lock:
            self._pipe = None
            self._entry = None
            self._is_vlm = False


# ---------- 后台导出 ----------

@dataclass
class ExportJob:
    model_id: str
    status: str = "pending"          # pending / running / done / failed
    started: float = field(default_factory=time.time)
    finished: float | None = None
    message: str = ""
    target: str = ""

    @property
    def log_path(self) -> Path:
        return MODELS_ROOT / f".export-{Path(self.target).name}.log"

    def tail(self, n: int = 2000) -> str:
        try:
            return self.log_path.read_text(errors="replace")[-n:]
        except Exception:
            return ""


class Exporter:
    """一次跑一个导出任务，状态可查。导出很慢（7B 要 30-60 分钟），
    所以是后台任务 + 轮询状态，不是同步 HTTP 请求。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.job: ExportJob | None = None
        self._thread: threading.Thread | None = None

    @staticmethod
    def available() -> bool:
        if not Path("/app/export_int4.sh").is_file() and not Path("export_int4.sh").is_file():
            return False
        try:
            import optimum.intel  # noqa: F401
            return True
        except Exception:
            return False

    @property
    def busy(self) -> bool:
        return self.job is not None and self.job.status in ("pending", "running")

    def start(self, model_id: str) -> ExportJob:
        with self.lock:
            if self.busy:
                raise RuntimeError(f"已有导出在跑: {self.job.model_id}")
            target = dir_for(model_id)
            job = ExportJob(model_id=model_id, target=str(target))
            self.job = job
            self._thread = threading.Thread(target=self._run, args=(job,), daemon=True)
            self._thread.start()
            return job

    def _run(self, job: ExportJob) -> None:
        job.status = "running"
        script = "/app/export_int4.sh" if Path("/app/export_int4.sh").is_file() else "export_int4.sh"
        env = {**os.environ, "MODEL_ID": job.model_id, "OUT": job.target}
        timeout = int(os.environ.get("EXPORT_TIMEOUT", "21600"))
        killed = threading.Event()
        try:
            # 导出要跑几十分钟，输出双写：落盘给 GET /admin/pull 读尾巴，
            # 同时透传到本进程 stdout——docker/podman logs 里直接能看进度。
            # 按块转发而不是按行：HF 下载进度条用 \r 刷新，按行读会攒很久才吐
            MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            with open(job.log_path, "wb") as log:
                p = subprocess.Popen(
                    ["bash", script], env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                timer = threading.Timer(timeout, lambda: (killed.set(), p.kill()))
                timer.start()
                try:
                    while chunk := p.stdout.read1(65536):
                        log.write(chunk)
                        log.flush()
                        sys.stdout.buffer.write(chunk)
                        sys.stdout.flush()
                    rc = p.wait()
                finally:
                    timer.cancel()
            if rc == 0 and is_exported(Path(job.target)):
                Path(job.target, ".z2e.json").write_text(
                    json.dumps({"model_id": job.model_id, "weight_format": WEIGHT_FORMAT})
                )
                job.status = "done"
            else:
                job.status = "failed"
            job.message = (f"导出超过 {timeout}s 被终止\n" if killed.is_set() else "") + job.tail()
        except Exception as e:
            job.status = "failed"
            job.message = f"{e}\n{job.tail()}"
        finally:
            job.finished = time.time()
