#!/usr/bin/env python
"""模型注册表 + pipeline 生命周期管理。

约定：每个模型是 $MODELS_ROOT 下的一个目录，里面有 openvino_model.xml。
目录名由 HF repo id 派生，例如 tencent/Hunyuan-MT-7B -> Hunyuan-MT-7B-int4-ov。

只支持 decoder-only 因果语言模型（openvino_genai.LLMPipeline 的适用范围）。
seq2seq 翻译模型（NLLB / M2M100 / Opus-MT 等）走的是另一套 API，这里不支持，
导出阶段会直接报错。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import openvino_genai as ov_genai

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
WEIGHT_FORMAT = os.environ.get("WEIGHT_FORMAT", "int4")


def dir_name_for(model_id: str, weight_format: str = WEIGHT_FORMAT) -> str:
    """tencent/Hunyuan-MT-7B -> Hunyuan-MT-7B-int4-ov"""
    base = model_id.rstrip("/").split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]", "-", base)
    return f"{base}-{weight_format}-ov"


def dir_for(model_id: str, weight_format: str = WEIGHT_FORMAT) -> Path:
    """已经是绝对路径就原样返回，否则按 repo id 派生。"""
    if "/" in model_id and Path(model_id).is_absolute():
        return Path(model_id)
    return MODELS_ROOT / dir_name_for(model_id, weight_format)


def is_exported(path: Path) -> bool:
    return (path / "openvino_model.xml").is_file()


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


def scan() -> dict[str, ModelEntry]:
    """扫 MODELS_ROOT，返回 {目录名: ModelEntry}。"""
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
    return out


def resolve(model_ref: str | None) -> ModelEntry | None:
    """把请求里的 model 字段解析成注册表条目。

    接受目录名（Hunyuan-MT-7B-int4-ov）、HF repo id（tencent/Hunyuan-MT-7B）、
    以及裸名字（Hunyuan-MT-7B）。
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

class PipelineManager:
    """一次只驻留一个模型。N305 上也塞不下两个 7B。

    generate 和切换共用一把锁：LLMPipeline 不是线程安全的，而且这颗 U 上
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
        self._pipe: ov_genai.LLMPipeline | None = None
        self._entry: ModelEntry | None = None
        self._load_seconds: float = 0.0

    @property
    def current(self) -> ModelEntry | None:
        return self._entry

    @property
    def load_seconds(self) -> float:
        return self._load_seconds

    def pipe(self) -> ov_genai.LLMPipeline:
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
            from translate import make_pipe  # 复用同一套设备配置

            t0 = time.perf_counter()
            self._pipe = make_pipe(str(entry.path), self.device, self.cache)
            self._load_seconds = time.perf_counter() - t0
            self._entry = entry

    def unload(self) -> None:
        with self.lock:
            self._pipe = None
            self._entry = None


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
        try:
            # 导出要跑几十分钟，输出直接落盘：容器日志和 GET /admin/pull 都能看到进度，
            # 不然整段过程是个黑盒
            MODELS_ROOT.mkdir(parents=True, exist_ok=True)
            with open(job.log_path, "w") as log:
                p = subprocess.run(
                    ["bash", script], env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=int(os.environ.get("EXPORT_TIMEOUT", "21600")),
                )
            if p.returncode == 0 and is_exported(Path(job.target)):
                Path(job.target, ".z2e.json").write_text(
                    json.dumps({"model_id": job.model_id, "weight_format": WEIGHT_FORMAT})
                )
                job.status = "done"
            else:
                job.status = "failed"
            job.message = job.tail()
        except Exception as e:
            job.status = "failed"
            job.message = f"{e}\n{job.tail()}"
        finally:
            job.finished = time.time()
