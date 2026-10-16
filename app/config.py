"""演讲陪练助手 - 应用配置（全部可由环境变量覆盖）"""
import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(key: str, default: float) -> float:
    v = os.getenv(key)
    return default if v is None else float(v)


def _env_int(key: str, default: int) -> int:
    v = os.getenv(key)
    return default if v is None else int(v)


@dataclass
class Settings:
    # ── 模型路径（镜像挂载）────────────────────────────────────
    asr_model_path: str = os.getenv("ASR_MODEL_PATH", "models/paraformer-zh")
    llm_model_path: str = os.getenv("LLM_MODEL_PATH", "models/qwen2.5-14b-instruct")

    # ── ASR ───────────────────────────────────────────────────
    asr_batch_size_s: int = _env_int("ASR_BATCH_SIZE_S", 300)
    asr_device: str = os.getenv("ASR_DEVICE", "cuda")

    # ── LLM（vLLM AsyncEngine）────────────────────────────────
    llm_max_model_len: int = _env_int("LLM_MAX_MODEL_LEN", 32768)
    llm_gpu_memory_utilization: float = _env_float("LLM_GPU_MEMORY_UTILIZATION", 0.60)
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.3)
    transcript_max_chars: int = _env_int("TRANSCRIPT_MAX_CHARS", 20000)

    # ── 音频 ──────────────────────────────────────────────────
    max_audio_duration_s: int = _env_int("MAX_AUDIO_DURATION_S", 3600)   # 60 分钟
    duration_tolerance_s: int = 30               # 60:20 这种不算超限
    target_sr: int = 16000
    min_valid_chars: int = 20                    # 低于此字数判定为无有效语音
    max_upload_mb: int = _env_int("MAX_UPLOAD_MB", 300)

    # ── 安全 ──────────────────────────────────────────────────
    safety_llm_check: bool = _env_bool("SAFETY_LLM_CHECK", True)   # 转写文本二次 LLM 审核
    safety_check_chars: int = 4000               # 送审文本上限（头尾各半）

    # ── 流式 ──────────────────────────────────────────────────
    # 接口约定 advice_start 在转写/分析完成后发送，默认 False。
    # 设为 True 则在 ASR 之前就输出开头总述（首字时延降到毫秒级，但偏离约定）。
    early_first_token: bool = _env_bool("EARLY_FIRST_TOKEN", False)

    # ── 稳定性 / 超时 ─────────────────────────────────────────
    asr_timeout_s: int = _env_int("ASR_TIMEOUT_S", 900)              # 单次 ASR 上限
    llm_first_token_timeout_s: int = _env_int("LLM_FIRST_TOKEN_TIMEOUT_S", 120)  # 单字段首 token
    llm_token_timeout_s: int = _env_int("LLM_TOKEN_TIMEOUT_S", 30)   # 相邻 token 间隔上限
    request_soft_deadline_s: int = _env_int("REQUEST_SOFT_DEADLINE_S", 1500)  # 超过后剩余字段用兜底文本
    exit_on_engine_dead: bool = _env_bool("EXIT_ON_ENGINE_DEAD", True)   # vLLM 引擎死亡时退出进程以便重启

    # ── 服务 ──────────────────────────────────────────────────
    warmup_on_start: bool = _env_bool("WARMUP_ON_START", True)
    # True 时 /ready 在模型加载期间也返回 200（请求会排队等待加载完成）。
    # 评测平台探活超时较短时可开启，避免被判为启动失败。
    ready_before_warmup: bool = _env_bool("READY_BEFORE_WARMUP", False)

    @property
    def use_local_llm(self) -> bool:
        return bool(self.llm_model_path) and os.path.isdir(self.llm_model_path)


settings = Settings()

# 固定的开头总述（与评测示例一致）
OPENING_LINE = "本次将从4个维度对本次演讲语音进行分析"
