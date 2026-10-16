"""vLLM 异步引擎封装：真·逐 token 流式 + prefix caching。

报告按字段续写：每次在同一前缀后追加一个字段，前缀 KV 命中缓存，
因此 10 次短生成的总耗时接近一次长生成。

稳定性：
  - 首 token / 相邻 token 超时（防止引擎卡死导致请求挂起）
  - 引擎死亡（CUDA error 等）后标记 dead，由 server 退出进程交给容器重启
"""
import asyncio
import logging
import uuid
from typing import AsyncIterator

from app.config import settings

log = logging.getLogger("coach.llm")


class LLMError(RuntimeError):
    """LLM 调用失败（超时/引擎异常），上层用兜底文本继续。"""


class LLMEngine:
    def __init__(self):
        self._engine = None
        self._tokenizer = None
        self.dead = False

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def load(self):
        if self._engine is not None:
            return
        if not settings.use_local_llm:
            raise RuntimeError(f"未找到 LLM 模型目录 {settings.llm_model_path!r}，请挂载模型并设置 LLM_MODEL_PATH")
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        self._tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_path, trust_remote_code=True)
        args = AsyncEngineArgs(
            model=settings.llm_model_path,
            max_model_len=settings.llm_max_model_len,
            gpu_memory_utilization=settings.llm_gpu_memory_utilization,
            enable_prefix_caching=True,
            trust_remote_code=True,
            disable_log_requests=True,
        )
        self._engine = AsyncLLMEngine.from_engine_args(args)

    def chat_prefix(self, system: str, user: str) -> str:
        """渲染 chat 模板到 assistant 开始处，之后可直接拼接预填文本。"""
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _check_alive(self):
        if self.dead:
            raise LLMError("engine dead")
        if self._engine is None:
            raise LLMError("engine not loaded")
        if getattr(self._engine, "errored", False):
            self.dead = True
            raise LLMError("engine errored")

    async def stream(self, prompt: str, max_tokens: int, stop: list | None = None,
                     temperature: float | None = None) -> AsyncIterator[str]:
        """对原始 prompt 续写，逐增量 yield 文本。超时/引擎异常抛 LLMError。"""
        self._check_alive()
        from vllm import SamplingParams
        sp = SamplingParams(
            temperature=settings.llm_temperature if temperature is None else temperature,
            top_p=0.8,
            repetition_penalty=1.05,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        rid = uuid.uuid4().hex
        sent = 0
        it = self._engine.generate(prompt, sp, rid).__aiter__()
        timeout = settings.llm_first_token_timeout_s
        try:
            while True:
                try:
                    out = await asyncio.wait_for(it.__anext__(), timeout)
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    raise LLMError(f"llm timeout after {timeout}s (sent={sent})")
                text = out.outputs[0].text
                if len(text) > sent:
                    yield text[sent:]
                    sent = len(text)
                    timeout = settings.llm_token_timeout_s
        except LLMError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as ex:
            # vLLM AsyncEngineDeadError 等：引擎已不可用
            name = type(ex).__name__
            if "Dead" in name or getattr(self._engine, "errored", False):
                self.dead = True
                log.error("llm engine dead: %s", name)
            raise LLMError(f"llm failed: {name}") from ex
        finally:
            # 客户端断开/超时时释放 KV
            try:
                await self._engine.abort(rid)
            except Exception:
                pass

    async def complete(self, prompt: str, max_tokens: int, stop: list | None = None,
                       temperature: float | None = None) -> str:
        parts = []
        async for d in self.stream(prompt, max_tokens, stop, temperature):
            parts.append(d)
        return "".join(parts)


llm_engine = LLMEngine()
