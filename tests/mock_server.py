"""无 GPU / 无模型的端到端联调启动器。

用法：python tests/mock_server.py [port]

只替换两处"模型加载"：
  - ASRPipeline.load / transcribe  → 按音频时长生成假转写（带句级时间戳）
  - LLMEngine.load / chat_prefix / stream → 按前缀返回固定一行文本（逐字流式）
其余环节（安全规则、意图解析、ffprobe/ffmpeg 解码、声学/文本统计、结构化流式报告、
SSE 协议、/ready 探活）全部走真实代码。
"""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 本地工具链（ffmpeg/ffprobe）
os.environ["PATH"] = os.path.join(ROOT, ".tools", "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("SAFETY_LLM_CHECK", "false")

import numpy as np

from app.asr import engine as asr_mod
from app.llm import engine as llm_mod

# ─────────────────────────── ASR mock ───────────────────────────
_SENTENCES = [
    "大家好，今天我想和大家分享一下我们团队这一年的工作成果。",
    "嗯，首先呢，我们先来看一下整体的背景。",
    "其实这个项目一开始的时候，那个，我们遇到了很多困难。",
    "然后我们通过不断地尝试，最终找到了一个比较好的方案。",
    "就是说，大家可以看到这张图，数据增长了百分之三十。",
    "接下来我想讲第二点，关于团队协作的问题。",
    "那么，各位有没有想过，为什么沟通效率这么重要呢？",
    "我觉得这个问题值得我们每一个人去思考。",
    "最后，感谢大家的聆听，希望今天的分享对各位有帮助。",
]


def _mock_asr_load(self):
    self._model = "mock"


def _mock_transcribe(self, samples: np.ndarray) -> asr_mod.ASRResult:
    dur = samples.size / 16000.0
    out = asr_mod.ASRResult()
    t = 0.5
    i = 0
    while t < dur - 1.0:
        text = _SENTENCES[i % len(_SENTENCES)]
        seg = min(len(text) * 0.25, dur - t)      # 约 4 字/秒
        out.timeline.append((text, t, t + seg))
        t += seg + 0.8                            # 句间停顿
        i += 1
    out.full_text = "".join(s for s, _, _ in out.timeline)
    out.char_count = len(asr_mod._PUNC_RE.sub("", out.full_text))
    out.speech_s = sum(e - s for _, s, e in out.timeline)
    out.filler_counts = asr_mod._count_fillers(out.timeline)
    return out


asr_mod.ASRPipeline.load = _mock_asr_load
asr_mod.ASRPipeline.transcribe = _mock_transcribe

# ─────────────────────────── LLM mock ───────────────────────────
_REPLIES = {
    "• 优点：": "主题明确，开头「今天我想和大家分享」直接点题，结构上有第一点第二点的层次。",
    "• 待优化点：": "论据支撑偏弱，「数据增长了百分之三十」缺少来源与对比，建议补充案例。",
    "1• ": "开场30秒内用一句话说清这次分享对听众的价值，再进入正题。",
    "2• ": "每讲完一个要点停顿1-2秒，再用「第二点」开启下一段。",
    "3• ": "录音回放时标出「嗯」「那个」出现的位置，下次讲到同一处时刻意停顿代替。",
}


def _mock_llm_load(self):
    self._engine = "mock"
    self._tokenizer = "mock"


def _mock_chat_prefix(self, system: str, user: str) -> str:
    return f"<system>{system}</system>\n<user>{user}</user>\n<assistant>"


async def _mock_stream(self, prompt: str, max_tokens: int, stop=None, temperature=None):
    tail = prompt.rstrip("\n").rsplit("\n", 1)[-1]
    if "安全" in prompt[:200] and max_tokens <= 3:      # 安全复核：直接放行
        yield "安全"
        return
    reply = next((v for k, v in _REPLIES.items() if tail.endswith(k)), "这一维度整体表现良好。")
    for ch in reply:
        await asyncio.sleep(0.003)
        yield ch


llm_mod.LLMEngine.load = _mock_llm_load
llm_mod.LLMEngine.chat_prefix = _mock_chat_prefix
llm_mod.LLMEngine.stream = _mock_stream

# ─────────────────────────── 启动 ───────────────────────────
if __name__ == "__main__":
    import uvicorn
    from app.api.server import app
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8089
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
