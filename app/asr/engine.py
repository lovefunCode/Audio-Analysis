"""ASR：FunASR Paraformer-large + FSMN-VAD + CT-Punc。
输出带标点全文 + 句级时间线（sentence_info），用于语速/停顿/语气词统计。
"""
import os
import re
import threading
from dataclasses import dataclass, field

import numpy as np

from app.config import settings

# 独立语气词（单字，Paraformer 常输出为单独字符）
FILLER_CHARS = ("嗯", "呃", "啊", "哦", "噢", "诶", "欸")
# 口头禅短语
FILLER_PHRASES = ("然后呢", "然后", "就是说", "就是", "那个", "这个", "其实", "对吧", "怎么说呢", "的话")


@dataclass
class ASRResult:
    full_text: str = ""
    timeline: list = field(default_factory=list)      # [(text, start_s, end_s)]
    char_count: int = 0                              # 去标点字数
    speech_s: float = 0.0                            # 句级累计语音时长
    filler_counts: dict = field(default_factory=dict)


_PUNC_RE = re.compile(r"[，。！？、；：,.!?;:\s“”\"'（）()《》…—-]")


def _count_fillers(timeline: list) -> dict:
    counts = {}
    for text, _, _ in timeline:
        # 单字语气词：仅统计出现在句首或标点后的（避免"是啊"之类正常语气）
        for ch in FILLER_CHARS:
            n = len(re.findall(rf"(?:^|[，。！？、,.!?\s]){ch}", text))
            if n:
                counts[ch] = counts.get(ch, 0) + n
        for ph in FILLER_PHRASES:
            n = text.count(ph)
            if n:
                counts[ph] = counts.get(ph, 0) + n
    # "然后呢" 已包含 "然后"，去重
    if "然后呢" in counts and "然后" in counts:
        counts["然后"] -= counts["然后呢"]
        if counts["然后"] <= 0:
            counts.pop("然后")
    if "就是说" in counts and "就是" in counts:
        counts["就是"] -= counts["就是说"]
        if counts["就是"] <= 0:
            counts.pop("就是")
    return counts


class ASRPipeline:
    def __init__(self):
        self._model = None
        self._lock = threading.Lock()     # funasr 模型非线程安全

    def load(self):
        if self._model is not None:
            return
        from funasr import AutoModel
        base = settings.asr_model_path
        vad = os.path.join(base, "vad")
        punc = os.path.join(base, "punc")
        self._model = AutoModel(
            model=base,
            vad_model=vad if os.path.isdir(vad) else "fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model=punc if os.path.isdir(punc) else "ct-punc",
            device=settings.asr_device,
            disable_update=True,
            disable_pbar=True,
        )

    def transcribe(self, samples: np.ndarray) -> ASRResult:
        """输入 16k float32 波形（直接传数组，免二次落盘）。"""
        self.load()
        with self._lock:
            res = self._model.generate(
                input=samples,
                batch_size_s=settings.asr_batch_size_s,
                sentence_timestamp=True,
            )
        out = ASRResult()
        texts = []
        for item in res or []:
            text = (item.get("text") or "").strip()
            if text:
                texts.append(text)
            for sent in item.get("sentence_info") or []:
                st = (sent.get("text") or "").strip()
                if st:
                    out.timeline.append((st, sent.get("start", 0) / 1000.0,
                                         sent.get("end", 0) / 1000.0))
        out.full_text = "".join(texts)
        if not out.timeline and out.full_text:
            # 无句级时间戳时退化为单句
            out.timeline = [(out.full_text, 0.0, samples.size / settings.target_sr)]
        out.char_count = len(_PUNC_RE.sub("", out.full_text))
        out.speech_s = sum(max(0.0, e - s) for _, s, e in out.timeline)
        out.filler_counts = _count_fillers(out.timeline)
        return out


def format_timeline(timeline: list, offset_s: float, max_chars: int) -> str:
    """带时间标记的转写文本（每约 30 秒插一个 [mm:ss]），让 LLM 能定位问题位置。
    超长时保留头 60% + 尾 30%，中间均匀抽样。"""
    lines, last_mark = [], -999.0
    for text, s, _ in timeline:
        t = s + offset_s
        if t - last_mark >= 30:
            lines.append(f"\n[{int(t // 60):02d}:{int(t % 60):02d}] ")
            last_mark = t
        lines.append(text)
    full = "".join(lines).strip()
    if len(full) <= max_chars:
        return full
    head = full[: int(max_chars * 0.6)]
    tail = full[-int(max_chars * 0.3):]
    mid = full[len(head): len(full) - len(tail)]
    # 中段均匀抽取若干 200 字片段，总量约 10%
    budget, piece = int(max_chars * 0.1), 200
    n_pieces = max(1, budget // piece)
    stride = max(piece, len(mid) // n_pieces)
    sample = "……".join(mid[i: i + piece] for i in range(0, len(mid), stride))[:budget]
    return f"{head}\n[……中段节选……]\n{sample}\n[……]\n{tail}"
