"""ASR：FunASR Paraformer-large + FSMN-VAD + CT-Punc。
输出带标点全文 + 句级时间线（sentence_info）+ 字/词级时间戳，
并给出语速、停顿、语气词位置等统计，供报告直接引用。
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

# 停顿判定阈值（秒）
PAUSE_MIN_S = 0.5          # 句间空隙 ≥ 此值计入"停顿"
LONG_PAUSE_S = 1.5         # 句间空隙 ≥ 此值计入"长停顿"


@dataclass
class ASRResult:
    full_text: str = ""
    timeline: list = field(default_factory=list)      # [(text, start_s, end_s)]
    words: list = field(default_factory=list)         # [(token, start_s, end_s)] 字/词级时间戳
    char_count: int = 0                              # 去标点字数
    speech_s: float = 0.0                            # 句级累计语音时长
    filler_counts: dict = field(default_factory=dict)
    filler_events: list = field(default_factory=list)  # [(filler, start_s, end_s, sentence_idx)]
    pauses: list = field(default_factory=list)         # [(start_s, end_s, gap_s)]，gap 均 ≥ PAUSE_MIN_S

    # ── 派生指标 ──
    audio_s: float = 0.0                              # 音频总时长
    sentence_count: int = 0
    avg_sentence_chars: float = 0.0                   # 平均每句字数
    longest_sentence_chars: int = 0
    speech_rate_cpm: float = 0.0                      # 语速：字/分钟（按有效语音时长）
    speaking_ratio: float = 0.0                       # 有效语音 / 音频总长
    pause_count: int = 0                              # 停顿次数
    long_pause_count: int = 0                         # 长停顿次数
    max_pause_s: float = 0.0                          # 最长停顿
    pause_ratio: float = 0.0                          # 停顿总时长 / 音频总长


_PUNC_RE = re.compile(r"[，。！？、；：,.!?;:\s“”\"'（）()《》…—-]")

# 短语按长度降序拼成一个正则：同一位置优先匹配更长短语，
# 天然避免 "然后呢" 再被重复匹配成 "然后"、"就是说" 再匹配成 "就是"。
_FILLER_PHRASE_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(FILLER_PHRASES, key=len, reverse=True)))
# 单字语气词：前面必须是句首/标点/空白，捕获组 1 即语气词本身
# （用 str.format 而非 f-string 内嵌引号，兼容 Python 3.11）
_FILLER_CHAR_RE = re.compile(
    r"(?:^|[，。！？、,.!?\s])([{}])".format("".join(FILLER_CHARS)))


def _time_at(text: str, pos: int, start_s: float, end_s: float) -> float:
    """把句内字符偏移按去标点字数比例换算成绝对时间。"""
    total = len(_PUNC_RE.sub("", text)) or 1
    before = len(_PUNC_RE.sub("", text[:pos]))
    return start_s + (end_s - start_s) * before / total


def _find_fillers(timeline: list) -> list:
    """定位语气词/口头禅：[(filler, start_s, end_s, sentence_idx)]。

    - 短语一次正则扫描，长短语优先，不会重复计数；
    - 单字语气词只认句首或标点/空白之后（避免"是啊"这类正常语气）；
    - 落在已匹配短语范围内的单字不再单独计数。
    """
    events = []
    for idx, (text, s, e) in enumerate(timeline):
        phrase_spans = []
        for m in _FILLER_PHRASE_RE.finditer(text):
            phrase_spans.append((m.start(), m.end()))
            st = _time_at(text, m.start(), s, e)
            en = _time_at(text, m.end(), s, e)
            events.append((m.group(), st, max(st, en), idx))
        for m in _FILLER_CHAR_RE.finditer(text):
            pos = m.start(1)
            if any(a <= pos < b for a, b in phrase_spans):
                continue
            st = _time_at(text, pos, s, e)
            en = _time_at(text, pos + 1, s, e)
            events.append((m.group(1), st, max(st, en), idx))
    events.sort(key=lambda ev: ev[1])
    return events


def _count_fillers(events: list) -> dict:
    counts = {}
    for filler, _, _, _ in events:
        counts[filler] = counts.get(filler, 0) + 1
    return counts


def _extract_words(sentence_info: list) -> list:
    """从 sentence_info[].timestamp 提取字/词级时间戳：[(token, start_s, end_s)]。

    FunASR 该字段形如 [[id, token, start_ms], ...]；每个 token 的结束时间取下一个
    token 的起点，末 token 取句尾。标点 token 与时间戳 < 0 的条目跳过。
    """
    words = []
    for sent in sentence_info or []:
        ts = sent.get("timestamp") or []
        sent_end_s = float(sent.get("end", 0)) / 1000.0
        for i, item in enumerate(ts):
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            tok = str(item[1]).strip()
            if not tok or _PUNC_RE.fullmatch(tok):
                continue
            try:
                st_ms = float(item[2])
            except (TypeError, ValueError):
                continue
            if st_ms < 0:
                continue
            en_ms = sent_end_s * 1000.0
            nxt = ts[i + 1] if i + 1 < len(ts) else None
            if isinstance(nxt, (list, tuple)) and len(nxt) >= 3:
                try:
                    if float(nxt[2]) >= st_ms:
                        en_ms = float(nxt[2])
                except (TypeError, ValueError):
                    pass
            words.append((tok, st_ms / 1000.0, max(st_ms, en_ms) / 1000.0))
    return words


def _detect_pauses(timeline: list, min_gap_s: float = PAUSE_MIN_S) -> list:
    """相邻句之间的静音空隙：[(prev_end_s, next_start_s, gap_s)]，仅保留 ≥ min_gap_s。"""
    pauses = []
    for (_, _, prev_end), (_, next_start, _) in zip(timeline, timeline[1:]):
        gap = next_start - prev_end
        if gap >= min_gap_s:
            pauses.append((prev_end, next_start, gap))
    return pauses


def _fill_metrics(out: ASRResult, audio_s: float) -> None:
    """汇总语速、停顿、篇幅等派生指标。"""
    out.audio_s = audio_s
    lens = [len(_PUNC_RE.sub("", t)) for t, _, _ in out.timeline]
    out.sentence_count = len(lens)
    out.avg_sentence_chars = (sum(lens) / len(lens)) if lens else 0.0
    out.longest_sentence_chars = max(lens, default=0)
    out.speech_rate_cpm = out.char_count / (out.speech_s / 60.0) if out.speech_s > 0.5 else 0.0
    out.speaking_ratio = min(1.0, out.speech_s / audio_s) if audio_s > 0 else 0.0
    out.pause_count = len(out.pauses)
    out.long_pause_count = sum(1 for _, _, d in out.pauses if d >= LONG_PAUSE_S)
    out.max_pause_s = max((d for _, _, d in out.pauses), default=0.0)
    pause_total = sum(d for _, _, d in out.pauses)
    out.pause_ratio = min(1.0, pause_total / audio_s) if audio_s > 0 else 0.0


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
            sentence_info = item.get("sentence_info") or []
            for sent in sentence_info:
                st = (sent.get("text") or "").strip()
                if st:
                    out.timeline.append((st, sent.get("start", 0) / 1000.0,
                                         sent.get("end", 0) / 1000.0))
            out.words.extend(_extract_words(sentence_info))
        out.full_text = "".join(texts)
        if not out.timeline and out.full_text:
            # 无句级时间戳时退化为单句
            out.timeline = [(out.full_text, 0.0, samples.size / settings.target_sr)]
        out.char_count = len(_PUNC_RE.sub("", out.full_text))
        out.speech_s = sum(max(0.0, e - s) for _, s, e in out.timeline)
        out.filler_events = _find_fillers(out.timeline)
        out.filler_counts = _count_fillers(out.filler_events)
        out.pauses = _detect_pauses(out.timeline)
        _fill_metrics(out, samples.size / settings.target_sr)
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
