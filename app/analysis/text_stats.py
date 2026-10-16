"""转写文本统计 → 维度一（结构）、维度二（措辞）、维度四（听众联结）的客观证据"""
import re
from dataclasses import dataclass, field

from app.asr.engine import ASRResult

_STRUCTURE_WORDS = ["首先", "其次", "再次", "最后", "第一", "第二", "第三", "总之",
                    "综上", "总结", "因此", "所以", "另外", "此外", "接下来", "一方面", "另一方面"]
_AUDIENCE_WORDS = ["大家", "各位", "你们", "您", "我们", "咱们", "领导", "老师", "评委", "同学"]
_QUESTION_RE = re.compile(r"[？?]|吗[，。]|呢[，。]|是不是|有没有|对不对")
_DATA_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:%|个|万|亿|倍|元|人|天|年|月|次|项|家)|百分之")


@dataclass
class TextStats:
    char_count: int = 0
    sentences: int = 0
    avg_sentence_len: float = 0.0
    long_sentence_ratio: float = 0.0
    filler_counts: dict = field(default_factory=dict)
    filler_per_100: float = 0.0
    structure_words: list = field(default_factory=list)
    audience_word_count: int = 0
    question_count: int = 0
    data_mentions: int = 0
    repeated_phrases: list = field(default_factory=list)

    def to_notes(self) -> str:
        lines = [
            f"全文约 {self.char_count} 字、{self.sentences} 句，平均句长 {self.avg_sentence_len:.0f} 字，"
            f"超长句(>40字)占比 {self.long_sentence_ratio:.0%}",
        ]
        if self.filler_counts:
            top = "、".join(f"「{w}」{n}次" for w, n in
                           sorted(self.filler_counts.items(), key=lambda x: -x[1])[:6])
            level = "偏多" if self.filler_per_100 >= 2 else ("适中" if self.filler_per_100 >= 0.8 else "较少")
            lines.append(f"语气词/口头禅：每百字 {self.filler_per_100:.1f} 处（{level}），高频：{top}")
        else:
            lines.append("语气词/口头禅：几乎未检测到")
        if self.repeated_phrases:
            lines.append("重复表达：" + "、".join(f"「{p}」{n}次" for p, n in self.repeated_phrases))
        lines.append("结构词：" + ("、".join(self.structure_words) if self.structure_words
                                 else "几乎没有『首先/其次/最后』等结构提示词"))
        lines.append(f"数据/数字表述：{self.data_mentions} 处")
        lines.append(f"听众指向词（大家/各位/您等）：{self.audience_word_count} 次；"
                     f"提问/设问：{self.question_count} 处")
        return "\n".join(lines)


def _repeated_ngrams(text: str, n: int = 4, min_count: int = 4, top: int = 3) -> list:
    """找高频重复的 4 字短语（排除结构词/语气词组合），反映啰嗦或车轱辘话。"""
    clean = re.sub(r"[^\u4e00-\u9fff]", "", text)
    counts = {}
    for i in range(len(clean) - n + 1):
        g = clean[i:i + n]
        counts[g] = counts.get(g, 0) + 1
    skip = ("然后", "就是", "这个", "那个", "我们", "大家", "一个")
    items = [(g, c) for g, c in counts.items()
             if c >= min_count and not any(s in g for s in skip)]
    items.sort(key=lambda x: -x[1])
    picked = []
    for g, c in items:
        if all(g[:3] not in p and g[1:] not in p for p, _ in picked):
            picked.append((g, c))
        if len(picked) >= top:
            break
    return picked


def analyze_text(asr: ASRResult) -> TextStats:
    st = TextStats(char_count=asr.char_count)
    text = asr.full_text
    sents = [s for s in re.split(r"[。！？!?；;]", text) if s.strip()]
    st.sentences = len(sents)
    if sents:
        lens = [len(re.sub(r"[，、,\s]", "", s)) for s in sents]
        st.avg_sentence_len = sum(lens) / len(lens)
        st.long_sentence_ratio = sum(1 for x in lens if x > 40) / len(lens)
    st.filler_counts = dict(asr.filler_counts)
    st.filler_per_100 = sum(st.filler_counts.values()) / max(1, asr.char_count) * 100
    st.structure_words = [w for w in _STRUCTURE_WORDS if w in text]
    st.audience_word_count = sum(text.count(w) for w in _AUDIENCE_WORDS)
    st.question_count = len(_QUESTION_RE.findall(text))
    st.data_mentions = len(_DATA_RE.findall(text))
    st.repeated_phrases = _repeated_ngrams(text)
    return st
