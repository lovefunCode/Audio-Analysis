"""意图识别：从用户指令解析分析片段 + 关注重点。
支持的时间表达：
  5-10分钟 / 5到10分钟 / 第5分钟到第10分钟 / 3分20秒到8分钟 / 05:00-10:30 / 1:02:00-1:10:00
  前5分钟 / 开头3分钟 / 最后5分钟 / 后半段 / 前半段 / 从第8分钟开始 / 第8分钟以后
  120秒到300秒 / 整个音频 / 全程 / 完整
"""
import re
from dataclasses import dataclass, field
from typing import Optional

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
           "七": 7, "八": 8, "九": 9, "十": 10, "半": 0.5}


def _cn_to_num(s: str) -> Optional[float]:
    """中文数字 → 数值，支持 一~九十九、半。"""
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    if s == "半":
        return 0.5
    if "十" in s:
        a, _, b = s.partition("十")
        tens = _CN_NUM.get(a, 1) if a else 1
        ones = _CN_NUM.get(b, 0) if b else 0
        return float(tens * 10 + ones)
    if len(s) == 1 and s in _CN_NUM:
        return float(_CN_NUM[s])
    return None


_NUM = r"(\d+(?:\.\d+)?|[零一二两三四五六七八九十]{1,3}|半)"
# 一个时间点：可选"第"，可选"X小时"，"X分(钟)"，"X秒"；至少一项
_POINT = (rf"第?\s*(?:{_NUM}\s*(?:个)?\s*(?:小时|h))?\s*"
          rf"(?:{_NUM}\s*(?:分钟|分|min|m(?![a-z])))?\s*"
          rf"(?:{_NUM}\s*(?:秒钟|秒|s(?![a-z])))?")
_CLOCK = r"(\d{1,2}):(\d{2})(?::(\d{2}))?"
_SEP = r"\s*(?:-|‐|‑|‒|–|—|―|~|～|〜|到|至|到第|至第)\s*"


def _point_to_sec(h, m, s) -> Optional[float]:
    vals = [_cn_to_num(x) if x else None for x in (h, m, s)]
    if all(v is None for v in vals):
        return None
    h, m, s = (v or 0.0 for v in vals)
    return h * 3600 + m * 60 + s


def _clock_to_sec(a, b, c) -> float:
    if c:   # h:mm:ss
        return int(a) * 3600 + int(b) * 60 + int(c)
    return int(a) * 60 + int(b)   # mm:ss


@dataclass
class Intent:
    start_s: float = 0.0
    end_s: Optional[float] = None         # None = 到结尾
    relative: str = ""                     # "head"/"tail"/"first_half"/"second_half"/"from"
    rel_value_s: float = 0.0
    segment_text: str = ""                 # 原文命中的时间描述，回显给 LLM
    has_segment: bool = False
    focus: list = field(default_factory=list)   # 用户关注点
    raw: str = ""

    def resolve(self, total_s: float) -> tuple[float, Optional[float]]:
        """结合音频总时长解析相对时间段。"""
        if self.relative == "head":
            return 0.0, min(self.rel_value_s, total_s) if total_s else self.rel_value_s
        if self.relative == "tail":
            return max(0.0, total_s - self.rel_value_s), None
        if self.relative == "first_half":
            return 0.0, total_s / 2 if total_s else None
        if self.relative == "second_half":
            return total_s / 2 if total_s else 0.0, None
        if self.relative == "from":
            return self.rel_value_s, None
        return self.start_s, self.end_s


_FOCUS_RULES = [
    ("节奏", r"节奏|语速|停顿|快慢|卡顿|拖沓|急"),
    ("吸引力", r"吸引|感染力|生动|有趣|抓人|打动|精彩"),
    ("逻辑结构", r"逻辑|结构|思路|条理|层次|框架"),
    ("用词表达", r"用词|措辞|口头禅|语气词|啰嗦|表达(?!节奏)|口语"),
    ("听众互动", r"听众|观众|互动|评委|共鸣|联结|领导"),
    ("状态自信", r"紧张|自信|状态|气场|情绪|音量|声音"),
    ("答辩场景", r"答辩|论文|开题|评审"),
    ("汇报场景", r"汇报|述职|晋升|周会|项目|复盘"),
]


def parse_intent(instruction: str) -> Intent:
    text = (instruction or "").strip()
    it = Intent(raw=text)
    it.focus = [name for name, pat in _FOCUS_RULES if re.search(pat, text)]
    if not text:
        return it

    # 1. 时钟格式 05:00-10:30
    m = re.search(_CLOCK + _SEP + _CLOCK, text)
    if m:
        a = _clock_to_sec(*m.group(1, 2, 3))
        b = _clock_to_sec(*m.group(4, 5, 6))
        if b > a:
            it.start_s, it.end_s, it.has_segment = a, b, True
            it.segment_text = m.group(0)
            return it

    # 2a. 共享单位区间 "5-10分钟" / "第5到第10分钟" / "120到300秒"
    m = re.search(rf"第?\s*{_NUM}{_SEP}第?\s*{_NUM}\s*(小时|分钟|分|秒钟|秒|min)", text)
    if m:
        unit = {"小时": 3600, "秒": 1, "秒钟": 1}.get(m.group(3), 60)
        a, b = _cn_to_num(m.group(1)), _cn_to_num(m.group(2))
        if a is not None and b is not None and b > a:
            it.start_s, it.end_s, it.has_segment = a * unit, b * unit, True
            it.segment_text = m.group(0).strip()
            return it

    # 2b. 各自带单位 "3分20秒到8分钟" / "第5分钟到第10分30秒"
    for m in re.finditer(_POINT + _SEP + _POINT, text):
        p1 = _point_to_sec(*m.group(1, 2, 3))
        p2 = _point_to_sec(*m.group(4, 5, 6))
        if p1 is not None and p2 is not None and p2 > p1:
            it.start_s, it.end_s, it.has_segment = p1, p2, True
            it.segment_text = m.group(0).strip()
            return it

    # 3. 相对片段
    m = re.search(rf"(?:前|开头|开始的?|最开始)\s*{_NUM}\s*(分钟|分|秒)", text)
    if m:
        v = _cn_to_num(m.group(1)) or 0
        it.relative, it.rel_value_s = "head", v * (60 if "分" in m.group(2) else 1)
        it.has_segment, it.segment_text = True, m.group(0)
        return it
    m = re.search(rf"(?:最后|结尾|末尾|后)\s*{_NUM}\s*(分钟|分|秒)", text)
    if m:
        v = _cn_to_num(m.group(1)) or 0
        it.relative, it.rel_value_s = "tail", v * (60 if "分" in m.group(2) else 1)
        it.has_segment, it.segment_text = True, m.group(0)
        return it
    m = re.search(rf"(?:从)?第?\s*{_NUM}\s*(分钟|分|秒)\s*(?:开始|以后|之后|往后|起)", text)
    if m:
        v = _cn_to_num(m.group(1)) or 0
        it.relative, it.rel_value_s = "from", v * (60 if "分" in m.group(2) else 1)
        it.has_segment, it.segment_text = True, m.group(0)
        return it
    if re.search(r"前半(段|部分)", text):
        it.relative, it.has_segment, it.segment_text = "first_half", True, "前半段"
        return it
    if re.search(r"后半(段|部分)", text):
        it.relative, it.has_segment, it.segment_text = "second_half", True, "后半段"
        return it

    # 4. 整体
    if re.search(r"整[个段篇]|全程|全部|完整|全文|从头到尾", text):
        it.segment_text = "整个音频"
    return it
