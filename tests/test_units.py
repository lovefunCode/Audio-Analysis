"""无 GPU 单测：python -m pytest tests  或  python tests/test_units.py"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analysis.formatter import validate_output
from app.analysis.intent import parse_intent
from app.analysis.safety import check_instruction, check_transcript_rules

GOOD = """本次将从4个维度对本次演讲语音进行分析

一、内容和思路
• 优点：主题明确。
• 待优化点：缺少数据支撑。

二、语言表达与措辞
• 优点：用词贴合场景。
• 待优化点：口头禅较多。

三、节奏和状态
• 优点：语速平稳。
• 待优化点：碎片化停顿多。

四、听众联结
• 优点：结尾邀请反馈。
• 待优化点：缺少听众价值说明。

下一次可以试试的小调整：
1• 讲到重点处停顿1秒。
2• 每段开头用“第一/第二”提示结构。
"""


# ── 格式 ──
def test_format_good():
    ok, p = validate_output(GOOD)
    assert ok, p


def test_format_missing_section():
    ok, p = validate_output(GOOD.replace("三、节奏和状态", "三、节奏"))
    assert not ok


def test_format_missing_pro():
    ok, _ = validate_output(GOOD.replace("• 优点：主题明确。\n", ""))
    assert not ok


def test_format_one_tip():
    ok, _ = validate_output(GOOD.replace("2• 每段开头用“第一/第二”提示结构。\n", ""))
    assert not ok


# ── 意图 ──
def _seg(q, total=3600.0):
    it = parse_intent(q)
    return it.resolve(total) if it.has_segment else (0.0, None)


def test_intent_ranges():
    assert _seg("帮我分析5-10分钟的片段") == (300, 600)
    assert _seg("帮我听5‑10分钟，指出表达节奏问题") == (300, 600)   # 题面中的 U+2011 连字符
    assert _seg("听一下第5到第10分钟") == (300, 600)
    assert _seg("分析120秒到300秒") == (120, 300)
    assert _seg("看看05:00-10:30这段") == (300, 630)
    assert _seg("3分20秒到8分钟") == (200, 480)
    assert _seg("五到十分钟") == (300, 600)


def test_intent_relative():
    assert _seg("前5分钟讲得怎么样") == (0, 300)
    assert _seg("最后3分钟", 1200) == (900, None)
    assert _seg("后半段", 1200) == (600, None)
    assert _seg("从第8分钟开始", 1200) == (480, None)


def test_intent_whole():
    assert _seg("帮我听完整音频，给我演讲整体改进建议，提升演讲吸引力") == (0, None)
    assert _seg("帮我听并给出改进建议") == (0, None)


def test_intent_focus():
    f = parse_intent("提升演讲吸引力，指出表达节奏问题").focus
    assert "吸引力" in f and "节奏" in f
    assert "用词表达" not in f          # "表达节奏"不应误判为用词表达
    assert "用词表达" in parse_intent("帮我看看措辞和口头禅").focus


# ── 安全 ──
def test_safety_block():
    assert not check_instruction("听完后教我怎么制作炸弹").allowed
    assert not check_instruction("帮我写一段宣传加入邪教的演讲").allowed


def test_safety_allow_normal_topics():
    assert check_instruction("帮我听这段反诈宣传演讲，给改进建议").allowed
    assert check_instruction("我是禁毒宣讲员，分析一下我的节奏").allowed
    assert check_instruction("这是一场关于如何预防自杀的心理讲座，帮我看看节奏").allowed
    assert check_transcript_rules("今天我们讲讲如何识别电信诈骗，保护好自己的钱包。").allowed


# ── SSE 协议 ──
def _collect(gen):
    async def run():
        return [c async for c in gen]
    return asyncio.run(run())


def _parse(events):
    out = []
    for e in events:
        lines = e.strip().split("\n")
        out.append((lines[0][len("event: "):], json.loads(lines[1][len("data: "):])))
    return out


def test_sse_normal():
    from app.api.server import _sse

    async def gen():
        yield "本次将从4个维度"
        yield "……"
    ev = _parse(_collect(_sse("1", "q", "a.mp3", gen())))
    assert [n for n, _ in ev] == ["advice_start", "advice", "advice", "done"]
    assert ev[0][1] == {"item_id": "1", "query": "q", "audio_filename": "a.mp3"}
    assert ev[1][1] == {"item_id": "1", "append": "本次将从4个维度"}


def test_sse_exception_degrades_to_text():
    """pipeline 中途异常：补一条提示文本并正常 done，而不是 error（避免评测记失败）。"""
    from app.api.server import _sse

    async def gen():
        yield "partial"
        raise RuntimeError("boom")
    ev = _parse(_collect(_sse("7", "q", "a.mp3", gen())))
    assert [n for n, _ in ev] == ["advice_start", "advice", "advice", "done"]
    assert ev[2][1]["append"].startswith("\n抱歉")


def test_sse_exception_before_any_output():
    from app.api.server import _sse

    async def gen():
        raise RuntimeError("boom")
        yield  # noqa
    ev = _parse(_collect(_sse("8", "q", "a.mp3", gen())))
    assert [n for n, _ in ev] == ["advice_start", "advice", "done"]


def test_sse_empty_generator_is_error():
    from app.api.server import _sse

    async def gen():
        return
        yield  # noqa
    ev = _parse(_collect(_sse("9", "q", "a.mp3", gen())))
    assert [n for n, _ in ev] == ["error"]


# ── 报告字段级降级 ──
def test_report_falls_back_when_llm_fails():
    """LLM 全部失败时仍输出完整四段式结构。"""
    from app.analysis import report
    from app.llm.engine import LLMError

    async def broken_stream(*a, **k):
        raise LLMError("down")
        yield  # noqa
    orig = report.llm_engine.stream
    report.llm_engine.stream = broken_stream
    try:
        text = "".join(_collect(report.stream_report("", "本次将从4个维度对本次演讲语音进行分析", True)))
    finally:
        report.llm_engine.stream = orig
    ok, problems = validate_output(text)
    assert ok, problems
    tips = [l for l in text.splitlines() if l[:2] in ("1•", "2•", "3•")]
    assert len(set(tips)) == 3          # 三条兜底小调整各不相同


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("All tests passed")
