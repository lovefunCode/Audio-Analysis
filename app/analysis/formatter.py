"""输出格式常量与整体校验（报告结构由 report.py 服务端控制，这里用于测试与监控）。"""
import re

SECTIONS = ["一、内容和思路", "二、语言表达与措辞", "三、节奏和状态", "四、听众联结"]
TIPS_TITLE = "下一次可以试试的小调整："


def validate_output(text: str) -> tuple[bool, list]:
    problems = []
    pos = [text.find(s) for s in SECTIONS]
    for sec, p in zip(SECTIONS, pos):
        if p < 0:
            problems.append(f"缺少章节:{sec}")
    present = [p for p in pos if p >= 0]
    if present != sorted(present):
        problems.append("章节顺序错误")
    tips_pos = text.find(TIPS_TITLE)
    for i, (sec, p) in enumerate(zip(SECTIONS, pos)):
        if p < 0:
            continue
        nxt = [q for q in pos[i + 1:] if q > p] + ([tips_pos] if tips_pos > p else []) + [len(text)]
        body = text[p:min(nxt)]
        if "• 优点：" not in body:
            problems.append(f"{sec}缺少优点")
        if "• 待优化点：" not in body:
            problems.append(f"{sec}缺少待优化点")
    if tips_pos < 0:
        problems.append("缺少小调整")
    else:
        n = len(re.findall(r"^\d• \S", text[tips_pos:], re.M))
        if not 2 <= n <= 3:
            problems.append(f"小调整{n}条，应为2-3条")
    return not problems, problems
