"""安全合规：指令与转写文本的违规检测。
两级：
  1. 规则预过滤（零延迟）：显式索取有害内容的指令 / 转写文本高危组合
  2. LLM 分类（可关）：对指令 + 转写文本做一次短输出判断
拦截时输出固定的纯文本提示，不输出任何有害内容。
规则只匹配"索取/宣扬/教唆"，避免误杀反诈、禁毒、反邪教等正常主题演讲。
"""
import re
from dataclasses import dataclass

from app.config import settings

_ASK = r"(帮我|替我|教我|告诉我|如何|怎么|怎样|写|生成|编|给我|提供|宣传|鼓吹|号召)"
_INSTRUCTION_BLOCK_PATTERNS = [
    _ASK + r".{0,12}(制作|制造|合成|配方|购买).{0,8}(炸弹|爆炸物|毒品|冰毒|枪支|生化武器)",
    _ASK + r".{0,10}(自杀|自残|杀人|投毒|绑架|勒索|洗钱|诈骗他人|搞传销)",
    _ASK + r".{0,10}(色情|淫秽|黄色段子|裸聊|约炮|性交易)",
    _ASK + r".{0,10}(推翻政府|颠覆政权|分裂国家|恐怖袭击|加入极端组织|加入邪教)",
    _ASK + r".{0,10}(入侵|黑进|破解).{0,8}(系统|账号|密码|银行|政府)",
    _ASK + r".{0,15}(仇恨|歧视|辱骂|诽谤|人身攻击).{0,6}(言论|演讲|稿|内容)",
]

# 转写文本高危词（命中多项才拦截，避免演讲中正常提及"诈骗防范"被误杀）
_TRANSCRIPT_RISK_PATTERNS = [
    (r"(炸弹|爆炸物|冰毒|海洛因|制毒|枪支弹药).{0,20}(制作|配方|购买|渠道|步骤)", 3),
    (r"(杀|弄死|干掉|灭了).{0,6}(他|她|你们|全家|所有人)", 2),
    (r"(恐怖袭击|圣战|极端组织|邪教组织).{0,15}(加入|招募|号召|组织)", 3),
    (r"(颠覆|推翻).{0,6}(政权|国家|政府)", 3),
    (r"(裸照|性交易|淫秽视频|约炮)", 2),
    (r"(传销|庞氏|拉人头).{0,15}(加入|投资|返利|下线)", 2),
    (r"(黑客|木马|钓鱼).{0,10}(教程|工具|方法)", 2),
    (r"(自杀|自残).{0,10}(方法|步骤|怎么|如何)", 3),
]
_TRANSCRIPT_BLOCK_SCORE = 3

REFUSAL_INSTRUCTION = (
    "抱歉，您的指令包含不符合内容安全规范的要求，我无法处理。\n"
    "我是演讲陪练助手，可以帮您分析演讲音频的内容思路、语言表达、节奏状态和听众联结，"
    "并给出改进建议。请换一个与演讲分析相关的指令再试。"
)

REFUSAL_TRANSCRIPT = (
    "抱歉，检测到该音频内容涉及违法违规或有害信息，无法进行演讲分析。\n"
    "演讲陪练助手仅支持对合法合规的演讲、汇报、答辩等内容提供表达优化建议。"
)

SAFETY_SYSTEM_PROMPT = """你是内容安全审核员。判断下面的【用户指令】和【演讲转写文本】是否包含以下任一类违规：
A. 违法犯罪：毒品/武器/爆炸物制作、黑客攻击、诈骗传销、洗钱等具体方法或教唆
B. 暴力恐怖：恐怖主义、极端组织宣传、煽动暴力、威胁伤害他人
C. 政治敏感：颠覆国家政权、分裂国家、煽动颠覆
D. 色情低俗：露骨性描写、性交易
E. 仇恨歧视：基于民族/宗教/性别/地域的仇恨言论或侮辱
F. 自伤：自杀自残的具体方法或鼓励
G. 恶意利用：借演讲分析之名索取上述内容

注意：演讲中正常提及"反诈宣传""禁毒教育""历史战争"等不算违规；只有宣扬、教唆、提供方法才算。

只输出一个词：SAFE 或 UNSAFE"""


@dataclass
class SafetyVerdict:
    allowed: bool
    message: str = ""          # 拦截时返回给用户的纯文本


_DEFENSIVE = re.compile(r"防止|预防|避免|远离|识别|防范|拒绝|反对|打击|禁止|揭露|警惕")


def check_instruction(instruction: str) -> SafetyVerdict:
    """规则层：指令预过滤。命中片段含防范类词语（如"如何预防自杀"）时放行。"""
    text = (instruction or "").strip()
    for pat in _INSTRUCTION_BLOCK_PATTERNS:
        m = re.search(pat, text, re.I)
        if m and not _DEFENSIVE.search(m.group(0)):
            return SafetyVerdict(allowed=False, message=REFUSAL_INSTRUCTION)
    return SafetyVerdict(allowed=True)


def check_transcript_rules(transcript: str) -> SafetyVerdict:
    """规则层：转写文本风险打分。"""
    score = 0
    for pat, w in _TRANSCRIPT_RISK_PATTERNS:
        if re.search(pat, transcript):
            score += w
            if score >= _TRANSCRIPT_BLOCK_SCORE:
                return SafetyVerdict(allowed=False, message=REFUSAL_TRANSCRIPT)
    return SafetyVerdict(allowed=True)


def build_safety_prompt(instruction: str, transcript: str) -> str:
    n = settings.safety_check_chars
    t = transcript if len(transcript) <= n else transcript[: n // 2] + "\n……\n" + transcript[-n // 2:]
    return f"【用户指令】\n{instruction or '（无）'}\n\n【演讲转写文本】\n{t}\n\n判断结果："


def parse_safety_response(resp: str) -> bool:
    """LLM 审核结果 → allowed。无法解析时放行（规则层已兜底）。"""
    r = (resp or "").strip().upper()
    if r.startswith("UNSAFE"):
        return False
    return True
