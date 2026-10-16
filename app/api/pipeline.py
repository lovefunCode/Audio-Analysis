"""流水线：安全 → 意图 → 时长校验 → 解码 → ASR → 统计 → 安全复核 → 结构化流式报告。
所有用户可见的异常都以纯文本提示输出（不抛 HTTP 错误），保证评测"有输出"。
"""
import asyncio
import logging
import time
from typing import AsyncIterator, Optional

from app.analysis import safety
from app.analysis.intent import parse_intent
from app.analysis.report import (SYSTEM_PROMPT, build_user_prompt,
                                 opening_line, stream_report)
from app.analysis.text_stats import analyze_text
from app.asr.engine import ASRPipeline, format_timeline
from app.audio.features import compute_features
from app.audio.loader import AudioError, AudioFile, decode, probe_duration
from app.config import settings
from app.llm.engine import LLMError, llm_engine

log = logging.getLogger("coach")


def fmt_dur(sec: float) -> str:
    sec = int(round(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    if h:
        return f"{h}小时{m}分{s}秒"
    return f"{m}分{s}秒" if m else f"{s}秒"


def fmt_clock(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def msg_overlimit(total: float) -> str:
    lim = settings.max_audio_duration_s // 60
    return (f"抱歉，该音频时长约 {fmt_dur(total)}，超过了单次分析 {lim} 分钟的上限，暂时无法完整解析。\n"
            f"建议：\n"
            f"1• 将音频裁剪到 {lim} 分钟以内后重新上传；\n"
            f"2• 或在指令中指定 {lim} 分钟以内的片段，例如「帮我分析第10-20分钟」。")


def msg_segment_too_long(seg_len: float) -> str:
    lim = settings.max_audio_duration_s // 60
    return (f"抱歉，指定的分析片段长度约 {fmt_dur(seg_len)}，超过了单次分析 {lim} 分钟的上限。\n"
            f"请将片段缩短到 {lim} 分钟以内，例如「帮我分析第10-20分钟」。")


def msg_start_beyond(start: float, total: float) -> str:
    return (f"抱歉，指定的起始时间 {fmt_clock(start)} 超出了音频总时长（{fmt_dur(total)}），无法截取该片段。\n"
            f"请指定音频时长范围内的时间段后重试。")


MSG_NO_SPEECH = ("抱歉，所选音频片段中没有识别到足够的有效语音（可能是静音、纯音乐或噪声），无法进行演讲分析。\n"
                 "请确认音频内容或调整分析时间段后重试。")

MSG_ASR_TIMEOUT = ("抱歉，该音频的语音识别耗时过长，本次分析未能完成。\n"
                   "建议：在指令中指定较短的时间段（例如「帮我分析前20分钟」）后重试。")

MSG_ASR_FAILED = ("抱歉，语音识别过程中出现异常，本次分析未能完成。\n"
                  "请确认音频清晰可听，或稍后重试。")


class CoachPipeline:
    def __init__(self):
        self.asr = ASRPipeline()

    def warmup(self):
        self.asr.load()
        llm_engine.load()

    async def run(self, file_bytes: bytes, filename: str, instruction: str,
                  form_start: float = 0.0, form_end: Optional[float] = None) -> AsyncIterator[str]:
        t0 = time.time()
        # 1. 指令安全（规则，零延迟）
        v = safety.check_instruction(instruction)
        if not v.allowed:
            yield v.message
            return

        # 2. 意图：指令中的时间段优先，其次表单参数
        intent = parse_intent(instruction)

        try:
            af = AudioFile(file_bytes, filename)
        except AudioError as e:
            yield str(e)
            return

        with af:
            total = await asyncio.to_thread(probe_duration, af.path)
            if intent.has_segment:
                start, end = intent.resolve(total)
            else:
                start, end = max(0.0, form_start or 0.0), form_end

            # 3. 时长 / 片段合法性
            if total > 0:
                if start >= total - 1:
                    yield msg_start_beyond(start, total)
                    return
                if end is not None and end > total:
                    end = None                      # 超出部分截到结尾
            seg_end = end if end is not None else total
            seg_len = (seg_end - start) if seg_end else 0.0
            limit = settings.max_audio_duration_s + settings.duration_tolerance_s
            is_segment = start > 0 or end is not None
            if is_segment and seg_len > limit:
                yield msg_segment_too_long(seg_len)
                return
            if not is_segment and total > limit:
                yield msg_overlimit(total)
                return

            if is_segment:
                end_txt = fmt_clock(seg_end) if seg_end else "结尾"
                seg_desc = f"{fmt_clock(start)}-{end_txt} 片段"
                if intent.segment_text:
                    seg_desc = f"用户指定「{intent.segment_text}」，即 {seg_desc}"
            else:
                end_txt = ""
                seg_desc = "完整音频"

            opening = opening_line(f"{fmt_clock(start)}-{end_txt}" if is_segment else "",
                                   intent.focus)
            early = settings.early_first_token
            if early:
                yield opening + "\n"
                log.info("first token %.0fms", (time.time() - t0) * 1000)

            # 4. 解码 + ASR（线程池，不阻塞事件循环）
            try:
                audio = await asyncio.to_thread(decode, af.path, start, end)
            except AudioError as e:
                yield ("\n" if early else "") + str(e)
                return

        # 时长探测失败（total=0）时，以实际解码长度兜底校验上限
        if total <= 0 and audio.duration_s > limit:
            yield ("\n" if early else "") + (msg_segment_too_long(audio.duration_s) if is_segment
                                             else msg_overlimit(audio.duration_s))
            return
        if total <= 0:
            total = start + audio.duration_s

        # ASR：线程池 + 超时。超时/异常 → 明确提示，不让请求挂死
        try:
            asr = await asyncio.wait_for(
                asyncio.to_thread(self.asr.transcribe, audio.samples),
                timeout=settings.asr_timeout_s)
        except asyncio.TimeoutError:
            log.error("asr timeout after %ss (%.0fs audio)", settings.asr_timeout_s, audio.duration_s)
            yield ("\n" if early else "") + MSG_ASR_TIMEOUT
            return
        except Exception:
            log.exception("asr failed")
            yield ("\n" if early else "") + MSG_ASR_FAILED
            return
        if asr.char_count < settings.min_valid_chars:
            yield ("\n" if early else "") + MSG_NO_SPEECH
            return
        log.info("asr done %.1fs, %d chars", time.time() - t0, asr.char_count)

        # 5. 转写内容安全：规则 + LLM 复核（LLM 失败时放行，规则层已兜底）
        v2 = safety.check_transcript_rules(asr.full_text)
        if v2.allowed and settings.safety_llm_check:
            try:
                p = llm_engine.chat_prefix(safety.SAFETY_SYSTEM_PROMPT,
                                           safety.build_safety_prompt(instruction, asr.full_text))
                resp = await llm_engine.complete(p, max_tokens=3, temperature=0.0)
                if not safety.parse_safety_response(resp):
                    v2 = safety.SafetyVerdict(allowed=False, message=safety.REFUSAL_TRANSCRIPT)
            except LLMError as ex:
                log.warning("safety llm check skipped: %s", ex)
        if not v2.allowed:
            yield ("\n" if early else "") + v2.message
            return

        # 6. 统计 + 报告（统计失败不影响出报告，只是少一部分客观依据）
        try:
            acoustic_notes = compute_features(audio, asr.char_count, asr.speech_s,
                                              asr.timeline, start).to_llm_notes()
        except Exception:
            log.exception("acoustic features failed")
            acoustic_notes = "（声学测量不可用）"
        try:
            text_notes = analyze_text(asr).to_notes()
        except Exception:
            log.exception("text stats failed")
            text_notes = "（文本统计不可用）"
        user = build_user_prompt(
            instruction=instruction, segment_desc=seg_desc,
            total_desc=fmt_dur(total) if total else "未知",
            focus=intent.focus,
            transcript=format_timeline(asr.timeline, start, settings.transcript_max_chars),
            acoustic=acoustic_notes, text_stats=text_notes)
        prefix = llm_engine.chat_prefix(SYSTEM_PROMPT, user)
        deadline = time.monotonic() + max(60.0, settings.request_soft_deadline_s - (time.time() - t0))
        async for chunk in stream_report(prefix, opening, emit_opening=not early, deadline=deadline):
            yield chunk
        log.info("done %.1fs", time.time() - t0)


pipeline = CoachPipeline()
