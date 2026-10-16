"""声学特征分析：语速、停顿、音量、能量动态 —— 对应维度三「节奏和状态」"""
import re
from dataclasses import dataclass, field

import numpy as np

from app.audio.loader import AudioData


@dataclass
class PaceStats:
    speech_s: float            # 有效语音时长
    char_count: int            # ASR 字数
    chars_per_min: float       # 语速（字/分钟）
    rate_label: str            # 偏慢/适中/偏快


@dataclass
class PauseStats:
    pauses: list = field(default_factory=list)   # [(start_s, end_s, dur_s)]
    total_pause_s: float = 0.0
    long_pauses_3s: int = 0       # ≥3s 长停顿数
    long_pauses_8s: int = 0       # ≥8s 超长停顿（可能忘词/卡壳）
    fragmented_rate: float = 0.0  # <0.8s 碎片停顿占比


@dataclass
class VolumeStats:
    overall_db: float           # 平均响度 dBFS
    std_db: float               # 响度波动
    silent_ratio: float         # 静音占比（<-50dB）
    soft_ratio: float           # 弱音(<-35dB)时长占比
    dynamic_label: str = ""     # 动态单一/起伏适中/起伏丰富


@dataclass
class AcousticReport:
    pace: PaceStats | None = None
    pauses: PauseStats | None = None
    volume: VolumeStats | None = None
    pace_windows: list = field(default_factory=list)   # [(mm:ss, 字/分钟)] 每分钟语速
    long_pause_marks: list = field(default_factory=list)  # 长停顿出现时间点 mm:ss

    def to_llm_notes(self) -> str:
        """转成给 LLM 的客观观测文本（非结论，让 LLM 下判断）。"""
        lines = []
        if self.pace:
            lines.append(
                f"语速：有效语音约 {self.pace.speech_s:.0f} 秒，共 {self.pace.char_count} 字，"
                f"约 {self.pace.chars_per_min:.0f} 字/分钟（{self.pace.rate_label}；中文汇报参考 180-260 字/分钟）"
            )
        if len(self.pace_windows) >= 3:
            vals = [v for _, v in self.pace_windows]
            fast = max(self.pace_windows, key=lambda x: x[1])
            slow = min(self.pace_windows, key=lambda x: x[1])
            cv = float(np.std(vals) / max(1.0, np.mean(vals)))
            stab = "前后语速稳定" if cv < 0.15 else ("语速有一定起伏" if cv < 0.3 else "语速忽快忽慢明显")
            lines.append(f"分段语速：最快在 {fast[0]} 附近约 {fast[1]:.0f} 字/分，"
                         f"最慢在 {slow[0]} 附近约 {slow[1]:.0f} 字/分（{stab}）")
        if self.pauses:
            p = self.pauses
            lines.append(
                f"停顿：共检测到 {len(p.pauses)} 次停顿，总停顿 {p.total_pause_s:.0f} 秒；"
                f"≥3秒长停顿 {p.long_pauses_3s} 次，≥8秒超长停顿 {p.long_pauses_8s} 次；"
                f"碎片化短停顿(<0.8s)占比 {p.fragmented_rate:.0%}（偏高说明'嗯啊'式碎片停顿多）"
            )
            if self.long_pause_marks:
                lines.append("长停顿出现位置：" + "、".join(self.long_pause_marks[:8]))
        if self.volume:
            v = self.volume
            lines.append(
                f"音量：平均响度 {v.overall_db:.0f} dBFS，波动 {v.std_db:.0f} dB，"
                f"静音占比 {v.silent_ratio:.0%}，弱音占比 {v.soft_ratio:.0%}（{v.dynamic_label}）"
            )
        return "\n".join(lines)


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def _pace_windows(timeline: list, offset_s: float, win_s: float = 60.0) -> list:
    """按 60s 窗口统计语速（字/分钟），基于句级时间戳。"""
    if not timeline:
        return []
    end = max(e for _, _, e in timeline)
    out = []
    w = 0.0
    while w < end:
        chars, dur = 0, 0.0
        for text, s, e in timeline:
            ov = min(e, w + win_s) - max(s, w)
            if ov > 0 and e > s:
                ratio = ov / (e - s)
                chars += len(re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)) * ratio
                dur += ov
        if dur >= 20:                      # 窗口内有效语音不足 20s 不计
            out.append((_mmss(w + offset_s), chars / dur * 60))
        w += win_s
    return out


def compute_features(audio: AudioData, asr_chars: int, speech_s_from_asr: float,
                     timeline: list | None = None, offset_s: float = 0.0) -> AcousticReport:
    """入口：audio + ASR 统计 → 声学报告。帧级 RMS + 能量 VAD。"""
    report = AcousticReport()
    report.pace_windows = _pace_windows(timeline or [], offset_s)
    sr = audio.sample_rate
    x = audio.samples

    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, 1 + (len(x) - frame_len) // hop)
    if len(x) < frame_len:
        rms = np.array([float(np.sqrt(np.mean(x * x)))], dtype=np.float32)
    else:
        # 滑窗向量化计算 RMS，避免逐帧 Python 循环
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, frame_len), strides=(hop * x.itemsize, x.itemsize))
        rms = np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)

    rms_db = 20 * np.log10(rms + 1e-10)
    db = np.maximum(rms_db, -80.0)
    # 自适应阈值：噪声底 + 8dB
    noise_floor = np.percentile(db, 20)
    vad_thresh = max(noise_floor + 8.0, -50.0)
    is_speech = db > vad_thresh
    frame_t = hop / sr

    # ── 停顿统计（连续非语音段 ≥0.4s 记为停顿；游程编码向量化）──
    min_pause_frames = int(0.4 / frame_t)
    padded = np.concatenate(([True], is_speech, [True])).astype(np.int8)
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == -1)      # 语音→静音
    ends = np.flatnonzero(diff == 1)         # 静音→语音
    keep = (ends - starts) >= min_pause_frames
    # 去掉首尾静音（录音开头/结尾的空白不算演讲停顿）
    pauses = [(s * frame_t, e * frame_t, (e - s) * frame_t)
              for s, e in zip(starts[keep], ends[keep])
              if s > 0 and e < n_frames]

    ps = PauseStats(pauses=pauses)
    ps.total_pause_s = sum(p[2] for p in pauses)
    ps.long_pauses_3s = sum(1 for p in pauses if p[2] >= 3.0)
    ps.long_pauses_8s = sum(1 for p in pauses if p[2] >= 8.0)
    n_pause = max(1, len(pauses))
    ps.fragmented_rate = sum(1 for p in pauses if p[2] < 0.8) / n_pause
    report.pauses = ps
    report.long_pause_marks = [f"{_mmss(p[0] + offset_s)}（{p[2]:.0f}秒）"
                               for p in pauses if p[2] >= 3.0]

    # ── 音量统计 ──
    speech_db = db[is_speech] if is_speech.any() else db
    vs = VolumeStats(
        overall_db=float(np.mean(speech_db)),
        std_db=float(np.std(speech_db)),
        silent_ratio=float(1.0 - is_speech.mean()),
        soft_ratio=float((speech_db < -35).mean()),
    )
    if vs.std_db < 4:
        vs.dynamic_label = "音量起伏单一，缺乏强弱对比"
    elif vs.std_db < 8:
        vs.dynamic_label = "音量起伏适中"
    else:
        vs.dynamic_label = "音量起伏丰富，强弱对比明显"
    report.volume = vs

    # ── 语速（用 ASR 有效语音时长更准；回退到 VAD 时长）──
    speech_s = speech_s_from_asr if speech_s_from_asr > 1 else float(is_speech.sum() * frame_t)
    if asr_chars > 0 and speech_s > 1:
        cpm = asr_chars / (speech_s / 60.0)
        if cpm < 160:
            label = "偏慢"
        elif cpm <= 260:
            label = "适中"
        else:
            label = "偏快"
        report.pace = PaceStats(
            speech_s=speech_s, char_count=asr_chars,
            chars_per_min=cpm, rate_label=label,
        )
    return report
