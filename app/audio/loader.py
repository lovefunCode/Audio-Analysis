"""音频加载：mp3/m4a → 16k mono float32，支持时长探测与片段截取。

注意：m4a(MP4 容器) 的 moov atom 常在文件末尾，ffmpeg 无法从管道 seek，
因此一律先落盘再用文件路径解码。
"""
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

import numpy as np

from app.config import settings

SUPPORTED_EXTS = (".mp3", ".m4a", ".mp4", ".aac", ".wav", ".flac", ".ogg")


class AudioError(Exception):
    """音频处理异常（message 直接面向用户）"""


@dataclass
class AudioData:
    samples: np.ndarray      # float32 mono
    sample_rate: int
    duration_s: float


class AudioFile:
    """上传音频的临时文件句柄，with 语句结束自动删除。"""

    def __init__(self, file_bytes: bytes, filename: str = ""):
        if not file_bytes:
            raise AudioError("上传的音频文件为空，请重新上传 mp3 或 m4a 格式的演讲音频。")
        ext = os.path.splitext(filename or "")[1].lower() or _sniff_ext(file_bytes)
        if ext not in SUPPORTED_EXTS:
            raise AudioError(f"暂不支持 {ext or '该'} 格式，请上传 mp3 或 m4a 格式的演讲音频。")
        fd, self.path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(file_bytes)
        self.ext = ext

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.path and os.path.exists(self.path):
            try:
                os.remove(self.path)
            except OSError:
                pass
        self.path = ""


def _sniff_ext(b: bytes) -> str:
    """无文件名时按魔数猜格式。"""
    if b[:3] == b"ID3" or (len(b) > 1 and b[0] == 0xFF and (b[1] & 0xE0) == 0xE0):
        return ".mp3"
    if b[4:8] == b"ftyp":
        return ".m4a"
    if b[:4] == b"RIFF":
        return ".wav"
    if b[:4] == b"fLaC":
        return ".flac"
    if b[:4] == b"OggS":
        return ".ogg"
    return ""


def probe_duration(path: str) -> float:
    """ffprobe 读取时长（秒）；失败回退 mutagen；都失败返回 0。"""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path],
            capture_output=True, timeout=60)
        if proc.returncode == 0:
            d = json.loads(proc.stdout or b"{}").get("format", {}).get("duration")
            if d:
                return float(d)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    try:
        from mutagen import File as MFile
        mf = MFile(path)
        if mf is not None and mf.info and mf.info.length:
            return float(mf.info.length)
    except Exception:
        pass
    return 0.0


def decode(path: str, start_s: float = 0.0, end_s: float | None = None) -> AudioData:
    """解码指定片段为 16k mono float32。"""
    offset = max(0.0, start_s)
    duration = (end_s - offset) if end_s is not None else 0.0
    if end_s is not None and duration <= 0:
        raise AudioError("指定的时间段无效（结束时间早于开始时间），请检查后重试。")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if offset > 0:
        cmd += ["-ss", f"{offset:.3f}"]          # 放在 -i 前：快速 seek
    cmd += ["-i", path]
    if duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(settings.target_sr), "-f", "f32le", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=300)
    except FileNotFoundError:
        return _decode_librosa(path, offset, duration)
    except subprocess.TimeoutExpired:
        raise AudioError("音频解码超时，请确认文件未损坏后重试。")
    if proc.returncode != 0:
        raise AudioError("音频文件无法解析，可能已损坏或格式不正确，请上传 mp3 或 m4a 格式的音频。")

    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if samples.size < settings.target_sr:       # 不足 1 秒
        raise AudioError("指定片段内没有可分析的音频内容，请检查时间段是否超出音频时长。")
    return AudioData(samples, settings.target_sr, samples.size / settings.target_sr)


def _decode_librosa(path: str, offset: float, duration: float) -> AudioData:
    import librosa
    try:
        y, sr = librosa.load(path, sr=settings.target_sr, mono=True, offset=offset,
                             duration=duration if duration > 0 else None)
    except Exception:
        raise AudioError("音频文件无法解析，可能已损坏或格式不正确，请上传 mp3 或 m4a 格式的音频。")
    if y.size < sr:
        raise AudioError("指定片段内没有可分析的音频内容，请检查时间段是否超出音频时长。")
    return AudioData(y.astype(np.float32), sr, y.size / sr)
