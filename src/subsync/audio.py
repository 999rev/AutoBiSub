import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class FFmpegNotFoundError(RuntimeError):
    """系统未安装 FFmpeg 或未加入 PATH"""


class AudioExtractionError(RuntimeError):
    """FFmpeg 抽取音轨失败"""


def ensure_ffmpeg_installed() -> str:
    """检查系统环境中是否存在可用的 ffmpeg 可执行文件"""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FFmpegNotFoundError(
            "未检测到 ffmpeg，请先安装并将其加入系统 PATH（macOS: brew install ffmpeg, Ubuntu: apt install ffmpeg）。"
        )
    return ffmpeg_bin


def extract_audio(
    media_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
) -> Path:
    """从视频/音频文件中提取 16kHz 单声道 WAV 音轨"""
    if not media_path.exists():
        raise FileNotFoundError(f"媒体文件不存在: {media_path}")

    ffmpeg_bin = ensure_ffmpeg_installed()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        str(output_wav),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise AudioExtractionError(f"FFmpeg 抽取音轨失败:\n{result.stderr[-500:]}")

    return output_wav


@contextmanager
def temporary_wav(media_path: Path, sample_rate: int = 16000) -> Iterator[Path]:
    """提取临时 WAV 并在上下文退出时自动清理"""
    with tempfile.TemporaryDirectory(prefix="subsync_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{media_path.stem}.wav"
        yield extract_audio(media_path, wav_path, sample_rate=sample_rate)
