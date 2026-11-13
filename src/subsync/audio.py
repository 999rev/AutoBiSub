import json
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

LANG_ALIASES: dict[str, set[str]] = {
    "en": {"en", "eng", "english"},
    "zh": {"zh", "zho", "chi", "cmn", "chinese"},
    "ja": {"ja", "jpn", "japanese"},
    "it": {"it", "ita", "italian"},
    "ko": {"ko", "kor", "korean"},
    "fr": {"fr", "fra", "fre", "french"},
    "de": {"de", "deu", "ger", "german"},
    "es": {"es", "spa", "spanish"},
}


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg not found in PATH."""


class AudioExtractionError(RuntimeError):
    """FFmpeg audio extraction failed."""


def ensure_ffmpeg_installed() -> str:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise FFmpegNotFoundError("ffmpeg not found in PATH.")
    return ffmpeg_bin


def detect_audio_stream_index(media_path: Path, target_lang: str | None) -> int:
    if not target_lang:
        return 0
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return 0

    cmd = [
        ffprobe_bin,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "a",
        str(media_path),
    ]
    res = subprocess.run(
        cmd,
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if res.returncode != 0:
        return 0

    try:
        streams = json.loads(res.stdout).get("streams", [])
    except json.JSONDecodeError:
        return 0

    lang_key = target_lang.lower().strip()
    accepted = LANG_ALIASES.get(lang_key, {lang_key})
    for idx, stream in enumerate(streams):
        tag_lang = str(stream.get("tags", {}).get("language", "")).lower().strip()
        if tag_lang in accepted:
            logger.info("Matched audio stream #%d (lang=%s)", idx, tag_lang)
            return idx

    return 0


def extract_audio(
    media_path: Path,
    output_wav: Path,
    sample_rate: int = 16000,
    audio_stream_index: int = 0,
) -> Path:
    if not media_path.exists():
        raise FileNotFoundError(f"Media not found: {media_path}")

    ffmpeg_bin = ensure_ffmpeg_installed()
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(media_path),
        "-map",
        f"0:a:{audio_stream_index}",
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
        capture_output=True,
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise AudioExtractionError(f"FFmpeg failed:\n{result.stderr[-500:]}")

    return output_wav


@contextmanager
def temporary_wav(
    media_path: Path,
    sample_rate: int = 16000,
    audio_stream_index: int | None = None,
    language: str | None = None,
) -> Iterator[Path]:
    resolved_idx = (
        audio_stream_index
        if audio_stream_index is not None
        else detect_audio_stream_index(media_path, language)
    )
    with tempfile.TemporaryDirectory(prefix="subsync_") as tmp_dir:
        wav_path = Path(tmp_dir) / f"{media_path.stem}.wav"
        yield extract_audio(
            media_path,
            wav_path,
            sample_rate=sample_rate,
            audio_stream_index=resolved_idx,
        )
