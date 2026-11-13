import logging
from dataclasses import dataclass
from pathlib import Path

from faster_whisper import WhisperModel

from subsync.models import SubtitleSegment

logger = logging.getLogger(__name__)


@dataclass
class ASRConfig:
    model_size_or_path: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    beam_size: int = 5
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    initial_prompt: str = "Uncensored casual American comedy dialogue with slang and profanity."


class LocalWhisperASR:
    def __init__(self, config: ASRConfig) -> None:
        self.cfg = config
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        """懒加载模型，避免 CLI 初始化阶段耗时"""
        if self._model is None:
            logger.info(
                "正在加载本地 Whisper 模型: %s (device=%s, compute_type=%s)",
                self.cfg.model_size_or_path,
                self.cfg.device,
                self.cfg.compute_type,
            )
            self._model = WhisperModel(
                self.cfg.model_size_or_path,
                device=self.cfg.device,
                compute_type=self.cfg.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> list[SubtitleSegment]:
        """执行本地转写并过滤空白片段"""
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        raw_segments, info = self.model.transcribe(
            str(audio_path),
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            initial_prompt=self.cfg.initial_prompt,
            word_timestamps=True,
            vad_filter=self.cfg.vad_filter,
            vad_parameters={"min_silence_duration_ms": self.cfg.min_silence_duration_ms, "speech_pad_ms": 150},
        )
        logger.info(
            "检测到语种: %s (置信度 %.2f)",
            info.language,
            info.language_probability,
        )

        segments: list[SubtitleSegment] = []
        idx = 1
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            segments.append(
                SubtitleSegment(
                    index=idx,
                    start=round(seg.words[0].start if getattr(seg, "words", None) else seg.start, 3),
                    end=round(seg.words[-1].end if getattr(seg, "words", None) else seg.end, 3),
                    text=text,
                )
            )
            idx += 1

        return segments
