import re
from pathlib import Path
from typing import Literal

from subsync.models import SubtitleSegment

SRT_TIMESTAMP_PATTERN = re.compile(
    r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
)


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为标准 SRT 时间戳 HH:MM:SS,mmm"""
    if seconds < 0:
        raise ValueError(f"时间戳不能为负数: {seconds}")

    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    sec = total_s % 60
    total_m = total_s // 60
    minute = total_m % 60
    hour = total_m // 60
    return f"{hour:02d}:{minute:02d}:{sec:02d},{ms:03d}"


def parse_timestamp(h: str, m: str, s: str, ms: str) -> float:
    """将 SRT 时间戳组件转换为秒数"""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def validate_segments(segments: list[SubtitleSegment]) -> None:
    """校验字幕时间轴与序号的合法性"""
    if not segments:
        raise ValueError("字幕列表为空，无法生成文件")

    seen_indices: set[int] = set()
    for seg in segments:
        if seg.index in seen_indices:
            raise ValueError(f"发现重复的字幕序号: {seg.index}")
        seen_indices.add(seg.index)

        if seg.end < seg.start:
            raise ValueError(
                f"第 {seg.index} 条字幕时间轴倒挂: {seg.start} > {seg.end}"
            )


def render_bilingual_srt(
    segments: list[SubtitleSegment],
    order: Literal["translated_first", "original_first"] = "translated_first",
) -> str:
    """将字幕片段渲染为双语 SRT 字符串"""
    validate_segments(segments)

    blocks: list[str] = []
    for idx, seg in enumerate(segments, start=1):
        time_line = f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}"
        original = seg.text.strip()
        translated = (seg.translated or "").strip()

        if translated and translated != original:
            text_lines = (
                f"{translated}\n{original}"
                if order == "translated_first"
                else f"{original}\n{translated}"
            )
        else:
            text_lines = original

        blocks.append(f"{idx}\n{time_line}\n{text_lines}")

    return "\n\n".join(blocks) + "\n"


def write_srt(
    segments: list[SubtitleSegment],
    output_path: Path,
    order: Literal["translated_first", "original_first"] = "translated_first",
    encoding: str = "utf-8",
) -> Path:
    """校验并写入双语 SRT 文件"""
    content = render_bilingual_srt(segments, order=order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding=encoding)
    return output_path


def parse_srt(srt_path: Path) -> list[SubtitleSegment]:
    """解析本地 .srt 文件为 SubtitleSegment 列表（支持纯字幕翻译模式）"""
    raw = srt_path.read_text(encoding="utf-8-sig")
    raw_blocks = re.split(r"\r?\n\r?\n+", raw.strip())

    segments: list[SubtitleSegment] = []
    for fallback_idx, block in enumerate(raw_blocks, start=1):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        ts_line_idx = 1 if lines[0].isdigit() else 0
        match = SRT_TIMESTAMP_PATTERN.search(lines[ts_line_idx])
        if not match:
            continue

        idx = int(lines[0]) if lines[0].isdigit() else fallback_idx
        start = parse_timestamp(*match.groups()[:4])
        end = parse_timestamp(*match.groups()[4:])
        text = " ".join(lines[ts_line_idx + 1 :])

        if text:
            segments.append(
                SubtitleSegment(index=idx, start=start, end=end, text=text)
            )

    return segments
