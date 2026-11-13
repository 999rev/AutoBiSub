from pathlib import Path

import pytest

from subsync.models import SubtitleSegment
from subsync.subtitle import (
    format_timestamp,
    parse_srt,
    render_bilingual_srt,
    validate_segments,
    write_srt,
)


def test_format_timestamp_rounding_boundary() -> None:
    assert format_timestamp(0.0) == "00:00:00,000"
    assert format_timestamp(65.5) == "00:01:05,500"
    assert format_timestamp(1.9999) == "00:00:02,000"
    assert format_timestamp(3661.005) == "01:01:01,005"


def test_format_timestamp_negative_raises() -> None:
    with pytest.raises(ValueError, match="不能为负数"):
        format_timestamp(-0.1)


def test_validate_segments_errors() -> None:
    with pytest.raises(ValueError, match="为空"):
        validate_segments([])

    with pytest.raises(ValueError, match="重复"):
        validate_segments(
            [
                SubtitleSegment(index=1, start=0.0, end=1.0, text="a"),
                SubtitleSegment(index=1, start=1.0, end=2.0, text="b"),
            ]
        )

    with pytest.raises(ValueError, match="倒挂"):
        validate_segments([SubtitleSegment(index=1, start=5.0, end=2.0, text="bad")])


def test_render_bilingual_srt_order_and_fallback() -> None:
    segments = [
        SubtitleSegment(index=1, start=0.0, end=2.0, text="Hello", translated="你好"),
        SubtitleSegment(index=2, start=2.5, end=4.0, text="OK", translated=""),
    ]

    translated_first = render_bilingual_srt(segments, order="translated_first")
    assert "1\n00:00:00,000 --> 00:00:02,000\n你好\nHello" in translated_first
    assert "2\n00:00:02,500 --> 00:00:04,000\nOK\n" in translated_first

    original_first = render_bilingual_srt(segments, order="original_first")
    assert "1\n00:00:00,000 --> 00:00:02,000\nHello\n你好" in original_first


def test_write_and_parse_srt_roundtrip(tmp_path: Path) -> None:
    srt_file = tmp_path / "sample.srt"
    segments = [
        SubtitleSegment(index=1, start=1.2, end=3.4, text="First line."),
        SubtitleSegment(index=2, start=4.0, end=6.5, text="Second line."),
    ]
    write_srt(segments, srt_file)

    parsed = parse_srt(srt_file)
    assert len(parsed) == 2
    assert parsed[0].text == "First line."
    assert parsed[1].start == pytest.approx(4.0)
    assert parsed[1].end == pytest.approx(6.5)
