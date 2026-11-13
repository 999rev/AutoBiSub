import json
from unittest.mock import AsyncMock, patch

import httpx
import respx

from subsync.models import SubtitleSegment
from subsync.translator import AsyncSubtitleTranslator, TranslatorConfig


def _mock_openai_response(items: list[dict]) -> httpx.Response:
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"items": items}, ensure_ascii=False),
                }
            }
        ]
    }
    return httpx.Response(200, json=body)


async def test_translate_batches_with_context_and_code_fences() -> None:
    segments = [
        SubtitleSegment(index=1, start=0.0, end=1.0, text="Line 1"),
        SubtitleSegment(index=2, start=1.0, end=2.0, text="Line 2"),
        SubtitleSegment(index=3, start=2.0, end=3.0, text="Line 3"),
    ]
    cfg = TranslatorConfig(
        api_key="test-key",
        batch_size=2,
        context_size=1,
        max_concurrency=2,
    )
    translator = AsyncSubtitleTranslator(cfg)

    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        user_msg = json.loads(payload["messages"][1]["content"])
        captured_payloads.append(user_msg)

        if user_msg["subtitles"][0]["index"] == 1:
            fenced = '```json\n{"items": [{"index": 1, "translated": "第一句"}, {"index": 2, "translated": "第二句"}]}\n```'
            return httpx.Response(
                200, json={"choices": [{"message": {"content": fenced}}]}
            )
        return _mock_openai_response([{"index": 3, "translated": "第三句"}])

    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=handler)
        result = await translator.translate_segments(segments)

    assert [s.translated for s in result] == ["第一句", "第二句", "第三句"]
    assert captured_payloads[1]["context"] == [{"index": 2, "text": "Line 2"}]


async def test_translate_missing_item_triggers_single_fallback() -> None:
    segments = [
        SubtitleSegment(index=1, start=0.0, end=1.0, text="Hello"),
        SubtitleSegment(index=2, start=1.0, end=2.0, text="World"),
    ]
    cfg = TranslatorConfig(api_key="test-key", batch_size=2)
    translator = AsyncSubtitleTranslator(cfg)

    with respx.mock:
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                _mock_openai_response([{"index": 1, "translated": "你好"}]),
                _mock_openai_response([{"index": 2, "translated": "世界"}]),
            ]
        )
        result = await translator.translate_segments(segments)

    assert route.call_count == 2
    assert [s.translated for s in result] == ["你好", "世界"]


async def test_translate_retries_on_rate_limit() -> None:
    segments = [SubtitleSegment(index=1, start=0.0, end=1.0, text="Retry me")]
    cfg = TranslatorConfig(api_key="test-key", max_retries=3)
    translator = AsyncSubtitleTranslator(cfg)

    with respx.mock, patch("subsync.translator.asyncio.sleep", new=AsyncMock()):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate_limited"}),
                _mock_openai_response([{"index": 1, "translated": "重试成功"}]),
            ]
        )
        result = await translator.translate_segments(segments)

    assert route.call_count == 2
    assert result[0].translated == "重试成功"
