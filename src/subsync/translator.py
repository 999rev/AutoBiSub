import asyncio
import json
import logging
import re
from dataclasses import dataclass

import httpx
from pydantic import ValidationError

from subsync.models import SubtitleSegment, TranslationBatchResponse

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一位专业的影视与技术视频双语字幕翻译专家。
请将输入的字幕条目准确、自然地翻译为{target_lang}。

严格遵守以下规则：
1. 保持口语化与上下文连贯，结合 context 中的前文语境理解代词和专业术语。
2. 必须严格按照原始 index 一一对应翻译，绝对不能合并、拆分或遗漏任何一条。
3. 只输出合法的 JSON 对象，格式如下，不要包含任何 Markdown 代码块标记或额外解释：
{{"items": [{{"index": 1, "translated": "译文内容"}}]}} """


@dataclass
class TranslatorConfig:
    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    target_lang: str = "简体中文"
    batch_size: int = 15
    context_size: int = 2
    max_concurrency: int = 5
    max_retries: int = 3
    timeout: float = 45.0


def _strip_code_fences(raw: str) -> str:
    """清理大模型可能附带的 ```json ... ``` 包裹"""
    cleaned = raw.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL)
    return match.group(1).strip() if match else cleaned


class AsyncSubtitleTranslator:
    def __init__(self, config: TranslatorConfig) -> None:
        self.cfg = config
        self._semaphore = asyncio.Semaphore(config.max_concurrency)

    async def translate_segments(
        self, segments: list[SubtitleSegment]
    ) -> list[SubtitleSegment]:
        """并发翻译所有字幕片段，返回填充了 translated 字段的新列表"""
        if not segments:
            return []

        batches: list[tuple[list[SubtitleSegment], list[SubtitleSegment]]] = []
        for i in range(0, len(segments), self.cfg.batch_size):
            ctx_start = max(0, i - self.cfg.context_size)
            context_segs = segments[ctx_start:i]
            target_segs = segments[i : i + self.cfg.batch_size]
            batches.append((context_segs, target_segs))

        headers = {
            "Authorization": f"Bearer {self.cfg.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(
            base_url=self.cfg.base_url.rstrip("/"),
            headers=headers,
            timeout=self.cfg.timeout,
        ) as client:
            tasks = [
                self._translate_batch_with_fallback(client, ctx, batch)
                for ctx, batch in batches
            ]
            batch_results = await asyncio.gather(*tasks)

        translated_map: dict[int, str] = {}
        for mapping in batch_results:
            translated_map.update(mapping)

        return [
            seg.model_copy(update={"translated": translated_map.get(seg.index, "")})
            for seg in segments
        ]

    async def _translate_batch_with_fallback(
        self,
        client: httpx.AsyncClient,
        context_segs: list[SubtitleSegment],
        target_segs: list[SubtitleSegment],
    ) -> dict[int, str]:
        """批量翻译 + 缺失条目单句降级补译"""
        result_map = await self._request_batch(client, context_segs, target_segs)

        missing_segs = [seg for seg in target_segs if seg.index not in result_map]
        if missing_segs:
            logger.warning(
                "批次中缺失 %d 条译文，触发单句降级补译: %s",
                len(missing_segs),
                [s.index for s in missing_segs],
            )
            for seg in missing_segs:
                single_res = await self._request_batch(client, [], [seg])
                result_map[seg.index] = single_res.get(seg.index, seg.text)

        return result_map

    async def _request_batch(
        self,
        client: httpx.AsyncClient,
        context_segs: list[SubtitleSegment],
        target_segs: list[SubtitleSegment],
    ) -> dict[int, str]:
        """带信号量控制与指数退避重试的 API 请求"""
        payload = {
            "model": self.cfg.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(target_lang=self.cfg.target_lang),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": [
                                {"index": s.index, "text": s.text}
                                for s in context_segs
                            ],
                            "subtitles": [
                                {"index": s.index, "text": s.text}
                                for s in target_segs
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                async with self._semaphore:
                    resp = await client.post("/chat/completions", json=payload)

                if resp.status_code == 429 or resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()

                content = resp.json()["choices"][0]["message"]["content"]
                parsed = TranslationBatchResponse.model_validate_json(
                    _strip_code_fences(content)
                )
                return {item.index: item.translated.strip() for item in parsed.items}

            except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                wait_seconds = min(2**attempt, 16)
                logger.warning(
                    "翻译请求失败 (第 %d/%d 次): %s，%.1fs 后重试",
                    attempt,
                    self.cfg.max_retries,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

        logger.error("批次翻译彻底失败，保留原文兜底: %s", last_error)
        return {}
