from pydantic import BaseModel, Field


class SubtitleSegment(BaseModel):
    """单条原始字幕片段（由 ASR 或 SRT 解析产生）"""

    index: int = Field(..., ge=1, description="1-based 字幕序号")
    start: float = Field(..., ge=0.0, description="开始时间（秒）")
    end: float = Field(..., ge=0.0, description="结束时间（秒）")
    text: str = Field(..., description="原始语音文本")
    translated: str | None = Field(default=None, description="翻译后的文本")


class TranslationItem(BaseModel):
    """LLM 返回的单条结构化校验模型"""

    index: int
    translated: str


class TranslationBatchResponse(BaseModel):
    """LLM 批量返回的 JSON 校验模型"""

    items: list[TranslationItem]
