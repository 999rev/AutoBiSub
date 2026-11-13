import asyncio
import logging
import os
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.logging import RichHandler

from subsync.asr import ASRConfig, LocalWhisperASR
from subsync.audio import temporary_wav
from subsync.subtitle import parse_srt, write_srt
from subsync.translator import AsyncSubtitleTranslator, TranslatorConfig
from subsync.watcher import watch_directory

app = typer.Typer(
    name="subsync",
    help="本地音视频语音识别与双语字幕自动化工具",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


def process_file(
    input_path: Path,
    output_path: Path | None,
    asr_cfg: ASRConfig,
    trans_cfg: TranslatorConfig,
    order: Literal["translated_first", "original_first"],
    skip_translate: bool,
) -> Path:
    """处理单个媒体或 SRT 文件的核心流水线"""
    if not input_path.exists():
        raise typer.BadParameter(f"输入文件不存在: {input_path}")

    target_srt = output_path or input_path.with_suffix(".bilingual.srt")

    if input_path.suffix.lower() == ".srt":
        console.print(f"[cyan]检测到 SRT 字幕文件，跳过转写:[/cyan] {input_path.name}")
        segments = parse_srt(input_path)
    else:
        asr = LocalWhisperASR(asr_cfg)
        with (
            console.status("[bold green]正在提取音轨并执行本地语音识别...[/bold green]"),
            temporary_wav(input_path) as wav_path,
        ):
                segments = asr.transcribe(wav_path)

    if not segments:
        raise RuntimeError("未识别到任何有效语音或字幕内容")

    console.print(f"[green]✓[/green] 共提取 {len(segments)} 条字幕片段")

    if not skip_translate:
        translator = AsyncSubtitleTranslator(trans_cfg)
        with console.status(
            f"[bold cyan]正在调用 {trans_cfg.model} 并发翻译为{trans_cfg.target_lang}...[/bold cyan]"
        ):
            segments = asyncio.run(translator.translate_segments(segments))
        console.print("[green]✓[/green] 字幕翻译完成")

    saved_path = write_srt(segments, target_srt, order=order)
    console.print(f"[bold green]全部完成！已保存至:[/bold green] {saved_path}")
    return saved_path


@app.command()
def run(
    input_path: Path = typer.Argument(..., help="视频、音频或 .srt 字幕文件路径"),
    output: Path | None = typer.Option(None, "-o", "--output", help="输出字幕路径"),
    model_size: str = typer.Option("small", "-m", "--model", help="Whisper 模型规格或本地路径"),
    device: str = typer.Option("cpu", "--device", help="推理设备: cpu / cuda / auto"),
    compute_type: str = typer.Option("int8", "--compute-type", help="量化精度: int8 / float16"),
    language: str | None = typer.Option(None, "-l", "--lang", help="源音频语种，如 en / ja"),
    target_lang: str = typer.Option("简体中文", "--target-lang", help="目标翻译语言"),
    llm_model: str = typer.Option(
        os.getenv("SUBSYNC_LLM_MODEL", "gpt-4o-mini"),
        "--llm-model",
        help="翻译使用的模型名称",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar=["SUBSYNC_API_KEY", "OPENAI_API_KEY"],
        help="OpenAI 兼容接口密钥",
    ),
    base_url: str = typer.Option(
        os.getenv("SUBSYNC_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        "--base-url",
        help="OpenAI 兼容接口 Base URL",
    ),
    order: Literal["translated_first", "original_first"] = typer.Option(
        "translated_first",
        "--order",
        help="双语显示顺序",
    ),
    skip_translate: bool = typer.Option(False, "--skip-translate", help="仅生成原语种字幕，不翻译"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="输出详细调试日志"),
) -> None:
    """对单个音视频或字幕文件生成双语 SRT 字幕"""
    _setup_logging(verbose)

    if not skip_translate and not api_key:
        console.print("[red]错误:[/red] 未配置 API Key，请设置 OPENAI_API_KEY 或传入 --api-key。")
        raise typer.Exit(code=1)

    asr_cfg = ASRConfig(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    trans_cfg = TranslatorConfig(
        api_key=api_key or "",
        base_url=base_url,
        model=llm_model,
        target_lang=target_lang,
    )

    try:
        process_file(
            input_path=input_path,
            output_path=output,
            asr_cfg=asr_cfg,
            trans_cfg=trans_cfg,
            order=order,
            skip_translate=skip_translate,
        )
    except Exception as exc:
        console.print(f"[bold red]处理失败:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc


@app.command()
def watch(
    watch_dir: Path = typer.Argument(..., help="需要监听的视频/音频目录"),
    db_path: Path | None = typer.Option(
        None,
        "--db",
        help="SQLite 去重状态数据库路径",
    ),
    recursive: bool = typer.Option(False, "-r", "--recursive", help="是否递归监听子目录"),
    scan_existing: bool = typer.Option(
        False, "--scan-existing", help="启动时是否扫描并处理目录内已有的未字幕文件"
    ),
    model_size: str = typer.Option("small", "-m", "--model", help="Whisper 模型规格或本地路径"),
    device: str = typer.Option("cpu", "--device", help="推理设备: cpu / cuda / auto"),
    compute_type: str = typer.Option("int8", "--compute-type", help="量化精度: int8 / float16"),
    language: str | None = typer.Option(None, "-l", "--lang", help="源音频语种，如 en / ja"),
    target_lang: str = typer.Option("简体中文", "--target-lang", help="目标翻译语言"),
    llm_model: str = typer.Option(
        os.getenv("SUBSYNC_LLM_MODEL", "gpt-4o-mini"),
        "--llm-model",
        help="翻译使用的模型名称",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        envvar=["SUBSYNC_API_KEY", "OPENAI_API_KEY"],
        help="OpenAI 兼容接口密钥",
    ),
    base_url: str = typer.Option(
        os.getenv("SUBSYNC_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")),
        "--base-url",
        help="OpenAI 兼容接口 Base URL",
    ),
    order: Literal["translated_first", "original_first"] = typer.Option(
        "translated_first",
        "--order",
        help="双语显示顺序",
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="输出详细调试日志"),
) -> None:
    """后台监听目录，自动为新增音视频生成双语 SRT 字幕"""
    _setup_logging(verbose)

    if not api_key:
        console.print("[red]错误:[/red] 未配置 API Key，请设置 OPENAI_API_KEY 或传入 --api-key。")
        raise typer.Exit(code=1)

    asr_cfg = ASRConfig(
        model_size_or_path=model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    trans_cfg = TranslatorConfig(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
        target_lang=target_lang,
    )

    def _callback(media_path: Path) -> Path:
        return process_file(
            input_path=media_path,
            output_path=None,
            asr_cfg=asr_cfg,
            trans_cfg=trans_cfg,
            order=order,
            skip_translate=False,
        )

    watch_directory(
        watch_dir=watch_dir,
        db_path=db_path or (Path.home() / ".subsync" / "state.db"),
        process_callback=_callback,
        recursive=recursive,
        scan_existing=scan_existing,
    )


@app.command()
def version() -> None:
    """显示当前版本"""
    console.print("subsync v0.1.0")


if __name__ == "__main__":
    app()
