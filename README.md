# subsync

基于 `faster-whisper` 本地语音识别与异步 LLM 并发翻译的音视频双语字幕自动化工具（CLI + 目录监听）。

## 核心特性

- **本地离线语音识别**：集成 `faster-whisper` 与内置 VAD 静音过滤，支持 CPU (`int8`) 与 GPU (`float16`)。
- **异步分块并发翻译**：基于 `asyncio` + `httpx` 携带上下文窗口并发翻译，支持 `429` 指数退避重试与漏行自动单句补译。
- **双语 SRT 校验与直译**：严格校验时间轴对齐与毫秒进位，支持视频提取转写及已有 `.srt` 文件直译。
- **后台目录监听**：使用 `watchdog` 监听下载目录，配合文件写入稳定性检测与 SQLite 快速指纹去重，避免重复处理。

## 快速开始

### 1. 前置依赖

确保系统已安装 `ffmpeg` 与 Python >= 3.10：

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg
```

### 2. 安装项目

```bash
pip install -e ".[dev]"
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY
```

### 3. 常用命令

```bash
# 处理单个视频并生成双语字幕
subsync run ./demo.mp4 -m small

# 直接翻译已有的外语 .srt 字幕
subsync run ./demo.srt

# 启动目录监听（自动处理新移入的视频并跳过已处理文件）
subsync watch ~/Downloads/Videos --scan-existing -m small
```

### 4. Docker 一键运行

```bash
docker compose up -d --build
```

## 运行测试

```bash
ruff check src tests
pytest -v
```
