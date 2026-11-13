# AutoBiSub

基于 `faster-whisper` 本地语音识别与异步 LLM 并发翻译的音视频双语字幕自动化工具（支持单文件转写、已有 `.srt` 直译与后台目录监听）。

## 核心特性

- **本地离线语音识别**：集成 `faster-whisper`，默认开启 VAD 静音过滤与 `word_timestamps` 逐词时间戳对齐，减少提前出字与静音幻觉。
- **多音轨自动匹配**：通过 `ffprobe` 读取音轨语言标签，传入 `-l en` 等参数时自动提取对应语种音轨（也支持 `--track` 手动指定音轨序号）。
- **异步分块并发翻译**：基于 `asyncio` + `httpx` 携带上下文窗口并发请求兼容 OpenAI 协议的接口（DeepSeek / OpenAI / 通义千问 / Ollama 等），支持 `429` 退避重试与漏行单句补译。
- **后台目录监听**：使用 `watchdog` 监听下载文件夹，配合文件写入稳定性检测与 SQLite 快速指纹去重，防止重复处理。

---

## 环境准备

### 1. 安装 Python 3.10+ 与 FFmpeg

- **Python**：需要 `>= 3.10`（Windows 若默认版本较低，可运行 `winget install Python.Python.3.11`）。
- **FFmpeg**：
  - Windows：`winget install Gyan.FFmpeg`（安装后重开终端，确保 `ffmpeg -version` 可正常输出）
  - macOS：`brew install ffmpeg`
  - Ubuntu / Debian：`sudo apt update && sudo apt install -y ffmpeg`

### 2. 创建虚拟环境并安装项目

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

*(macOS / Linux 激活命令为 `source .venv/bin/activate`)*

### 3. 开启 NVIDIA GPU 加速（可选，无独显可跳过）

如果本机有 NVIDIA 显卡，无需安装完整 CUDA Toolkit，在虚拟环境中补装精简运行库即可开启 `--device cuda --compute-type float16`：

```powershell
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 -i https://pypi.tuna.tsinghua.edu.cn/simple
$env:Path += ";$PWD\.venv\Lib\site-packages\nvidia\cublas\bin;$PWD\.venv\Lib\site-packages\nvidia\cudnn\bin"
```

---

## 配置环境变量

### 1. 本地模型下载镜像（国内网络推荐）

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

### 2. 翻译模型接口（任选一种）

**方案 A：DeepSeek API（推荐，并发速度快）**
```powershell
$env:OPENAI_BASE_URL="https://api.deepseek.com/v1"
$env:SUBSYNC_LLM_MODEL="deepseek-chat"
$env:OPENAI_API_KEY="你的DeepSeek_API_KEY"
```

**方案 B：本地 Ollama（100% 离线免费，适合小批量或挂机）**
```powershell
$env:OPENAI_BASE_URL="http://localhost:11434/v1"
$env:SUBSYNC_LLM_MODEL="qwen2.5:7b"
$env:OPENAI_API_KEY="ollama"
```

---

## 使用方法

### 1. 处理单个视频（生成双语 `.bilingual.srt`）

```powershell
# CPU 模式（默认）
subsync run ".\demo.mp4" -m small -l en

# GPU 加速 + 高精度模型（适合多口音/快语速/俚语场景）
subsync run ".\demo.mp4" -m large-v3-turbo -l en --device cuda --compute-type float16

# 仅生成原语种字幕，不调用翻译接口
subsync run ".\demo.mp4" -m small -l en --skip-translate

# 手动指定第 2 条音轨（序号从 0 开始）
subsync run ".\demo.mkv" --track 1 -l en
```

### 2. 直接翻译已有的 `.srt` 字幕

```powershell
subsync run ".\demo.srt"
```

### 3. 后台监听目录（自动处理新移入的视频）

```powershell
subsync watch "D:\Downloads\Videos" --scan-existing -m large-v3-turbo -l en --device cuda --compute-type float16
```

按 `Ctrl + C` 即可优雅停止监听；已处理过的文件指纹默认保存在 `~/.subsync/state.db`。

---

## 本地测试与代码检查

```powershell
ruff check src tests
pytest -v
```
