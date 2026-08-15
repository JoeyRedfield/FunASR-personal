# AGENTS.md — FunASR 课堂录音转写项目

## 项目简介

本地部署 FunASR，把中文课堂录音转成带时间戳、标点、说话人标签的文字笔记（Markdown / SRT / JSON），并提供 OpenAI 兼容的本地转写 API。

- 推理设备：macOS / Apple Silicon（M4），默认 CPU，可选 MPS
- 模型链路：`paraformer-zh`（识别+时间戳）+ `fsmn-vad`（语音分段）+ `ct-punc`（标点）+ `cam++`（说话人分离）
- 模型来源：ModelScope（`hub="ms"`，国内下载快），首次运行自动下载，之后全离线

## 常用命令

```bash
# 初始化环境（Python 3.12，需要 ffmpeg：brew install ffmpeg）
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

# 转写录音（推荐入口）
python scripts/transcribe.py samples/asr_example_zh.wav
python scripts/transcribe.py 课堂录音.m4a --hotword "机器学习 20" --device mps

# 启动 OpenAI 兼容 API（默认 http://127.0.0.1:8000）
bash scripts/start_api.sh

# 下载官方中文样例音频
bash scripts/download_sample.sh
```

转写结果输出到 `notes/`（同名 `.md` / `.srt` / `.json`）。脚本用 ffmpeg 先把任意音频格式转成 16 kHz 单声道 WAV 再推理；长音频按 `batch_size_s=300` 分批控制内存。

## 项目结构

```text
FunASR/
├── AGENTS.md             # 本文件
├── README.md
├── requirements.txt
├── .python-version       # 固定 3.12
├── scripts/
│   ├── transcribe.py     # 课堂录音一键转写（推荐入口）
│   ├── start_api.sh      # 启动 OpenAI 兼容转写 API
│   └── download_sample.sh# 下载官方中文样例音频
├── samples/              # 样例音频（首次测试用）
└── notes/                # 转写结果默认输出目录（已 gitignore）
```

## 约定与注意事项

- 默认用中文交流与写文档。
- `notes/`、`.venv/`、模型缓存、`__pycache__`、`.DS_Store` 等已在 `.gitignore` 中；**个人录音（如 `*.m4a`）属于隐私数据，不得提交到 git**，新增的大文件应加入 `.gitignore` 而非入库。
- 需要删除任何文件前，先询问用户。
- 模型默认缓存在 `~/.cache/modelscope/models/`（本机约 2.1 GB，含 paraformer / fsmn-vad / ct-punc / cam++ 四套模型），与项目内 `models/` 目录不同；想改位置用环境变量 `MODELSCOPE_CACHE`。
- 本机 shell 直连网络通常不通，执行下载/安装类命令前先走代理：
  `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890`
- 模型下载慢时改用 ModelScope 本地目录；MPS 不稳定就退回 `--device cpu`（CPU 已足够快）。
- 项目相关个人信息（技术栈、联系方式等）见本机私人笔记（`~/Documents/notes/private/个人信息/个人技术栈和情况简介.md`）。

## 开发约定

- 保持 `transcribe.py` 为推荐入口：CLI 参数、延迟导入模型、临时 WAV 目录这些结构不要破坏。
- 输出文件命名与输入同名（`stem`），新增输出格式时同步更新 `render_outputs` 与 README。
- 远程仓库为 `JoeyRedfield/FunASR-personal`，推送前确认不包含隐私音频与笔记。
