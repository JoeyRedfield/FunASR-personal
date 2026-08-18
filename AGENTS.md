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

# LLM 后处理：纠错顺滑 + 结构化笔记（自动读取 .env 里的 DeepSeek 配置）
python scripts/polish_notes.py notes/体验改善培训内容/体验改善培训内容.json

# 手机/平板录音评估（prepare 后依次执行其余阶段）
python scripts/evaluate_recording_devices.py prepare
python scripts/evaluate_recording_devices.py run-asr
python scripts/evaluate_recording_devices.py run-notes
python scripts/evaluate_recording_devices.py report

# 运行测试
python -m unittest discover -s tests -v

# 启动 OpenAI 兼容 API（默认 http://127.0.0.1:8000）
bash scripts/start_api.sh

# 下载官方中文样例音频
bash scripts/download_sample.sh
```

转写结果默认输出到 `notes/<录音文件名>/`（同名前缀的 `.md` / `.srt` / `.json` 集中存放）；LLM 后处理的 `.整理.md` / `.polished.json` / `.结构化笔记.md` 也写入同一前缀目录。`--flat-output` 只供需要旧平铺布局的现有集成使用。脚本用 ffmpeg 先把任意音频格式转成 16 kHz 单声道 WAV 再推理；长音频按 `batch_size_s=300` 分批控制内存。

## 标准录音处理与归档工作流

```bash
# 1. 批量转写：每个录音写入 notes/<stem>/
python scripts/transcribe.py 录音1.wav 录音2.m4a -o notes

# 2. 后处理：输入各自目录内的 JSON，结果仍回到同一 stem 目录
python scripts/polish_notes.py \
  notes/录音1/录音1.json notes/录音2/录音2.json

# 3. 检查结构化笔记和输出完整性，再运行测试
python -m unittest discover -s tests -v
```

批量输入的 stem 必须唯一；脚本会在模型或 LLM 调用前拒绝重复前缀，避免覆盖。设备评估使用 `evaluate_recording_devices.py` 的分阶段目录和内部 `--flat-output` 兼容模式，不要手工改成普通录音目录。提交前只提交代码、规则、文档和测试，`notes/`、录音、`.env` 与评估材料保持忽略。

## 项目结构

```text
FunASR/
├── AGENTS.md             # 本文件
├── README.md
├── requirements.txt
├── .python-version       # 固定 3.12
├── scripts/
│   ├── transcribe.py     # 课堂录音一键转写（推荐入口）
│   ├── polish_notes.py   # LLM 纠错顺滑 + 结构化笔记
│   ├── evaluate_recording_devices.py # 手机/平板录音成对评估
│   ├── start_api.sh      # 启动 OpenAI 兼容转写 API
│   └── download_sample.sh# 下载官方中文样例音频
├── docs/
│   ├── device-evaluation.md # 录音设备评估指南与评分规则
│   └── roadmap.md        # 转写后续方案记录（互动笔记/搜索问答/统计）
├── tests/
│   ├── test_evaluate_recording_devices.py # 设备评估测试
│   └── test_output_layout.py # 普通转写与笔记输出目录测试
├── samples/              # 样例音频（首次测试用）
└── notes/                # 按录音前缀分目录的输出根目录（已 gitignore）
```

## 约定与注意事项

- 默认用中文交流与写文档。
- `notes/`、`undo/`、`archive/`、`.venv/`、模型缓存、`__pycache__`、`.DS_Store` 等已在 `.gitignore` 中；**个人录音属于隐私数据，不得提交到 git**。项目支持的 `.aac` / `.flac` / `.m4a` / `.mp3` / `.mp4` / `.wav` 已全局忽略，只对官方样例 `samples/asr_example_zh.wav` 放行；新增音频格式时同步补充 `.gitignore`。
- 需要删除任何文件前，先询问用户。
- LLM 的 key（DeepSeek）存于项目根 `.env`（已 gitignore），**不要提交或外泄**；`polish_notes.py` 会自动读取。本地 `qwen3:0.6b` 实测质量差（丢信息、重复句子），后处理优先用 DeepSeek，需离线时改用 ≥8B 模型。
- 模型默认缓存在 `~/.cache/modelscope/models/`（本机约 2.1 GB，含 paraformer / fsmn-vad / ct-punc / cam++ 四套模型），与项目内 `models/` 目录不同；想改位置用环境变量 `MODELSCOPE_CACHE`。
- 本机 shell 直连网络通常不通，执行下载/安装类命令前先走代理：
  `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890`
- 模型下载慢时改用 ModelScope 本地目录；MPS 不稳定就退回 `--device cpu`（CPU 已足够快）。
- 项目相关个人信息（技术栈、联系方式等）见本机私人笔记（`~/Documents/notes/private/个人信息/个人技术栈和情况简介.md`）。

## 开发约定

- 保持 `transcribe.py` 为推荐入口：CLI 参数、延迟导入模型、临时 WAV 目录这些结构不要破坏。
- 输出文件命名与输入同名（`stem`），普通流程默认写入 `<output-dir>/<stem>/`，批量输入的 stem 必须唯一；设备评估通过 `--flat-output` 保持其分阶段目录合同。新增输出格式时同步更新 `render_outputs`、相关测试与 README。
- LLM 后处理配置走 `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` 环境变量（`.env` 已配 DeepSeek，脚本自动加载），脚本兜底才连本地 Ollama；尽量不引入新依赖。
- 设备评估的操作、评分权重和报告判定以 `docs/device-evaluation.md` 为准；真实音频、转写、盲评材料和评分结果只保留在已忽略目录中。
- 远程仓库为 `JoeyRedfield/FunASR-personal`，推送前确认不包含隐私音频与笔记。
