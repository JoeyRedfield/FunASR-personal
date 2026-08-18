# FunASR 课堂录音转写项目

本地部署 FunASR，把中文课堂录音转成带时间戳、标点、说话人标签的文字笔记。

- 推理设备：macOS / Apple Silicon（M4），默认 CPU，可选 MPS
- 模型链路：paraformer-zh（识别+时间戳）+ fsmn-vad（语音分段）+ ct-punc（标点）+ cam++（说话人分离）
- 模型来源：ModelScope（国内下载快），首次运行自动下载，之后全离线
- 输出格式：Markdown 笔记、SRT 字幕、JSON 原始结果
- 附加能力：OpenAI 兼容的本地转写 API（`funasr-server`）

## 项目结构

```text
FunASR/
├── README.md
├── requirements.txt
├── .python-version
├── scripts/
│   ├── transcribe.py        # 课堂录音一键转写（推荐入口）
│   ├── polish_notes.py      # LLM 纠错顺滑 + 结构化笔记
│   ├── evaluate_recording_devices.py # 手机/平板录音成对评估
│   ├── start_api.sh         # 启动 OpenAI 兼容转写 API
│   └── download_sample.sh   # 下载官方中文样例音频
├── docs/
│   ├── device-evaluation.md # 手机/平板录音评估操作与评分说明
│   └── roadmap.md           # 转写后续方案记录
├── tests/
│   └── test_evaluate_recording_devices.py # 设备评估单元与流程测试
├── samples/                 # 样例音频（首次测试用）
└── notes/                   # 转写结果默认输出目录
```

## 快速开始

### 1. 创建环境并安装依赖

> 如果你的 shell 直连网络不通（本机通常需要走 127.0.0.1:7890 代理），先执行：
> `export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890`

```bash
cd ~/Desktop/code/FunASR
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

验证安装：

```bash
python -c "import torch, funasr; print(torch.__version__, funasr.__version__)"
```

### 2. 下载样例音频并测试

```bash
bash scripts/download_sample.sh
python scripts/transcribe.py samples/asr_example_zh.wav
```

第一次运行会自动从 ModelScope 下载模型（约 2～3 GB），之后全离线。转写结果输出到 `notes/`：

- `asr_example_zh.md`：带时间戳和说话人标签的课堂笔记
- `asr_example_zh.srt`：字幕文件
- `asr_example_zh.json`：FunASR 原始结果

### 3. 转写自己的录音

```bash
python scripts/transcribe.py "~/Desktop/课堂录音.m4a"

# 指定输出目录
python scripts/transcribe.py 课堂录音.m4a -o ~/Documents/课堂笔记

# 批量转写
python scripts/transcribe.py 录音1.m4a 录音2.m4a -o ./notes

# 课程热词（词 权重，权重 1~100）
python scripts/transcribe.py 课堂录音.m4a --hotword "机器学习 20 线性回归 20"

# 不加载说话人分离（更快）
python scripts/transcribe.py 课堂录音.m4a --no-spk

# 尝试 MPS 加速（不稳定则去掉 --device mps）
python scripts/transcribe.py 课堂录音.m4a --device mps
```

脚本会自动用 ffmpeg 把任意格式（m4a/mp3/wav/mp4 等）转成 16 kHz 单声道 WAV，再调用 FunASR。

## 可选：LLM 纠错顺滑 + 结构化笔记

转写得到的是逐句流水账，可用 LLM 做后处理：

- **纠错顺滑**：改正识别错别字/术语错误（如“工案改善”→“提案改善”），删除“呃、嗯、以以”等口头禅
- **结构化笔记**：生成概述、大纲、核心要点、行动项、专有名词与常见问答

```bash
python scripts/polish_notes.py notes/体验改善培训内容.json
```

输出（与输入同名）：

- `notes/<名称>.整理.md`：纠错顺滑后的逐句文本（保留时间戳/说话人）
- `notes/<名称>.polished.json`：纠错后逐句数据（供后续互动笔记/搜索使用）
- `notes/<名称>.结构化笔记.md`：结构化课堂笔记

LLM 后端默认本地 Ollama（OpenAI 兼容端点），可用环境变量切换：

```bash
export LLM_BASE_URL=http://127.0.0.1:11434/v1   # 默认值
export LLM_MODEL=qwen3:0.6b                      # 默认值，建议用 >=8B 模型
export LLM_API_KEY=                              # 云端 API 时填写
```

常用参数：`--limit N` 只试跑前 N 句；`--skip-polish` / `--skip-notes` 跳过某一步；`--segments-per-chunk` 控制每批句子数。完整方案见 [docs/roadmap.md](docs/roadmap.md)。

## 手机与平板录音评估

把同一课程的同步录音命名为 `<课程>-手机.wav` 和 `<课程>-平板.m4a`，放进 `undo/`，然后分阶段运行：

```bash
python scripts/evaluate_recording_devices.py prepare
python scripts/evaluate_recording_devices.py run-asr
python scripts/evaluate_recording_devices.py run-notes
# 填写人工参考文本与 notes/device-evaluation/ratings.json 后：
python scripts/evaluate_recording_devices.py report
```

`run-asr` 实际调用本项目推荐入口 `scripts/transcribe.py`，固定使用 CPU、完整的 VAD/标点/说话人模型链路且不设置热词；`run-notes` 实际调用 `scripts/polish_notes.py`，两端共用 `.env` 中的相同 DeepSeek 配置。完整操作与评分说明见 [docs/device-evaluation.md](docs/device-evaluation.md)。

运行设备评估测试：`python -m unittest discover -s tests -v`。

## 可选：启动本地转写 API

```bash
bash scripts/start_api.sh
```

默认监听 `http://127.0.0.1:8000`，提供 OpenAI 兼容接口：

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@samples/asr_example_zh.wav \
  -F model=paraformer \
  -F response_format=verbose_json
```

可用环境变量：`FUNASR_MODEL`（默认 paraformer，已随转写脚本缓存；也可改为 sensevoice 体验更快速度）、`FUNASR_DEVICE`（默认 cpu）、`FUNASR_HOST`、`FUNASR_PORT`。

## 性能参考（M4 / 16GB）

| 项目 | 预期 |
|---|---|
| 转写速度 | paraformer 全链路约 6～15 分钟 / 90 分钟课 |
| 内存峰值 | 约 2～4 GB |
| 磁盘占用 | 环境约 2 GB + 模型约 2～3 GB |

## 模型缓存

首次运行会自动从 ModelScope 下载模型，默认保存到用户缓存目录（macOS/Linux 为 `~/.cache/modelscope/models/`，本机实际约 2.1 GB）。下载完成后即可全离线使用。

| 模型 | 用途 | 缓存目录 | 大小 |
|---|---|---|---|
| paraformer | 识别 + 时间戳 | `iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | ~950 MB |
| fsmn-vad | 语音分段 | `iic--speech_fsmn_vad_zh-cn-16k-common-pytorch` | ~4 MB |
| ct-punc | 标点恢复 | `iic--punc_ct-transformer_cn-en-common-vocab471067-large` | ~1.1 GB |
| cam++ | 说话人分离 | `iic--speech_campplus_sv_zh-cn_16k-common` | ~28 MB |

说明：

- 想改缓存位置：设置环境变量 `MODELSCOPE_CACHE=/你的/路径`，之后新模型会下载到新目录（旧缓存不会自动迁移）。
- 想重下模型：删除对应缓存目录后，下次运行会自动重新下载。
- 该缓存位于家目录，与项目内的 `models/`（本地模型目录，已 gitignore）不是一回事，两者互不影响。

## 常见问题

- **为什么用 Python 3.12？** 系统 Python 3.14 与 torch/torchaudio 的 macOS wheel 兼容性有风险，3.12 最稳。
- **模型下载慢？** 脚本默认 `hub="ms"` 走 ModelScope；也可手动 `modelscope download` 后把模型路径换成本地目录。
- **MPS 报错？** 退回 CPU 即可，CPU 已经足够快。
- **专有名词错？** 使用 `--hotword "术语 20"`，权重建议 10～30。

## 参考

- [FunASR 中文 README](https://github.com/modelscope/FunASR/blob/main/README_zh.md)
- [FunASR PyPI](https://pypi.org/project/funasr/)
- [OpenAI 兼容 API 服务](https://github.com/modelscope/FunASR/blob/main/examples/openai_api/README.md)
