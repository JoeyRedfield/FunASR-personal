# 手机与平板课堂录音评估

本指南用于比较同一课堂中手机和平板录音对 FunASR 转写和结构化笔记质量的影响。工具负责配对、同步和裁切，语音识别与笔记生成仍由项目现有脚本完成。

所有原始录音和结果都位于已被 Git 忽略的 `undo/`、`notes/` 中。不要把录音、逐字稿、盲评材料或评分结果提交到仓库。

## 1. 准备录音

每门课需要一份手机录音和一份平板录音。两份录音必须来自同一场课堂，文件名只在末尾设备名称上有区别：

```text
undo/
├── 质量基础知识培训-手机.wav
├── 质量基础知识培训-平板.m4a
├── 新员工三级培训指引-手机.wav
└── 新员工三级培训指引-平板.m4a
```

支持 AAC、FLAC、M4A、MP3、MP4 和 WAV。原始编码、声道数和采样率可以不同，后续会统一转成 16 kHz 单声道 WAV。
默认抽取三个互不重叠的 3 分钟片段，因此两端共同录制区间必须至少有 9 分钟。

激活项目 Python 环境，并确认 ffmpeg 可用：

```bash
source .venv/bin/activate
ffmpeg -version
python scripts/evaluate_recording_devices.py --help
```

## 2. 对齐与裁切

```bash
python scripts/evaluate_recording_devices.py prepare
```

`prepare` 会完成以下工作：

- 按 `-手机`、`-平板` 自动配对课程；
- 根据两端语音能量变化计算启动时间偏移；
- 裁出两端完整的共同录制区间；
- 把交集分成前、中、后三段，每段选择一个 3 分钟高语音密度窗口；
- 生成清单、每个课程抽样片段对应的空白人工参考文本和评分模板。

默认输出目录为 `notes/device-evaluation/`。重要文件包括：

```text
notes/device-evaluation/
├── aligned/       # 完整共同区间，每门课两份
├── samples/       # 前中后抽样片段，每门课六份
├── references/    # 已按片段命名的空白人工参考文本
├── manifest.json  # 文件哈希、时间偏移、命令与模型配置
└── ratings.json   # 人工评分模板
```

工具不会覆盖已有结果。如果需要重新评估，请指定新的输出目录，例如：

```bash
python scripts/evaluate_recording_devices.py prepare \
  --output-dir notes/device-evaluation-2
```

后续阶段需要显式指向这份清单；生成报告时也要指定同目录的评分表、盲评映射和输出路径：

```bash
python scripts/evaluate_recording_devices.py run-asr \
  --manifest notes/device-evaluation-2/manifest.json
python scripts/evaluate_recording_devices.py run-notes \
  --manifest notes/device-evaluation-2/manifest.json
python scripts/evaluate_recording_devices.py report \
  --manifest notes/device-evaluation-2/manifest.json \
  --ratings notes/device-evaluation-2/ratings.json \
  --blind-key notes/device-evaluation-2/blind/key.json \
  --output notes/device-evaluation-2/report.md
```

自动对齐失败时，先确认两份文件是否确实为同步录音。录音启停时间相差超过 30 分钟时，可扩大搜索范围：

```bash
python scripts/evaluate_recording_devices.py prepare --max-offset-seconds 3600
```

## 3. 使用 FunASR 转写

```bash
python scripts/evaluate_recording_devices.py run-asr
```

该命令读取 `manifest.json`，然后统一调用：

```bash
python scripts/transcribe.py <全部完整交集和抽样片段> \
  -o notes/device-evaluation/asr \
  --device cpu \
  --flat-output
```

设备评估按 `asr/`、`notes/` 阶段隔离结果，因此显式使用兼容参数 `--flat-output`，继续维持评估清单和续跑逻辑所依赖的平铺文件名；普通转写仍默认按录音前缀分目录。

因此两台设备使用完全相同的 FunASR 链路：

- `paraformer-zh`：中文识别与时间戳；
- `fsmn-vad`：语音活动检测；
- `ct-punc`：标点恢复；
- `cam++`：说话人分离。

评估不使用热词，也不为某台设备调整参数。完整交集和抽样片段都会输出 `.md`、`.srt`、`.json`，评分使用未经过 LLM 修改的 `.json`。

## 4. 生成整理稿和笔记

确认 `.env` 中的 `LLM_BASE_URL`、`LLM_MODEL`、`LLM_API_KEY` 已配置，然后运行：

```bash
python scripts/evaluate_recording_devices.py run-notes
```

该命令统一调用 `scripts/polish_notes.py --flat-output`，在评估专用的 `notes/` 阶段目录中生成纠错稿和结构化笔记。完成后会把每门课的手机、平板整课笔记随机匿名为 A/B，放入：

```text
notes/device-evaluation/blind/<课程>/
├── A.整理.md
├── A.结构化笔记.md
├── B.整理.md
└── B.结构化笔记.md
```

在笔记评分完成前，不要打开 `blind/key.json`，该文件保存 A/B 与设备的对应关系。

首次执行会生成 `llm-execution.json`，记录实际使用的端点、模型、是否配置 key 和 key 指纹，但不保存 key。后续续跑会同时核对该记录、当前 `.env` 与已有 `polished.json` 的模型；任一不一致都会拒绝混用结果。需要更换 LLM 配置时，应使用新的评估输出目录。

## 5. 人工校对与评分

逐一播放 `samples/` 中同编号的手机和平板录音，把确认后的逐字内容填入已经创建好的 `references/<片段>.txt`。参考文本可以交叉聆听两端确认，但不要标注设备。

随后填写 `ratings.json` 中所有 `null` 值，分数范围均为 0–100：

- `term_number_accuracy`：专有名词和数字的准确程度；
- `omission_score`：内容越完整，分数越高；
- `timestamp_speaker_score`：时间戳和说话人标签的可用程度；
- `factual_fidelity`：盲评笔记是否忠实于课堂内容；
- `coverage`：关键内容覆盖程度；
- `usability`：结构、大纲和行动项是否便于使用。

先填写 A/B 笔记评分，再查看设备映射。字符错误率 CER 由工具根据人工参考文本自动计算，无需手工填写。

## 6. 生成结论

```bash
python scripts/evaluate_recording_devices.py report
```

报告输出为：

- `notes/device-evaluation/report.md`：适合阅读的结果与建议；
- `notes/device-evaluation/report.json`：逐课程和总体结构化得分，以及 `reproducibility` 区块中的音频哈希、交集范围、抽样位置、实际命令、模型链路和公开 LLM 配置。该文件不包含课堂转写正文、原始录音路径或 API key。

总分按照以下权重计算：

| 指标 | 权重 |
|---|---:|
| 字符准确率（`1 - CER`） | 35% |
| 专有名词和数字准确率 | 20% |
| 漏句评分 | 10% |
| 时间戳与说话人 | 5% |
| 笔记事实忠实度 | 15% |
| 笔记内容覆盖率 | 10% |
| 笔记结构可用性 | 5% |

设备只有在总分领先至少 5 分、平均 CER 相对改善至少 10%，并且至少在两门课程中得分领先时，才会被建议为默认设备。否则报告会给出“质量无显著差异”，此时应再根据摆放、续航、存储和日常操作选择设备。
