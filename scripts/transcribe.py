#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
课堂录音 FunASR 本地转写脚本（macOS / Apple Silicon 友好）

依赖：
  - Python 3.12 虚拟环境（uv venv .venv --python 3.12）
  - requirements.txt 中的依赖
  - 系统已安装 ffmpeg（brew install ffmpeg）

用法示例：
  python scripts/transcribe.py 课堂录音.m4a
  python scripts/transcribe.py 录音1.m4a 录音2.wav -o ./notes --hotword "机器学习 20"
  python scripts/transcribe.py 课堂录音.m4a --device mps

输出（默认按输入文件名前缀分目录）：
  <output-dir>/<stem>/<stem>.md   —— 带时间戳和说话人标签的课堂笔记
  <output-dir>/<stem>/<stem>.srt  —— 字幕文件
  <output-dir>/<stem>/<stem>.json —— FunASR 原始结果（时间戳/说话人/分段全文）
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def to_wav_16k(src: Path, dst: Path) -> None:
    """用 ffmpeg 统一转成 16 kHz 单声道 WAV。"""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
        capture_output=True,
    )


def fmt_srt_ts(ms: float) -> str:
    ms = int(ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms2 = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms2:03d}"


def recording_output_dir(base_dir: Path, stem: str, *, flat_output: bool = False) -> Path:
    """返回单个录音的输出目录。"""
    return base_dir if flat_output else base_dir / stem


def ensure_unique_stems(paths: list[Path]) -> None:
    """拒绝会写入同一前缀目录的批量输入。"""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for path in paths:
        if path.stem in seen:
            duplicates.add(path.stem)
        seen.add(path.stem)
    if duplicates:
        raise ValueError(f"输入文件名前缀重复：{'、'.join(sorted(duplicates))}")


def render_outputs(stem: str, raw: dict, output_dir: Path) -> None:
    """把 FunASR 结果写为 md / srt / json 三份文件。"""
    (output_dir / f"{stem}.json").write_text(
        json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sentences = raw.get("sentence_info") or []
    md_lines = [f"# {stem}", ""]
    srt_lines = []

    if sentences:
        for i, seg in enumerate(sentences, 1):
            start = seg.get("start", 0)
            end = seg.get("end", start)
            spk = seg.get("spk", "?")
            text = seg.get("sentence") or seg.get("text") or ""
            md_lines.append(f"[{start / 1000:.0f}s] 说话人{spk}: {text}")
            srt_lines.append(str(i))
            srt_lines.append(f"{fmt_srt_ts(start)} --> {fmt_srt_ts(end)}")
            srt_lines.append(f"[说话人{spk}] {text}")
            srt_lines.append("")
    else:
        md_lines.append(raw.get("text", ""))

    (output_dir / f"{stem}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (output_dir / f"{stem}.srt").write_text("\n".join(srt_lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="FunASR 课堂录音本地转写")
    parser.add_argument("audios", nargs="+", type=Path, help="录音文件（m4a/wav/mp3/mp4 等）")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("./notes"))
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps"],
        help="推理设备：M4 上默认 cpu 最稳，mps 可尝试加速",
    )
    parser.add_argument(
        "--hotword",
        default="",
        help="热词，例如：'机器学习 20 人工智能 20'（词 权重）",
    )
    parser.add_argument(
        "--no-spk",
        action="store_true",
        help="不加载说话人分离模型 cam++，速度快一些",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="直接写入输出根目录（供已有集成兼容；默认按录音文件名前缀分目录）",
    )
    args = parser.parse_args()
    try:
        ensure_unique_stems([path for path in args.audios if path.exists()])
    except ValueError as error:
        parser.error(str(error))

    from funasr import AutoModel  # 延迟导入，便于先打印帮助信息

    model_kwargs = {
        "model": "paraformer-zh",          # 中文识别 + 时间戳
        "vad_model": "fsmn-vad",           # 语音活动检测（切分长音频）
        "punc_model": "ct-punc",           # 标点恢复
        "device": args.device,
        "hub": "ms",                       # 优先从 ModelScope 下载，国内更快
    }
    if not args.no_spk:
        model_kwargs["spk_model"] = "cam++"  # 说话人分离

    print(f"加载模型：{model_kwargs}")
    model = AutoModel(**model_kwargs)
    hotword = args.hotword or None

    with tempfile.TemporaryDirectory(prefix="funasr_") as tmp:
        for src in args.audios:
            if not src.exists():
                print(f"跳过不存在的文件：{src}", file=sys.stderr)
                continue
            stem = src.stem
            output_dir = recording_output_dir(
                args.output_dir, stem, flat_output=args.flat_output
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            wav_path = Path(tmp) / f"{stem}.wav"
            print(f"转换音频：{src} -> 16kHz 单声道 WAV")
            to_wav_16k(src, wav_path)

            print(f"开始转写：{src}")
            result = model.generate(
                input=str(wav_path),
                batch_size_s=300,          # 长音频按 5 分钟一批，控制内存
                hotword=hotword,
            )
            raw = result[0]
            render_outputs(stem, raw, output_dir)
            print(f"完成：{output_dir / (stem + '.md')}")


if __name__ == "__main__":
    main()
