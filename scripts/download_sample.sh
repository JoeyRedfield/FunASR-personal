#!/usr/bin/env bash
# 下载 FunASR 官方中文样例音频
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p samples
curl -L \
  https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav \
  -o samples/asr_example_zh.wav
echo "样例已保存到 samples/asr_example_zh.wav"
