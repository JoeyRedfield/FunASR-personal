#!/usr/bin/env bash
# 启动 FunASR OpenAI 兼容转写 API
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d ".venv" ]; then
  echo "未找到 .venv，请先执行：uv venv .venv --python 3.12 && uv pip install -r requirements.txt"
  exit 1
fi

source .venv/bin/activate
exec funasr-server \
  --model "${FUNASR_MODEL:-paraformer}" \
  --device "${FUNASR_DEVICE:-cpu}" \
  --host "${FUNASR_HOST:-127.0.0.1}" \
  --port "${FUNASR_PORT:-8000}"
