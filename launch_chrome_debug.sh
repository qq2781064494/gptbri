#!/usr/bin/env bash
set -euo pipefail

CHROME_BIN="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PROFILE_DIR="${HOME}/.chatgpt_bridge_chrome_profile"
DEBUG_PORT="${DEBUG_PORT:-9222}"
CHAT_URL="${1:-https://chatgpt.com/}"

if [[ ! -x "${CHROME_BIN}" ]]; then
  echo "未找到 Google Chrome：${CHROME_BIN}" >&2
  echo "请先安装系统版 Google Chrome。" >&2
  exit 1
fi

mkdir -p "${PROFILE_DIR}"

exec "${CHROME_BIN}" \
  --remote-debugging-port="${DEBUG_PORT}" \
  --user-data-dir="${PROFILE_DIR}" \
  "${CHAT_URL}"
