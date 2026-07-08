#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

export BRIDGE_HOST="${BRIDGE_HOST:-127.0.0.1}"
export BRIDGE_PORT="${BRIDGE_PORT:-8000}"
export BRIDGE_CDP_URL="${BRIDGE_CDP_URL:-http://127.0.0.1:9222}"
export BRIDGE_MODEL_NAME="${BRIDGE_MODEL_NAME:-chatgpt-web}"

if [[ -z "${BRIDGE_API_KEY:-}" ]]; then
  echo "提示：未设置 BRIDGE_API_KEY，本地服务将不校验 API key。" >&2
  echo "如需校验，请先执行：export BRIDGE_API_KEY=sk-local-test" >&2
fi

exec python llm_api_server.py
