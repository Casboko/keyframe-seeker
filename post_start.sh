#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${RUNPOD_LOG_DIR:-/vol/logs/pod_start}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%dT%H%M%S)_post.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

REPO_DIR=${RUNPOD_REPO_DIR:-/workspace}
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR=${REPO_DIR} が存在しません" >&2
  exit 1
fi
cd "${REPO_DIR}"

echo "[INFO] running make smoke"
if make smoke; then
  echo "[INFO] post_start completed"
else
  echo "[FATAL] make smoke failed" >&2
  exit 2
fi
