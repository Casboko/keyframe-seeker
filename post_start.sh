#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${RUNPOD_LOG_DIR:-/vol/logs/pod_start}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%dT%H%M%S)_post.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

REPO_DIR=${RUNPOD_REPO_DIR:-/workspace/keyframe-seeker}
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR=${REPO_DIR} が存在しません" >&2
  exit 1
fi
cd "${REPO_DIR}"

SAMPLE_SRC="${REPO_DIR}/data/raw/big_buck_bunny_360p30.mp4"
SAMPLE_DST="${RUN_DATA_DIR:-/vol/data}/raw/big_buck_bunny_360p30.mp4"
if [[ -f "${SAMPLE_SRC}" && ! -f "${SAMPLE_DST}" ]]; then
  mkdir -p "$(dirname "${SAMPLE_DST}")"
  cp -n "${SAMPLE_SRC}" "${SAMPLE_DST}"
  echo "[INFO] copied sample video to ${SAMPLE_DST}"
fi

echo "[INFO] running make smoke"
if make smoke; then
  echo "[INFO] post_start completed"
else
  echo "[FATAL] make smoke failed" >&2
  exit 2
fi
