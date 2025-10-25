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
SAMPLE_URL="https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4"

ensure_sample_video() {
  local src="$1"
  local dst="$2"
  local url="$3"

  mkdir -p "$(dirname "${dst}")"
  if [[ -f "${dst}" ]]; then
    echo "[INFO] sample video already present -> ${dst}"
    return 0
  fi

  if [[ -f "${src}" ]]; then
    cp -n "${src}" "${dst}"
    echo "[INFO] copied sample video to ${dst}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    echo "[INFO] downloading sample video from ${url}"
    if curl -L -o "${dst}" "${url}"; then
      echo "[INFO] downloaded sample video -> ${dst}"
      return 0
    fi
    echo "[WARN] failed to download sample video from ${url}" >&2
    rm -f "${dst}"
  else
    echo "[WARN] curl not available; cannot download sample video" >&2
  fi

  return 1
}

ensure_sample_video "${SAMPLE_SRC}" "${SAMPLE_DST}" "${SAMPLE_URL}" || \
  echo "[WARN] sample video is unavailable; smoke tests may fail."

if [[ -f "${SAMPLE_DST}" && ! -e "${SAMPLE_SRC}" ]]; then
  mkdir -p "$(dirname "${SAMPLE_SRC}")"
  if ln -s "${SAMPLE_DST}" "${SAMPLE_SRC}" 2>/dev/null; then
    echo "[INFO] symlinked sample video into repo -> ${SAMPLE_SRC}"
  else
    cp -n "${SAMPLE_DST}" "${SAMPLE_SRC}"
    echo "[INFO] copied sample video into repo -> ${SAMPLE_SRC}"
  fi
fi

echo "[INFO] running make smoke"
if make smoke; then
  echo "[INFO] post_start completed"
else
  echo "[FATAL] make smoke failed" >&2
  exit 2
fi
