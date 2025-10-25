#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${RUNPOD_LOG_DIR:-/vol/logs/pod_start}
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/$(date +%Y%m%dT%H%M%S)_pre.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

REPO_DIR=${RUNPOD_REPO_DIR:-/workspace/keyframe-seeker}
if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[ERROR] REPO_DIR=${REPO_DIR} が存在しません" >&2
  exit 1
fi
cd "${REPO_DIR}"

ensure_apt_package() {
  local pkg="$1"
  if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
    NEED_APT_UPDATE=1
    PKG_LIST+=("${pkg}")
  fi
}

NEED_APT_UPDATE=0
PKG_LIST=()
ensure_apt_package make
ensure_apt_package git
ensure_apt_package python3-pip
ensure_apt_package python3-venv
if (( NEED_APT_UPDATE == 1 )); then
  echo "[INFO] installing missing apt packages: ${PKG_LIST[*]}"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y "${PKG_LIST[@]}"
fi

retry() {
  local max_attempts=${1:-3}
  local delay_seconds=${2:-10}
  shift 2
  local cmd=("$@")
  local attempt=1
  while true; do
    if "${cmd[@]}"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      echo "[ERROR] command failed after ${attempt} attempts: ${cmd[*]}" >&2
      return 1
    fi
    echo "[WARN] command failed (attempt ${attempt}); retrying in ${delay_seconds}s..." >&2
    sleep "${delay_seconds}"
    attempt=$((attempt + 1))
  done
}

export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/vol/.cache/pip}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/vol/.cache/uv}
mkdir -p "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"

echo "[INFO] running make install"
retry 3 10 make install
echo "[INFO] pre_start completed"
