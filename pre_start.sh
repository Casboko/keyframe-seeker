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

if [[ ! -f conf/data/default.yaml ]]; then
  echo "[INFO] conf/data/default.yaml not found; creating default mapping"
  mkdir -p conf/data
  cat > conf/data/default.yaml <<'EOF'
root: ${oc.env:RUN_DATA_DIR, /vol/data}
paths:
  raw: ${data.root}/raw
  interim: ${data.root}/interim
  processed: ${data.root}/processed

artifacts:
  root: ${oc.env:RUN_ARTIFACTS_DIR, /vol/artifacts}
  mlruns: ${oc.env:RUN_MLFLOW_DIR, ${data.artifacts.root}/mlruns}
EOF
fi

python3 - <<'PY'
from pathlib import Path

config_path = Path("conf/config.yaml")
if config_path.exists():
    lines = config_path.read_text().splitlines()
    needle = "  - _self_"
    if needle.strip() not in {line.strip() for line in lines}:
        updated = []
        inserted = False
        for line in lines:
            updated.append(line)
            if line.strip() == "- runpod: default" and not inserted:
                updated.append(needle)
                inserted = True
        if not inserted:
            updated.append(needle)
        config_path.write_text("\n".join(updated) + "\n")
PY

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
export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_NO_INPUT=1

DATA_ROOT=${RUN_DATA_DIR:-/vol/data}
ARTIFACT_ROOT=${RUN_ARTIFACTS_DIR:-/vol/artifacts}
mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/interim" "${DATA_ROOT}/processed" "${ARTIFACT_ROOT}" "${RUNPOD_LOG_DIR:-/vol/logs/pod_start}"

echo "[INFO] running make install"
retry 3 10 make install
echo "[INFO] pre_start completed"
