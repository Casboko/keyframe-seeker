#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR=${RUNPOD_REPO_DIR:-/workspace/keyframe-seeker}
LOG_DIR=${RUNPOD_LOG_DIR:-/vol/logs/pod_start}
mkdir -p "${LOG_DIR}"

export PIP_BREAK_SYSTEM_PACKAGES=1
export PIP_NO_INPUT=1
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-/vol/.cache/pip}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/vol/.cache/uv}
mkdir -p "${PIP_CACHE_DIR}" "${UV_CACHE_DIR}"
export SHELL=${SHELL:-/bin/bash}

if [[ ! -x "${REPO_DIR}/pre_start.sh" ]]; then
  echo "[ERROR] ${REPO_DIR}/pre_start.sh が見つかりません" >&2
  exit 1
fi

RUNPOD_REPO_DIR="${REPO_DIR}" "${REPO_DIR}/pre_start.sh"

if [[ -x "${REPO_DIR}/post_start.sh" ]]; then
  if ! RUNPOD_REPO_DIR="${REPO_DIR}" "${REPO_DIR}/post_start.sh"; then
    echo "[WARN] post_start.sh failed (see logs)" >&2
  fi
fi

python3 -m pip install --upgrade pip
pip install --upgrade \
  jupyterlab \
  jupyter-server-terminals \
  terminado \
  ipykernel \
  notebook-shim \
  nbformat \
  nbclient

python3 -m ipykernel install --user --name keyframe --display-name "Python 3 (keyframe)"

JUPYTER_PORT=${JUPYTER_PORT:-8888}
echo "[INFO] starting jupyter lab on port ${JUPYTER_PORT}"
jupyter lab \
  --no-browser \
  --ip=0.0.0.0 \
  --port="${JUPYTER_PORT}" \
  --allow-root \
  --ServerApp.root_dir="${REPO_DIR}" \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.port_retries=0 \
  --ServerApp.allow_remote_access=True \
  --ServerApp.trust_xheaders=True \
  --ServerApp.allow_origin_pat='^https://[a-z0-9-]+-[0-9]+\.proxy\.runpod\.net$' \
  --ServerApp.terminals_enabled=True \
  --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' \
  > /vol/logs/jupyter.log 2>&1 &

touch /vol/logs/jupyter_ready
exec tail -f /dev/null
