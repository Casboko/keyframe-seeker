#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR=${1:-/vol/data/interim}
OUTPUT_DIR=${2:-/vol/artifacts}
STAMP=$(date +%Y%m%d_%H%M%S)

if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "[ERROR] target directory not found: ${TARGET_DIR}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
ARCHIVE_PATH="${OUTPUT_DIR}/interim_${STAMP}.tar.gz"

echo "[INFO] archiving ${TARGET_DIR} -> ${ARCHIVE_PATH}"
tar -C "${TARGET_DIR}" -czf "${ARCHIVE_PATH}" .
echo "${ARCHIVE_PATH}"
