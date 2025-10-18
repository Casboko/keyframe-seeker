#!/usr/bin/env bash
set -euo pipefail

INPUT_VIDEO="${1:-data/raw/sample.mp4}"

if [[ ! -f "${INPUT_VIDEO}" ]]; then
  echo "ERROR: ${INPUT_VIDEO} が存在しません。テスト用の動画ファイルを data/raw/ に配置してください。" >&2
  exit 1
fi

scenedetect -i "${INPUT_VIDEO}" detect-content list-scenes
