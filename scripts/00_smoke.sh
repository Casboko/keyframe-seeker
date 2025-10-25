#!/usr/bin/env bash
set -euo pipefail

python3 -c "import hydra, mlflow; print('hydra_mlflow_import_ok')"

python3 - <<'PY'
import mlflow

with mlflow.start_run():
    mlflow.log_param("ping", "pong")
    mlflow.log_metric("metric", 1.0)

print("mlflow_run_ok")
PY

python3 src/utils/check_config.py

if [[ -f "data/raw/big_buck_bunny_360p30.mp4" ]]; then
    echo "== ingest sampler smoke =="
    python3 scripts/01_ingest.py \
        --input data/raw/big_buck_bunny_360p30.mp4 \
        --fps 4 \
        --max-frames 12 \
        --overwrite \
        --no-contact-sheet
else
    echo "ingest sampler smoke skipped (data/raw/big_buck_bunny_360p30.mp4 not found)"
fi

echo "== video tooling check (non-fatal) =="
if command -v ffmpeg >/dev/null 2>&1; then
    ffmpeg -version | head -n1
else
    echo "ffmpeg: not found (host run? ok)"
fi
if command -v mkvmerge >/dev/null 2>&1; then
    mkvmerge -V | head -n1
else
    echo "mkvmerge: not found (host run? ok)"
fi
if command -v ffprobe >/dev/null 2>&1; then
    ffprobe -version | head -n1
else
    echo "ffprobe: not found (host run? ok)"
fi
