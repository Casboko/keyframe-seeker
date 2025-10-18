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
