# Runpod Notes

- `pod_start.sh` は Pod 起動時に呼び出し、`nvidia-smi` / PyTorch / FAISS / PySceneDetect の存在確認を行います。
- Pod 上では以下の環境変数設定を想定しています（例）:
  - `PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`
  - `RUN_OUTPUT_DIR=/workspace/artifacts/runs`
- 永続化ボリューム例:
  - `/workspace/data` → Runpod Network Volume (input)
  - `/workspace/artifacts` → Runpod Network Volume (output/cache)
- Serverless 環境での環境変数と永続化ボリューム設定は Runpod 公式ドキュメントを参照してください。
