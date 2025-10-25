# Keyframe Seeker (Bootstrap)

このリポジトリは Runpod 上でのキーフレーム抽出パイプライン構築を想定した初期雛形です。現時点で確定しているランタイム方針や開発フローは `docs/DECISIONS.md` と SOP v1.5 を参照してください。

## 現状の進捗メモ

- CUDA / PyTorch / FAISS / ffmpeg の基準値を `docs/DECISIONS.md` に記録済み。
- Hydra 設定の defaults 骨格と MLflow スモーク用スクリプト（`scripts/00_smoke.sh`）を追加。
- PySceneDetect と OpenCLIP + FAISS の最小スモーク（`scripts/01_ingest.sh`, `scripts/02_sample_embed.py`）を用意。依存解決の都合で `numpy==1.26.4` / `scipy==1.11.4` / `tifffile==2024.8.30` / `contourpy==1.2.1` / `opencv-python-headless==4.8.1.78` を採用。
- Dockerfile は `nvidia/cuda:12.8.1-runtime-ubuntu22.04` をベースに依存をインストールする構成に更新。
- `requirements-base.in` / `constraints-base.txt` による二段ロックと `make deps-base` / `make install` フローを整備。

## PR 駆動運用メモ

- タスク切り出しからレビューまでの 6 ステップテンプレートは `docs/PROCESS/` を参照。
- PR 本文テンプレートは `.github/pull_request_template.md`（軽微修正は `.github/PULL_REQUEST_TEMPLATE/quick-fix.md`）。
- コミット規約は `docs/PROCESS/COMMIT_CONVENTION.md`、レビュー観点は `docs/PROCESS/AGENTS.md`、DoD は `docs/PROCESS/DEFINITION_OF_DONE.md`。
- 変更履歴は `CHANGELOG.md` の Keep a Changelog 形式で Unreleased セクションから更新する。

## サンプルデータ

- `data/raw/big_buck_bunny_360p30.mp4` を同梱。出典: Big Buck Bunny (Blender Foundation) CC-BY 3.0, https://peach.blender.org/download/
- 取得コマンド例:
  ```bash
  curl -L -o data/raw/big_buck_bunny_360p30.mp4 \
    https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4
  ```
- `PATH=.venv_scen/bin:$PATH bash scripts/01_ingest.sh data/raw/big_buck_bunny_360p30.mp4` で PySceneDetect スモークを実行済み（scenedetect 0.6.7.1, opencv-python 4.12.0.88）。

## 次のステップ

1. 依存を追加・更新した場合は `make deps-base` で `constraints-base.txt` を再生成し、`make install` で `--index-url` を使って PyTorch/cu126 を取得しつつ GPU 版 FAISS を導入する（※ `numpy<2` / `scipy<1.12` / `tifffile==2024.8.30` / `contourpy==1.2.1` / `opencv-python-headless==4.8.1.78` を維持し、FAISS の `numpy<2` 制約と両立させる）。
2. `faiss-gpu-cu12` の安定入手可否を定期確認し、必要に応じて `[fix_cuda]` フォールバックや代替手段を README に追記する。
3. Runpod Pod / Serverless での環境変数・永続化設定を整理し、実際の `runpod/pod_start.sh` 実行ログを「Runpod 運用メモ」に反映する。

## Runpod 運用メモ

- ネットワークボリュームは `/vol` にマウントする前提。テンプレートで以下の ENV を指定すると Hydra 設定と揃います。  
  `RUN_DATA_DIR=/vol/data` / `RUN_ARTIFACTS_DIR=/vol/artifacts` / `RUN_MLFLOW_DIR=/vol/artifacts/mlruns` / `RUNPOD_REPO_DIR=/workspace`。必要に応じて `RUNPOD_LOG_DIR=/vol/logs` も上書きしてください。
- Pod 起動時の自動初期化は `pre_start.sh` / `post_start.sh` が担当します。Runpod 公式イメージの `/start.sh` から自動的に呼ばれるため、テンプレートに両ファイルを含めるだけで `make install`（リトライ付き）→`make smoke` が実行されます。ログは `/vol/logs/pod_start/` に保存されます。
- キャッシュは永続ボリューム側に配置されます（`PIP_CACHE_DIR=/vol/.cache/pip` / `UV_CACHE_DIR=/vol/.cache/uv`）。Pod 再生成時も pip/uv が速く復帰します。
- Pod 起動後は必要に応じて `runpod/pod_start.sh` を手動で呼び、`nvidia-smi` / PyTorch / FAISS / PySceneDetect の実行結果を確認してください。
- Pod 上での確認結果（サンプル）:
  ```
  ==== NVIDIA SMI ====
  NVIDIA-SMI 535.161.07 ...
  ==== PyTorch / FAISS / Python ====
  python_version: 3.11.x
  torch_available: True
  torch_version: 2.7.1+cu126
  cuda_available: True
  faiss_available: True
  faiss_version: 1.8.0
  ==== PySceneDetect ====
  PySceneDetect 0.6.7
  ```

## Tooling check (container)

```bash
# Pull from GHCR (image name synced with build workflow)
docker pull ghcr.io/casboko/keyframe-seeker/keyframe-seeker:cu126

# Run a one-shot check inside the container
docker run --rm -it ghcr.io/casboko/keyframe-seeker/keyframe-seeker:cu126 bash -lc \
  'ffmpeg -version | head -n1; mkvmerge -V | head -n1; which ffprobe || true'
```

> Note: build 時の PyTorch index は `pyproject.toml` の uv index で cu126 に固定済み。`--build-arg` 指定は不要です。
