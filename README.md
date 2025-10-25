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

- ネットワークボリュームは `/vol` にマウントする前提。テンプレートの Environment Variables には最低限 `RUN_DATA_DIR=/vol/data`、`RUN_ARTIFACTS_DIR=/vol/artifacts`、`RUN_OUTPUT_DIR=/vol/artifacts/runs`、`RUNPOD_LOG_DIR=/vol/logs`、`TZ=Asia/Tokyo` を追加する。
- Start Command（GPU Pod 用）の例:
  ```bash
  /bin/bash -lc '
  set -Eeuo pipefail

  if ! command -v git >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y git ca-certificates
  fi

  REPO_DIR=/workspace/keyframe-seeker
  if [[ -d "${REPO_DIR}/.git" ]]; then
    git -C "${REPO_DIR}" fetch --prune
    git -C "${REPO_DIR}" reset --hard origin/main
  else
    rm -rf "${REPO_DIR}"
    git clone https://github.com/casboko/keyframe-seeker.git "${REPO_DIR}"
  fi
  cd "${REPO_DIR}"
  git submodule update --init --recursive

  bash scripts/runpod/bootstrap.sh
  '
  ```
  ポート 8888 を公開すれば `https://<pod-id>-8888.proxy.runpod.net` から Jupyter にアクセスできる。
- Pod 起動時の自動初期化は `scripts/runpod/bootstrap.sh` → `pre_start.sh` → `post_start.sh` の順で行われる。`bootstrap.sh` が JupyterLab / jupyter-server-terminals / terminado / ipykernel などを最新化し、Python カーネルを `Python 3 (keyframe)` 名で登録すると同時に `allow_origin_pat` / `allow_remote_access` / `trust_xheaders` / `port_retries=0` をセットして Runpod プロキシ越しの CORS/XSRF を回避する。ログは `/vol/logs/pod_start/`、pip/uv キャッシュは `/vol/.cache/{pip,uv}` に保存される。
- `post_start.sh` はサンプル動画を `/vol/data/raw/` へコピーし、`make smoke` を実行して Hydra/LFS の疎通確認と ingest のスモークを行う。
- 成果物のダウンロードには、リポジトリ付属の `scripts/archive_interim.sh` を利用して `/vol/artifacts` にアーカイブを生成し、ローカルから `ssh -p <port> ... tar -C /vol/artifacts -czf - interim_*.tar.gz` → `tar.exe -C <local-dir> -xzf -` の手順で取得するのが簡単。
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
