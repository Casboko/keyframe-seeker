# Decisions (Runtime Baseline)

- **CUDA**: 12.8 系（Runpod の Pod が 12.8 主流のため）
  - Docker base: `nvidia/cuda:12.8.1-devel-ubuntu22.04` を採用。
- **PyTorch**: 2.7.x（安定線） + **CUDA 12.8** 対応
  - PyTorch 公式「Get Started / Previous Versions」で Compute Platform に **CUDA 12.8** を指定し、案内される pip コマンドを利用する。
  - 例: `--index-url https://download.pytorch.org/whl/cu128`（実際のコマンドは公式サイトの最新版を参照して貼り付けること）。
- **FAISS GPU**: `faiss-gpu-cu12`（CUDA 12.* 系ホイールを採用）。
- **動画分割要件**: **ffmpeg は必須**（PySceneDetect は ffmpeg / mkvmerge を要求）。
- **設定管理**: Hydra の `defaults` リスト方式でコンフィググループを差し替える運用。
- **実験ログ**: MLflow Tracking を利用（初期は `mlruns/` ローカル、将来的にリモート先へ移行）。
- **依存ロック**: `requirements-base.in` を `pip-compile` して `constraints-base.txt` を生成し、`make install` で cu128 指定の Torch 群と `faiss-gpu-cu12`（必要なら `[fix_cuda]`）を導入する二段方式。FAISS の `numpy<2` 制約に合わせ、`numpy==1.26.4` / `scipy==1.11.4` / `tifffile==2024.8.30` / `contourpy==1.2.1` / `opencv-python-headless==4.8.1.78` をピン止め。
- **Runpod 運用**: Pod 起動時の ENV / 永続化は Runpod ドキュメントの Pods / Serverless ガイドラインに準拠。
