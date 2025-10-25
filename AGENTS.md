# Repository Process Overview

PR 駆動の 6 ステップフローを前提に、コミュニケーションテンプレとレビュー規則を統合しました。詳細は `docs/PROCESS/` 以下のテンプレート群を参照してください。

## プロセス概要
1. **タスク切り出し**: `docs/PROCESS/TASK_BRIEF_PR_FIRST.md` を使って背景・目的・受け入れ基準を整理。
2. **仕様確認**: `docs/PROCESS/VSCODE_CODEX_VALIDATION_CHECKLIST.md` に沿って要件の穴を塞ぐ。
3. **実装指示**: `docs/PROCESS/CLOUD_CODEX_INSTRUCTION_TEMPLATE.md` をベースに作業手順と必須コマンドを共有。
4. **自動レビュー**: `docs/PROCESS/AGENTS.md` の観点でドラフト PR を評価。
5. **Definition of Done**: `docs/PROCESS/DEFINITION_OF_DONE.md` を満たしたら Ready for review。
6. **マージ後フォロー**: `CHANGELOG.md` の Unreleased セクションを更新し、必要な SOP を同期。

## コミット / PR ポリシー
- **コミットメッセージ**: Conventional Commits (`docs/PROCESS/COMMIT_CONVENTION.md`)。`<type>(<scope>): <subject>` を基本形とし、BREAKING CHANGE は本文で明示。
- **PR テンプレ**: `.github/pull_request_template.md` をデフォルト適用。軽微な修正用に `.github/PULL_REQUEST_TEMPLATE/quick-fix.md` を用意。
- **Issue 連携**: `Closes #123` を本文に記述するとデフォルトブランチマージ時に自動クローズ。
- **CODEOWNERS**: `CODEOWNERS` に記載した領域別オーナーへ Ready 時にレビュー依頼が飛ぶ。メンテナの GitHub ID に置き換えて運用する。

## ビルド・テストコマンド
- `make deps-base`: 依存追加・更新時に `constraints-base.txt` を再生成。
- `make install`: CUDA 12.6 ホイールの PyTorch / FAISS を再構築（`PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`）。
- `make smoke`: Hydra + MLflow の最小スモーク (`scripts/00_smoke.sh`)。
- `bash scripts/01_ingest.sh data/raw/big_buck_bunny_360p30.mp4`: PySceneDetect スモーク。
- `python scripts/02_sample_embed.py`: OpenCLIP + FAISS のサンプル埋め込み動作確認。

## コーディング / ドキュメント方針
- Python は 4 スペース + 型ヒント推奨。Black(88) / isort / Ruff / Yamllint を `pre-commit` で実行。
- Hydra 設定は `conf/` 配下のグループ構造を保ちつつ差し替え。
- 大規模データは DVC 管理、MLflow ログは当面ローカル (`mlruns/`)。
- Runpod 固有設定は `runpod/` 配下でテンプレート化し、秘密情報は共有しない。

## 参考ドキュメント
- ランタイム決定: `docs/DECISIONS.md`
- PR/レビュー基準: `docs/PROCESS/AGENTS.md`, `docs/PROCESS/DEFINITION_OF_DONE.md`
- コミット規約: `docs/PROCESS/COMMIT_CONVENTION.md`
- 変更履歴: `CHANGELOG.md`
