# Cloud Codex Instruction

## Objective
- 一文で達成したいこと（例：ベースイメージに ffmpeg/mkvtoolnix を導入し、PRテンプレに検証手順を追加）

## Constraints / Policies
- cu126 ホイール（`PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`）
- `faiss-gpu-cu12==1.12.0` / `numpy<2`
- 既存のDocker分割・uv構成は維持
- コミットは Conventional Commits（feat/fix/chore...）

## Deliverables
- 変更ファイル一覧
- 動作確認ログ（コマンドと要約）
- PR（ドラフト→Ready）と本文（下記テンプレ適用）

## Plan of Attack（推奨手順）
1) ブランチ `chore/ffmpeg-mkv-capture-cu126` 作成
2) `docker/Dockerfile` に `ffmpeg mkvtoolnix` 追加（1RUNで）
3) `README.md` に GHCR Pull + 簡易Smokeを追記
4) （必要なら）`docker-compose.yml` の `PYTORCH_INDEX_URL` を最終確認
5) ローカルビルド→`ffmpeg -version` / `mkvmerge -V` / `ffprobe` 確認
6) コミット & プッシュ→**Draft PR** 作成→自動レビュー（AGENTS準拠）

## Commands (must run)
```bash
docker build -f docker/Dockerfile -t ks:local \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
docker run --rm -it ks:local bash -lc \
  'ffmpeg -version | head -n1; mkvmerge -V | head -n1; which ffprobe'
```

## PR Meta

* タイトル：`chore: add ffmpeg+mkvtoolnix to base image (cu126)`
* ラベル：`codex`, `size/S`, `area/docker`
* レビュワ：CODEOWNERS
* 本文：`.github/pull_request_template.md` 適用
