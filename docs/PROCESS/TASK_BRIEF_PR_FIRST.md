# Task Brief (PR-first)

## タイトル（作業名）
例）chore: add ffmpeg+mkvtoolnix to base image (cu126)

## 背景 / 目的（Why now?）
- 何が困っているか
- どのユーザー体験 / パイプラインに影響するか

## 期待アウトカム / 計測
- 達成条件（定量）：
  - 例）`ffmpeg -version` / `mkvmerge -V` がコンテナ内で成功
  - 例）kseek v1.1のscan→packが/vol/outのサンプルで完走
- 非機能（性能/安定性/互換性など）

## スコープ / 非スコープ
- スコープ：
- 非スコープ：（将来拡張の駐車場）

## 変更概要（案）
- 触るディレクトリ/ファイルの列挙
- 主要関数/CLI/設定の変更点

## リスク / 代替案 / ロールバック
- 影響範囲（Dockerサイズ増/ビルド時間など）
- ロールバック手順（revert で戻す等）

## 受け入れ基準（Acceptance Criteria）
- [ ] 必須1
- [ ] 必須2

## 最小テスト計画（概要）
- コマンド例、期待ログ、エラー時の挙動

## PRメタ（初期値）
- ブランチ：`feat/...` or `chore/...`
- ラベル：`codex`, `size/S|M|L` など
- レビュワ：CODEOWNERSに従う
- Issue連携：必要なら `Closes #123` をPR本文に
