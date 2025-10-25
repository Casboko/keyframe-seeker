# Code Review Rules for Cloud Codex

## 重点観点
1. 正しさ（バグ/例外、I/O整合、既存制約：cu126, numpy<2, faiss-gpu-cu12==1.12.0）
2. 再現性・冪等性（同コマンド→同成果）
3. 性能・サイズ（Docker層追加の適否、キャッシュ/レイヤ最小化）
4. 安全性（権限/秘密情報/外部通信）
5. 可読性（命名/分割/コメント）
6. テスト（Smoke必須、例外系1ケース以上）
7. ドキュメント（README/CHANGELOG/PR本文の網羅）

## NG例
- テンプレ未充足（受け入れ基準・テスト欠落）
- 依存の暗黙変更（cu126→別系列 等）
- コミット規約違反（Conventional Commits）

## レビューフロー
- Draftの間は所見のみ → Ready後に**必須修正**判定
- 必要なら `request changes`（再実行コマンドも明記）
