# VSCode Codex Validation Checklist

## A. リポ同期と前提
- [ ] default branch 最新化、差分最小化
- [ ] 依存制約の整合：`cu126` / `PYTORCH_INDEX_URL` / `faiss-gpu-cu12==1.12.0` / `numpy<2`
- [ ] Dockerfile に ffmpeg / mkvtoolnix 追加済み（which/--version で確認）

## B. 仕様の確定要否
- [ ] 追加/変更ファイル一覧と目的が1:1対応している
- [ ] 設定（Hydraなど）のデフォルト値・上書き方法が明確
- [ ] エラー時の挙動（例外/戻り値/ログ）が定義済み
- [ ] I/O スキーマ（例：`timeline.json`, `scores.parquet`）が明示

## C. テスト観点
- [ ] Smoke（/vol/out のサンプルで scan→score→timeline→pack）
- [ ] 例外系（入力なし/破損動画/VFR境界ズレ）
- [ ] 性能とメモリ（フレームサンプリング密度/時間）
- [ ] 冪等性・再現性（同コマンドで同成果物）

## D. ドキュメント更新
- [ ] README への手順追記
- [ ] CHANGELOG 追記（Unreleased または Next）
- [ ] PRテンプレの各項目を満たせる根拠がある

## 結論
- [ ] 仕様は十分具体的 → 実装へGO
- [ ] 仕様の曖昧点あり → ChatGPTへ質問リスト返却（箇条書き）
