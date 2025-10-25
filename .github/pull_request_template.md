## 背景 / 課題
- 何が問題で、誰が困る？

## 変更点（このPRがやること）
- 実装方針（採った/捨てた案）
- 触ったファイル：

## 受け入れ基準（チェックリスト）
- [ ] 主要コマンドが成功（例：ffmpeg/mkvmerge/ffprobe の検出）
- [ ] 既存のパイプライン（scan→score→timeline→pack）がサンプルで完走
- [ ] 依存制約（cu126 / numpy<2 / faiss-gpu-cu12==1.12.0）に抵触しない

## テスト計画 / 再現手順
```bash
# 最小Smoke（例）
docker build -f docker/Dockerfile -t ks:local \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 .
docker run --rm -it ks:local bash -lc \
  'ffmpeg -version | head -n1; mkvmerge -V | head -n1; which ffprobe'
```

## 影響範囲 / リスク & ロールバック

* 影響（ビルド時間、イメージサイズ等）
* ロールバック手順（revert / disable）

## ドキュメント更新

* README, CHANGELOG, SOP の更新有無

## 関連（任意）

* Closes #123  ← Issueを使う場合（自動クローズ）
