# Conventional Commits（要旨）
- 形式：`<type>(<scope>): <subject>`
- 主な type：feat / fix / chore / docs / refactor / test / ci / build
- BREAKING CHANGE は本文/フッター先頭に `BREAKING CHANGE:` を記述
- SemVer と連動：feat→MINOR, fix→PATCH, BREAKING→MAJOR

例：
feat(pack): add -force_key_frames option to align HLS segments
fix(score): normalize Laplacian variance on VFR inputs
chore(docker): add ffmpeg+mkvtoolnix to base image (cu126)
