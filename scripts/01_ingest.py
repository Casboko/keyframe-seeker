#!/usr/bin/env python3
"""Video sampler that extracts scene-aware keyframes at a fixed FPS.

This script detects shot/scene boundaries using PySceneDetect, trims a configurable
margin around each boundary, and samples frames at a fixed cadence (default: 4 fps)
within the remaining portion of each scene. Frames are written to the interim data
directory alongside per-frame metadata and a lightweight visual contact sheet to
help with quick validation.

Example:
    python scripts/01_ingest.py \
        --input data/raw/big_buck_bunny_360p30.mp4 \
        --fps 4 \
        --max-frames 48 \
        --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "OpenCV (cv2) が見つかりません。`make install` で依存をインストールしてください。"
    ) from exc

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from PIL import Image
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

LOG = logging.getLogger("ingest")
CONTACT_SHEET_LIMIT = 36
CONTACT_SHEET_COLUMNS = 6
CONTACT_SHEET_THUMB = (240, 240)
CONTACT_SHEET_MARGIN = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scene-aware frame sampler.")
    parser.add_argument(
        "--input",
        required=True,
        help="入力する動画ファイルへのパス（相対/絶対）。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=4.0,
        help="シーン内でのサンプリングレート [frames per second]（既定: 4.0）。",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=27.0,
        help="PySceneDetect ContentDetector の閾値（既定: 27.0）。",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.25,
        help="ショット境界から除外するマージン秒数（開始/終了に適用）。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="出力のベースディレクトリ（未指定時は Hydra の data.paths.interim を利用）。",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="出力フォルダ名に使う識別子（未指定の場合は動画ファイル名 stem）。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="保存するフレーム数の上限（テスト用 / None で制限なし）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の出力ディレクトリがある場合に削除して上書きします。",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="コンタクトシート画像の生成をスキップします。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを有効化します。",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config() -> OmegaConf:
    """Hydra 設定を読み込み、OmegaConf を返す。"""
    config_dir = Path(__file__).resolve().parent.parent / "conf"
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found: {config_dir}")

    # Hydra はシングルトンのため、既に初期化済みならリセットする。
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="ingest"):
        cfg = compose(config_name="config")
    return cfg


def resolve_data_paths(cfg: OmegaConf) -> Tuple[Path, Path]:
    """Hydra 設定から raw/interim のパスを取得し、存在しない場合はローカル fallback を返す。"""
    project_root = Path(__file__).resolve().parent.parent
    raw_default = project_root / "data" / "raw"
    interim_default = project_root / "data" / "interim"

    raw_path = Path(str(cfg.data.paths.raw))
    interim_path = Path(str(cfg.data.paths.interim))

    if not raw_path.exists():
        raw_path = raw_default
    if not interim_path.exists():
        interim_path = interim_default
        interim_path.mkdir(parents=True, exist_ok=True)

    return raw_path, interim_path


def resolve_video_path(input_arg: str, raw_root: Path) -> Path:
    """入力引数から動画パスを解決する。存在しない場合は raw_root 起点で再探索。"""
    candidate = Path(input_arg)
    if candidate.exists():
        return candidate.resolve()
    alt_candidate = raw_root / input_arg
    if alt_candidate.exists():
        return alt_candidate.resolve()
    raise FileNotFoundError(f"video not found: {input_arg}")


def detect_scenes(video_path: Path, threshold: float) -> Tuple[List[Tuple[float, float]], Optional[float]]:
    """PySceneDetect でシーン境界を取得し、秒単位の区間リストを返す。"""
    LOG.info("detecting scenes via PySceneDetect (threshold=%.2f)...", threshold)
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)

    scene_timecodes = scene_manager.get_scene_list()
    duration_sec: Optional[float] = None
    try:
        if video.duration is not None:
            duration_sec = video.duration.get_seconds()
    except AttributeError:
        duration_sec = None
    finally:
        release = getattr(video, "release", None)
        close = getattr(video, "close", None)
        if callable(release):
            release()
        elif callable(close):
            close()

    scenes_sec: List[Tuple[float, float]] = []
    for start_tc, end_tc in scene_timecodes:
        scenes_sec.append((start_tc.get_seconds(), end_tc.get_seconds()))

    if not scenes_sec and duration_sec is not None:
        scenes_sec.append((0.0, duration_sec))
        LOG.debug("no explicit scenes detected; using full duration %.3fs", duration_sec)

    LOG.info("detected %d scene(s)", len(scenes_sec))
    return scenes_sec, duration_sec


def build_trimmed_intervals(
    scenes: Sequence[Tuple[float, float]],
    margin: float,
) -> List[Dict[str, float]]:
    """ショット境界付近の margin を除外した利用可能区間を生成。"""
    intervals: List[Dict[str, float]] = []
    for idx, (start, end) in enumerate(scenes):
        trimmed_start = start + margin
        trimmed_end = end - margin
        if trimmed_end <= trimmed_start:
            LOG.debug(
                "scene %d too short after margin trimming (start=%.3f, end=%.3f, margin=%.3f) -> skipped",
                idx,
                start,
                end,
                margin,
            )
            continue
        interval = {
            "scene_index": idx,
            "start": trimmed_start,
            "end": trimmed_end,
            "duration": trimmed_end - trimmed_start,
        }
        intervals.append(interval)

    LOG.info("usable intervals after trimming: %d", len(intervals))
    return intervals


def build_sampling_schedule(
    intervals: Sequence[Dict[str, float]],
    fps: float,
    max_frames: Optional[int] = None,
) -> List[Dict[str, float]]:
    """サンプリングする時刻とシーン情報のスケジュールを生成。"""
    if fps <= 0:
        raise ValueError("fps must be positive.")

    schedule: List[Dict[str, float]] = []
    step = 1.0 / fps
    total = 0

    for interval in intervals:
        length = interval["end"] - interval["start"]
        num_samples = max(1, int(math.floor(length * fps)) + 1)
        for rel_idx in range(num_samples):
            sample_time = interval["start"] + rel_idx * step
            if sample_time > interval["end"] + 1e-6:
                break
            schedule.append(
                {
                    "time": sample_time,
                    "scene_index": interval["scene_index"],
                    "scene_start": interval["start"],
                    "scene_end": interval["end"],
                    "scene_relative_index": rel_idx,
                }
            )
            total += 1
            if max_frames is not None and total >= max_frames:
                LOG.debug("sampling schedule truncated at max_frames=%d", max_frames)
                return schedule

    return schedule


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            LOG.debug("removing existing directory: %s", output_dir)
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise FileExistsError(
                f"output directory {output_dir} already exists and is not empty. "
                "Use --overwrite to remove it."
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: Iterable[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            json.dump(record, fp, ensure_ascii=False)
            fp.write("\n")


def write_summary(summary: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)


def create_contact_sheet(image_paths: Sequence[Path], output_path: Path) -> Optional[Path]:
    if not image_paths:
        return None

    selected = list(image_paths[:CONTACT_SHEET_LIMIT])
    thumb_w, thumb_h = CONTACT_SHEET_THUMB
    columns = CONTACT_SHEET_COLUMNS
    margin = CONTACT_SHEET_MARGIN
    rows = math.ceil(len(selected) / columns)

    sheet_width = columns * thumb_w + (columns + 1) * margin
    sheet_height = rows * thumb_h + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(255, 255, 255))

    for idx, path in enumerate(selected):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h))
        except Exception as exc:  # pylint: disable=broad-except
            LOG.warning("failed to load %s for contact sheet (%s)", path, exc)
            continue
        col = idx % columns
        row = idx // columns
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + margin)
        sheet.paste(img, (x, y))

    sheet.save(output_path)
    LOG.info("contact sheet saved -> %s", output_path)
    return output_path


def sample_frames(
    video_path: Path,
    video_id: str,
    schedule: Sequence[Dict[str, float]],
    frames_dir: Path,
    fps: float,
    max_frames: Optional[int] = None,
) -> Tuple[List[Dict], List[Path], Dict[str, Optional[float]]]:
    """OpenCV を用いてスケジュールに沿ってフレームを保存する。"""
    if not schedule:
        return [], [], {}

    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video with OpenCV: {video_path}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    vfr_flag = src_fps <= 1e-3
    step = 1.0 / fps
    tolerance = step * 0.5

    for item in schedule:
        if not vfr_flag and src_fps > 0:
            item["frame_index_est"] = int(round(item["time"] * src_fps))
        else:
            item["frame_index_est"] = None

    metadata: List[Dict] = []
    saved_paths: List[Path] = []
    sample_idx = 0
    saved_count = 0
    frame_index = 0

    while cap.isOpened() and sample_idx < len(schedule):
        success, frame = cap.read()
        if not success:
            break

        if src_fps > 0:
            current_time = frame_index / src_fps
        else:
            current_time = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

        while sample_idx < len(schedule):
            target = schedule[sample_idx]
            target_time = target["time"]
            frame_est = target.get("frame_index_est")
            # 近傍に到達したか判定
            within_time = current_time + tolerance >= target_time
            within_frame = frame_est is None or frame_index >= frame_est - 1
            if not (within_time and within_frame):
                break

            frame_name = f"frame_{saved_count:06d}.jpg"
            frame_path = frames_dir / frame_name
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write frame to {frame_path}")

            record = {
                "video_id": video_id,
                "video_path": str(video_path),
                "frame_path": str(frame_path),
                "frame_index": frame_index,
                "capture_time_sec": float(target_time),
                "scene_index": target["scene_index"],
                "scene_frame_index": target["scene_relative_index"],
                "scene_start_sec": target["scene_start"],
                "scene_end_sec": target["scene_end"],
                "target_fps": fps,
                "source_fps": src_fps if src_fps > 0 else None,
                "source_frame_count": frame_count if frame_count > 0 else None,
                "width": width,
                "height": height,
                "vfr_detected": vfr_flag,
                "global_index": saved_count,
            }
            metadata.append(record)
            saved_paths.append(frame_path)

            saved_count += 1
            sample_idx += 1
            if max_frames is not None and saved_count >= max_frames:
                LOG.debug("stopped after reaching max_frames=%d", max_frames)
                cap.release()
                return metadata, saved_paths, {
                    "src_fps": src_fps if src_fps > 0 else None,
                    "frame_count": frame_count if frame_count > 0 else None,
                    "width": width,
                    "height": height,
                    "vfr_detected": vfr_flag,
                }

        frame_index += 1

    cap.release()

    if sample_idx < len(schedule):
        LOG.warning(
            "video ended before completing schedule (%d/%d frames saved)",
            sample_idx,
            len(schedule),
        )

    return metadata, saved_paths, {
        "src_fps": src_fps if src_fps > 0 else None,
        "frame_count": frame_count if frame_count > 0 else None,
        "width": width,
        "height": height,
        "vfr_detected": vfr_flag,
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    cfg = load_config()
    raw_root, interim_root = resolve_data_paths(cfg)

    video_path = resolve_video_path(args.input, raw_root)
    video_id = args.video_id or video_path.stem
    output_root = args.output_root or (interim_root / video_id)
    output_root = Path(output_root)
    frames_dir = output_root / "frames"

    prepare_output_dir(output_root, overwrite=args.overwrite)

    scenes, duration = detect_scenes(video_path, args.scene_threshold)
    intervals = build_trimmed_intervals(scenes, args.margin)
    if not intervals:
        LOG.warning("no usable intervals after applying margin=%.2f; nothing to do.", args.margin)
        return 0

    schedule = build_sampling_schedule(intervals, args.fps, args.max_frames)
    if not schedule:
        LOG.warning("sampling schedule is empty (fps=%.2f).", args.fps)
        return 0

    LOG.info("sampling %d frame(s) at %.2f fps (max=%s)", len(schedule), args.fps, args.max_frames or "∞")
    metadata, saved_paths, source_info = sample_frames(
        video_path=video_path,
        video_id=video_id,
        schedule=schedule,
        frames_dir=frames_dir,
        fps=args.fps,
        max_frames=args.max_frames,
    )

    if not metadata:
        LOG.warning("no frames extracted for %s", video_path)
        return 0

    run_ts = datetime.now(timezone.utc).isoformat()
    for record in metadata:
        record["ingest_timestamp"] = run_ts

    meta_path = output_root / "meta.jsonl"
    write_jsonl(metadata, meta_path)
    LOG.info("metadata -> %s", meta_path)

    summary = {
        "video_id": video_id,
        "video_path": str(video_path),
        "output_dir": str(output_root),
        "frames_dir": str(frames_dir),
        "frames_saved": len(metadata),
        "schedule_planned": len(schedule),
        "target_fps": args.fps,
        "scene_count": len(scenes),
        "interval_count": len(intervals),
        "scene_threshold": args.scene_threshold,
        "margin_sec": args.margin,
        "duration_sec": duration,
        "source_fps": source_info.get("src_fps"),
        "source_frame_count": source_info.get("frame_count"),
        "width": source_info.get("width"),
        "height": source_info.get("height"),
        "vfr_detected": source_info.get("vfr_detected"),
        "max_frames": args.max_frames,
        "ingested_at": run_ts,
    }
    summary_path = output_root / "summary.json"
    write_summary(summary, summary_path)
    LOG.info("summary -> %s", summary_path)

    if not args.no_contact_sheet:
        contact_path = output_root / "contact_sheet.jpg"
        create_contact_sheet(saved_paths, contact_path)

    LOG.info(
        "ingest complete: %d frame(s) saved to %s",
        len(metadata),
        frames_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
