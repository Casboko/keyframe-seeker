#!/usr/bin/env python3
"""Generate OpenCLIP embeddings for sampled video frames and build a FAISS index.

This script consumes the outputs produced by ``scripts/01_ingest.py`` (frames and
metadata under ``data/interim/<video-id>/`` by default), encodes each frame with
OpenCLIP, L2 正規化した特徴量を保存しつつ、FAISS IndexFlatIP を構築します。
最小構成として ``artifacts/embeddings`` に ``*.npy``、``artifacts/faiss`` に
``*.faiss`` を書き出します。

Typical usage::

    python scripts/02_sample_embed.py --video-id big_buck_bunny_360p30 \
        --batch-size 128 --precision fp16 --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "FAISS (faiss-gpu-cu12) がインポートできません。`make install` 実行後に再試行してください。"
    ) from exc

try:
    import open_clip  # type: ignore
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "open-clip-torch が見つかりません。`make install` で依存をインストールしてください。"
    ) from exc

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, errors as oc_errors
from PIL import Image
from torch.utils.data import DataLoader, Dataset

LOG = logging.getLogger("embed")


@dataclass
class EmbedRecord:
    """Case class for describing one embedding entry."""

    frame_path: str
    frame_index: Optional[int]
    capture_time_sec: Optional[float]
    scene_index: Optional[int]
    embedding_offset: int


class FrameDataset(Dataset):
    """Dataset that loads RGB frames and applies the OpenCLIP preprocess transform."""

    def __init__(self, frame_paths: Sequence[Path], preprocess):
        self._frame_paths: List[Path] = list(frame_paths)
        self._preprocess = preprocess

    def __len__(self) -> int:
        return len(self._frame_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self._frame_paths[idx]
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(f"failed to load frame: {path}") from exc
        tensor = self._preprocess(rgb)
        return tensor, str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenCLIP embedding + FAISS index builder.")
    parser.add_argument(
        "--video-id",
        type=str,
        help="対象動画の ID（既定では data/interim/<video-id>/frames を探索）。",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="フレーム画像が格納されたディレクトリ。指定時は video-id より優先。",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        help="ingest が生成した meta.jsonl へのパス（省略時は frames-dir から推測）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="OpenCLIP 推論時のバッチサイズ（既定: 128）。",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
        help="AMP の精度指定。CUDA 利用時のみ有効（既定: fp16）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader のワーカ数（既定: 2）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="OpenCLIP 推論で利用するデバイス（既定: CUDA が利用可能なら cuda、さもなくば cpu）。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="結果を書き出す artifacts ルート。未指定時は Hydra 設定 / ローカル既定を利用。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="デバッグ用に処理するフレーム数の上限を指定。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の埋め込み・インデックスファイルが存在する場合に上書きします。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細ログを出力します。",
    )
    return parser.parse_args()


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config() -> OmegaConf:
    """Hydra 設定を読み込む（ingest と同様のパス解決）。"""
    config_dir = Path(__file__).resolve().parent.parent / "conf"
    if not config_dir.exists():
        raise FileNotFoundError(f"config directory not found: {config_dir}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base=None, job_name="embed"):
        cfg = compose(config_name="config")
    return cfg


def resolve_data_paths(cfg: OmegaConf) -> Tuple[Path, Path]:
    """Hydra の data.paths とローカルデフォルトを突き合わせ、raw/interim を返す。"""
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


def resolve_output_paths(cfg: OmegaConf, output_root_arg: Optional[Path]) -> Tuple[Path, Path]:
    """artifacts/embeddings, artifacts/faiss のディレクトリを返す。"""
    if output_root_arg is not None:
        root = output_root_arg
    else:
        artifacts_root: Optional[Path] = None
        try:
            artifacts_root = Path(str(cfg.data.artifacts.root))
        except (AttributeError, oc_errors.OmegaConfBaseException, ValueError):
            artifacts_root = None

        if artifacts_root is None or str(artifacts_root).strip() == "":
            env_root = os.environ.get("RUN_ARTIFACTS_DIR")
            if env_root:
                artifacts_root = Path(env_root)

        if artifacts_root is None:
            artifacts_root = Path(__file__).resolve().parent.parent / "artifacts"

        root = artifacts_root

    root = root.expanduser()
    embeddings_dir = root / "embeddings"
    faiss_dir = root / "faiss"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    faiss_dir.mkdir(parents=True, exist_ok=True)
    return embeddings_dir, faiss_dir


def collect_frame_paths(frames_dir: Path, limit: Optional[int] = None) -> List[Path]:
    """frames ディレクトリからフレーム画像を収集する。"""
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"no JPG frames found in {frames_dir}")
    if limit is not None:
        frame_paths = frame_paths[:limit]
    return frame_paths


def load_meta(meta_path: Path) -> Dict[str, Dict]:
    """meta.jsonl を読み込み、frame_path をキーにした辞書へ変換する。"""
    if not meta_path.exists():
        LOG.warning("meta.jsonl が見つかりませんでした -> %s", meta_path)
        return {}

    records: Dict[str, Dict] = {}
    with meta_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOG.warning("meta.jsonl のパースに失敗しました（無視します）: %s", line)
                continue
            key_variants = {
                record.get("frame_path"),
                str(Path(record.get("frame_path", "")).resolve()),
                Path(record.get("frame_path", "")).name,
            }
            for key in key_variants:
                if key:
                    records[key] = record
    return records


def ensure_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")


def resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_amp_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def build_embeddings(
    frame_paths: Sequence[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    precision: str,
) -> Tuple[np.ndarray, List[str]]:
    """Encode frames into OpenCLIP embeddings."""
    dataset = FrameDataset(frame_paths, preprocess)

    def collate_fn(batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
        images = torch.stack([item[0] for item in batch], dim=0)
        paths = [item[1] for item in batch]
        return images, paths

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers, 0),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    amp_dtype = select_amp_dtype(precision)
    context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype and device.type == "cuda" else nullcontext()
    )

    model.eval()
    model.to(device)

    embed_dim: Optional[int] = None
    features = []
    total = len(dataset)
    processed = 0

    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            with context:
                outputs = model.encode_image(images)

            if embed_dim is None:
                embed_dim = outputs.shape[-1]
                LOG.debug("embedding dimension detected: %d", embed_dim)

            outputs = outputs / outputs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            feats = outputs.detach().float().cpu().numpy()
            features.append(feats)
            processed += feats.shape[0]

            LOG.debug(
                "batch %d: processed %d/%d frames (batch=%d)",
                batch_idx,
                processed,
                total,
                feats.shape[0],
            )

    if not features:
        raise RuntimeError("no embeddings were generated; dataset may be empty.")

    embeddings = np.concatenate(features, axis=0)
    return embeddings, [str(p) for p in frame_paths]


def build_faiss_index(
    embeddings: np.ndarray,
    use_gpu: bool,
    prefer_float16: bool,
) -> faiss.Index:
    """Construct IndexFlatIP (optionally on GPU) from embeddings."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be 2-D (num_samples, dim)")

    num_frames, dim = embeddings.shape
    LOG.info("building FAISS index (num=%d, dim=%d)...", num_frames, dim)

    index = faiss.IndexFlatIP(dim)
    resources = None
    gpu_index = None

    if use_gpu and faiss.get_num_gpus() > 0:
        resources = faiss.StandardGpuResources()
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = prefer_float16
        try:
            gpu_index = faiss.index_cpu_to_gpu(resources, 0, index, cloner)
        except Exception as exc:  # pylint: disable=broad-except
            LOG.warning("failed to initialise FAISS GPU index (%s). Falling back to CPU.", exc)
            gpu_index = None

    target_index = gpu_index if gpu_index is not None else index
    target_index.add(embeddings)

    if gpu_index is not None:
        LOG.info("transferring GPU index back to CPU for persistence.")
        return faiss.index_gpu_to_cpu(gpu_index)
    return index


def load_model_and_preprocess(cfg: OmegaConf, device: torch.device):
    model_name = cfg.embed.model
    pretrained = cfg.embed.pretrained
    LOG.info("loading OpenCLIP model=%s pretrained=%s", model_name, pretrained)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.to(device)
    return model, preprocess


def prepare_manifest(frame_paths: Sequence[Path], meta: Dict[str, Dict]) -> List[EmbedRecord]:
    manifest: List[EmbedRecord] = []
    for offset, frame_path in enumerate(frame_paths):
        meta_key_candidates = [
            str(frame_path),
            str(frame_path.resolve()),
            frame_path.name,
        ]
        record: Optional[Dict] = None
        for key in meta_key_candidates:
            if key in meta:
                record = meta[key]
                break
        manifest.append(
            EmbedRecord(
                frame_path=str(frame_path),
                frame_index=(record or {}).get("frame_index"),
                capture_time_sec=(record or {}).get("capture_time_sec"),
                scene_index=(record or {}).get("scene_index"),
                embedding_offset=offset,
            )
        )
    return manifest


def write_manifest(manifest_path: Path, manifest: Iterable[EmbedRecord]) -> None:
    data = [
        {
            "frame_path": record.frame_path,
            "frame_index": record.frame_index,
            "capture_time_sec": record.capture_time_sec,
            "scene_index": record.scene_index,
            "embedding_offset": record.embedding_offset,
        }
        for record in manifest
    ]
    with manifest_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    LOG.info("manifest -> %s", manifest_path)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    cfg = load_config()
    _raw_root, interim_root = resolve_data_paths(cfg)

    if args.frames_dir:
        frames_dir = args.frames_dir.expanduser().resolve()
    else:
        if not args.video_id:
            raise ValueError("video-id または frames-dir のいずれかを指定してください。")
        frames_dir = (interim_root / args.video_id / "frames").resolve()

    frame_paths = collect_frame_paths(frames_dir, limit=args.max_frames)
    LOG.info("frames collected: %d (dir=%s)", len(frame_paths), frames_dir)

    if args.meta_path:
        meta_path = args.meta_path.expanduser().resolve()
    else:
        meta_path = frames_dir.parent / "meta.jsonl"
    meta_records = load_meta(meta_path)

    embeddings_dir, faiss_dir = resolve_output_paths(cfg, args.output_root)

    video_token = args.video_id or frames_dir.parent.name
    embeddings_path = embeddings_dir / f"{video_token}.npy"
    manifest_path = embeddings_dir / f"{video_token}_manifest.json"
    faiss_path = faiss_dir / f"{video_token}.faiss"

    for path in (embeddings_path, manifest_path, faiss_path):
        ensure_overwrite(path, args.overwrite)

    device = resolve_device(args.device)
    LOG.info("using device: %s", device)

    model, preprocess = load_model_and_preprocess(cfg, device)

    embeddings, ordered_frames = build_embeddings(
        frame_paths=frame_paths,
        model=model,
        preprocess=preprocess,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        precision=args.precision,
    )

    expected_dim = int(cfg.index.dimension)
    if embeddings.shape[1] != expected_dim:
        raise RuntimeError(
            f"embedding dimension mismatch (expected {expected_dim}, got {embeddings.shape[1]}). "
            "Hydra 設定の index.dimension を確認してください。"
        )

    LOG.info("writing embeddings -> %s", embeddings_path)
    np.save(embeddings_path, embeddings)

    manifest = prepare_manifest([Path(p) for p in ordered_frames], meta_records)
    write_manifest(manifest_path, manifest)

    prefer_float16 = args.precision in {"fp16", "bf16"}
    use_gpu_index = bool(cfg.index.get("use_gpu", True))
    index = build_faiss_index(
        embeddings,
        use_gpu=use_gpu_index,
        prefer_float16=prefer_float16,
    )

    LOG.info("writing FAISS index -> %s", faiss_path)
    faiss.write_index(index, str(faiss_path))

    LOG.info(
        "embed complete: frames=%d saved_embeddings=%s index=%s",
        embeddings.shape[0],
        embeddings_path,
        faiss_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
