from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"
DEFAULT_VQVAE_CKPT = PROJECT_ROOT / "ckpt" / "vqvae_large_im5_uncond_bsq32.pth"

from fractal3d import (  # noqa: E402
  OctreeConfig,
  ShapeOctreeDataset,
  collate_shapes,
)
from fractal3d.config import parse_args_with_config  # noqa: E402
from fractal3d.octgpt_vqvae import encode_bsq_tokens, load_octgpt_vqvae  # noqa: E402


def batch_uid(batch: dict[str, object]) -> str:
  uid = batch["uid"]
  if isinstance(uid, (list, tuple)):
    return str(uid[0])
  return str(uid)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Cache frozen OctGPT VQVAE BSQ tokens for 3DFractalGen VQ training.")
  parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
  parser.add_argument("--filelist", default=None)
  parser.add_argument("--output-dir", default="outputs/vq_cache")
  parser.add_argument("--vqvae-ckpt", default=str(DEFAULT_VQVAE_CKPT))

  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--depth-stop", type=int, default=6)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)
  parser.add_argument("--sample-seed", type=int, default=0)
  parser.add_argument("--dim", type=int, default=128)
  parser.add_argument("--num-vq-embed", type=int, default=32)
  parser.add_argument("--vq-groups", type=int, default=32)

  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--overwrite", action="store_true")
  return parse_args_with_config(parser)


def main() -> None:
  args = parse_args()
  if args.depth_stop > args.depth:
    raise ValueError("--depth-stop must be <= --depth")

  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  config = OctreeConfig(
    depth=args.depth,
    full_depth=args.full_depth,
    points_scale=args.points_scale,
    max_points=args.max_points,
    sample_seed=args.sample_seed,
  )
  dataset = ShapeOctreeDataset(args.data, config=config, filelist=args.filelist)
  loader_kwargs = {
    "num_workers": args.num_workers,
    "collate_fn": collate_shapes,
    "pin_memory": str(args.device).startswith("cuda"),
  }
  if args.num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2
  loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    **loader_kwargs,
  )

  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)
  cached = 0
  skipped = 0
  for batch in tqdm(loader, desc="cache vq tokens", dynamic_ncols=True):
    uid = batch_uid(batch)
    path = output_dir / f"{uid}.pt"
    if path.exists() and not args.overwrite:
      skipped += 1
      continue

    octree = batch["octree_gt"].to(device)
    with torch.no_grad():
      indices, codes, code_depth = encode_bsq_tokens(vqvae, octree)
    if code_depth != args.depth_stop:
      raise ValueError(
        f"{uid}: VQVAE code depth is {code_depth}, expected {args.depth_stop}.")

    tmp_path = path.with_suffix(".tmp")
    torch.save({
      "uid": uid,
      "indices": indices.cpu(),
      "codes": codes.cpu(),
      "code_depth": int(code_depth),
      "depth": args.depth,
      "full_depth": args.full_depth,
      "depth_stop": args.depth_stop,
      "points_scale": args.points_scale,
      "max_points": args.max_points,
      "sample_seed": args.sample_seed,
      "dim": args.dim,
      "num_vq_embed": args.num_vq_embed,
      "vq_groups": args.vq_groups,
    }, tmp_path)
    tmp_path.replace(path)
    cached += 1

  print(f"cached={cached} skipped={skipped} output_dir={output_dir}")


if __name__ == "__main__":
  main()
