from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from fractal3d.octgpt_vqvae import load_octgpt_vqvae  # noqa: E402
from sample_vae import (  # noqa: E402
  init_generated_octree,
  sdf_surface_points,
  write_ply,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Decode saved VAE sample .pt files into surface PLY files.")
  parser.add_argument("--sample", default=None, help="Single *_sample.pt file.")
  parser.add_argument("--sample-dir", default=None, help="Directory of *_sample.pt files.")
  parser.add_argument("--vqvae-ckpt", default="ckpt/vqvae_large_im5_uncond_bsq32.pth")
  parser.add_argument("--output-dir", default=None)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument(
    "--vqvae-update-octree",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def sample_paths(args: argparse.Namespace) -> list[Path]:
  if args.sample is not None:
    return [Path(args.sample)]
  if args.sample_dir is None:
    raise ValueError("Pass either --sample or --sample-dir.")
  return sorted(Path(args.sample_dir).glob("*_sample.pt"))


def reconstruct_octree(sample: dict, full_depth: int, code_depth: int, device: torch.device):
  octree = init_generated_octree(full_depth, code_depth, device)
  split_by_depth = sample["split_by_depth"]
  for depth in sorted(int(d) for d in split_by_depth):
    split = split_by_depth[depth].to(device).int()
    octree.octree_split(split, depth)
    if depth < code_depth:
      octree.octree_grow(depth + 1, update_neigh=True)
  return octree


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  paths = sample_paths(args)
  if not paths:
    raise FileNotFoundError("No *_sample.pt files found.")

  if args.output_dir is None:
    output_dir = paths[0].parent
  else:
    output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)
  for path in paths:
    sample = torch.load(path, map_location="cpu", weights_only=False)
    code_depth = int(sample.get("code_depth", 6))
    decode_depth = int(sample.get("decode_depth", code_depth + 2))
    vq_indices = sample["vq_indices"].to(device).long()
    octree = reconstruct_octree(sample, args.full_depth, code_depth, device)
    codes = vqvae.quantizer.extract_code(vq_indices)
    surface, surface_sdf = sdf_surface_points(
      vqvae,
      codes,
      code_depth,
      decode_depth,
      octree,
      args.sdf_points,
      args.surface_points,
      args.chunk_size,
      device,
      args.vqvae_update_octree,
    )
    stem = path.name.replace("_sample.pt", "")
    write_ply(output_dir / f"{stem}_surface.ply", surface)
    torch.save({
      **sample,
      "surface_points": surface,
      "surface_sdf": surface_sdf,
    }, output_dir / path.name)
    print(
      f"decoded={path} nodes_depth_stop={int(octree.nnum[code_depth])} "
      f"saved={output_dir / f'{stem}_surface.ply'}")


if __name__ == "__main__":
  main()
