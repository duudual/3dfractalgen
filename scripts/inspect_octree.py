from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"

from fractal3d import OctreeConfig, ShapeOctreeDataset


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--data",
      default=str(DEFAULT_DATA_DIR),
      help="Dataset root directory.")
  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)
  parser.add_argument("--index", type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  config = OctreeConfig(
    depth=args.depth,
    full_depth=args.full_depth,
    points_scale=args.points_scale,
    max_points=args.max_points,
  )
  dataset = ShapeOctreeDataset(Path(args.data), config=config)
  sample = dataset[args.index]
  octree = sample["octree_gt"]

  print(f"uid: {sample['uid']}")
  print(f"depth: {octree.depth}, full_depth: {octree.full_depth}")
  for depth in range(octree.full_depth, octree.depth + 1):
    print(f"depth {depth}: nodes={int(octree.nnum[depth])}")
  split_seq = sample["split_seq"]
  print(f"split_seq length: {split_seq.numel()}")
  print(f"split labels: split={int(split_seq.sum())}, leaf={int(split_seq.numel() - split_seq.sum())}")


if __name__ == "__main__":
  main()
