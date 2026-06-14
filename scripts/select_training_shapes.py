from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Select ShapeNet training shapes and export their GT OBJs.")
  parser.add_argument("--data", default="data/02691156")
  parser.add_argument("--output", default="outputs/training_filelist.txt")
  parser.add_argument("--count", type=int, default=4)
  parser.add_argument(
    "--stride",
    type=int,
    default=1,
    help="Select every K-th shape after sorting.")
  parser.add_argument(
    "--offset",
    type=int,
    default=0,
    help="Start index before applying --stride.")
  parser.add_argument(
    "--absolute",
    action="store_true",
    help="Write absolute shape directory paths instead of uid names.")
  parser.add_argument(
    "--obj-name",
    default="models/model_normalized.obj",
    help="Relative GT OBJ path inside each shape directory.")
  parser.add_argument(
    "--gt-output-dir",
    default=None,
    help="Optional directory for copied GT OBJ files.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  root = Path(args.data)
  if args.count < 1:
    raise ValueError("--count must be >= 1")
  if args.stride < 1:
    raise ValueError("--stride must be >= 1")
  if args.offset < 0:
    raise ValueError("--offset must be >= 0")
  shape_dirs = sorted(p for p in root.iterdir() if (p / "pointcloud.npz").exists())
  if not shape_dirs:
    raise RuntimeError(f"No pointcloud.npz files found under {root}")
  selected = shape_dirs[args.offset::args.stride][:args.count]
  if not selected:
    raise RuntimeError("No shapes selected. Check --offset, --stride, and --count.")

  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  lines = [str(p.resolve()) if args.absolute else p.name for p in selected]
  output.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote {len(lines)} entries to {output}")
  for line in lines:
    print(line)

  if args.gt_output_dir is not None:
    gt_output_dir = Path(args.gt_output_dir)
    gt_output_dir.mkdir(parents=True, exist_ok=True)
    for index, shape_dir in enumerate(selected):
      src = shape_dir / args.obj_name
      if not src.exists():
        raise FileNotFoundError(f"Missing GT OBJ for {shape_dir.name}: {src}")
      dst = gt_output_dir / f"{index:04d}_{shape_dir.name}.obj"
      shutil.copy2(src, dst)
    print(f"copied {len(selected)} GT OBJ files to {gt_output_dir}")


if __name__ == "__main__":
  main()
