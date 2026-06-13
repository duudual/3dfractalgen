from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Create a small ShapeNet uid filelist for overfit/debug runs.")
  parser.add_argument("--data", default="data/02691156")
  parser.add_argument("--output", default="outputs/debug_filelist.txt")
  parser.add_argument("--count", type=int, default=4)
  parser.add_argument(
    "--absolute",
    action="store_true",
    help="Write absolute shape directory paths instead of uid names.")
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  root = Path(args.data)
  shape_dirs = sorted(p for p in root.iterdir() if (p / "pointcloud.npz").exists())
  if not shape_dirs:
    raise RuntimeError(f"No pointcloud.npz files found under {root}")
  selected = shape_dirs[:args.count]
  output = Path(args.output)
  output.parent.mkdir(parents=True, exist_ok=True)
  lines = [str(p.resolve()) if args.absolute else p.name for p in selected]
  output.write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"wrote {len(lines)} entries to {output}")
  for line in lines:
    print(line)


if __name__ == "__main__":
  main()
