from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import ocnn
import torch
from ocnn.octree import Octree, Points
from torch.utils.data import Dataset


@dataclass(frozen=True)
class OctreeConfig:
  depth: int = 8
  full_depth: int = 3
  points_scale: float = 0.5
  max_points: int | None = 120000
  distort: bool = False
  noise_std: float = 0.005
  seq_start_depth: int | None = None
  seq_stop_depth: int | None = None


def _shape_dirs(root: Path, filelist: Path | None) -> list[Path]:
  if filelist is None:
    return sorted(p for p in root.iterdir() if (p / "pointcloud.npz").exists())

  dirs: list[Path] = []
  for line in filelist.read_text(encoding="utf-8").splitlines():
    item = line.strip()
    if not item:
      continue
    path = Path(item)
    if not path.is_absolute():
      path = root / item
    dirs.append(path)
  return dirs


def points_to_octree(
  points: torch.Tensor,
  normals: torch.Tensor,
  depth: int,
  full_depth: int,
) -> tuple[Octree, Points]:
  point_cloud = Points(points=points, normals=normals)
  point_cloud.clip(min=-1.0, max=1.0)

  octree = Octree(depth=depth, full_depth=full_depth)
  octree.build_octree(point_cloud)
  return octree, point_cloud


def octree_to_split_sequence(
  octree: Octree,
  start_depth: int,
  stop_depth: int,
) -> torch.Tensor:
  """Return 0/1 split labels from start_depth to stop_depth - 1."""
  children = octree.children[start_depth:stop_depth]
  if len(children) == 0:
    return torch.empty(0, dtype=torch.long)
  return torch.cat(children).ge(0).long()


class ShapeOctreeDataset(Dataset):
  """ShapeNet-style pointcloud.npz dataset converted to OCNN octrees.

  This mirrors the data part of octgpt/datasets/shapenet.py without the
  thsolver dependency or image/text/SDF branches.
  """

  def __init__(
    self,
    root: str | Path,
    config: OctreeConfig | None = None,
    filelist: str | Path | None = None,
  ) -> None:
    self.root = Path(root)
    self.config = config or OctreeConfig()
    self.filelist = Path(filelist) if filelist is not None else None
    self.shape_dirs = _shape_dirs(self.root, self.filelist)

    if len(self.shape_dirs) == 0:
      raise RuntimeError(f"No pointcloud.npz files found under {self.root}")

  def __len__(self) -> int:
    return len(self.shape_dirs)

  def __getitem__(self, index: int) -> dict[str, object]:
    shape_dir = self.shape_dirs[index]
    raw = np.load(shape_dir / "pointcloud.npz")

    points = torch.from_numpy(raw["points"].astype(np.float32))
    normals = torch.from_numpy(raw["normals"].astype(np.float32))
    points = points / self.config.points_scale

    max_points = self.config.max_points
    if max_points is not None and points.shape[0] > max_points:
      choice = np.random.choice(points.shape[0], size=max_points, replace=False)
      points = points[choice]
      normals = normals[choice]

    octree_gt, points_gt = points_to_octree(
      points=points,
      normals=normals,
      depth=self.config.depth,
      full_depth=self.config.full_depth,
    )

    if self.config.distort:
      noise_std = torch.rand(1).item() * self.config.noise_std
      noisy_points = points + noise_std * torch.randn_like(points)
      noisy_normals = normals + noise_std * torch.randn_like(normals)
      octree_in, points_in = points_to_octree(
        points=noisy_points,
        normals=noisy_normals,
        depth=self.config.depth,
        full_depth=self.config.full_depth,
      )
    else:
      octree_in, points_in = octree_gt, points_gt

    seq_start = self.config.seq_start_depth or self.config.full_depth
    seq_stop = self.config.seq_stop_depth or self.config.depth
    split_seq = octree_to_split_sequence(octree_gt, seq_start, seq_stop)

    return {
      "uid": shape_dir.name,
      "octree_in": octree_in,
      "octree_gt": octree_gt,
      "points_in": points_in,
      "points_gt": points_gt,
      "split_seq": split_seq,
    }


def collate_shapes(batch: Iterable[dict[str, object]]) -> dict[str, object]:
  output = ocnn.dataset.CollateBatch(merge_points=False)(list(batch))
  if "split_seq" in output:
    output["split_seq"] = [seq.long() for seq in output["split_seq"]]
  return output
