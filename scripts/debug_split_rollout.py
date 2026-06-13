from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from fractal3d import OctreeConfig, ShapeOctreeDataset, collate_shapes  # noqa: E402
from sample_vae import (  # noqa: E402
  load_cached_tokens,
  load_vae,
  sample_binary_logits,
  sample_structure_and_vq,
  split_by_depth,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Compare teacher-forced split prediction with free-run rollout.")
  parser.add_argument("--data", default="data/02691156")
  parser.add_argument("--filelist", default="outputs/debug16_filelist.txt")
  parser.add_argument("--vq-cache-dir", default="outputs/vq_cache_debug16_d6")
  parser.add_argument("--vae-ckpt", required=True)
  parser.add_argument("--num-samples", type=int, default=16)
  parser.add_argument("--temperature-split", type=float, default=0.6)
  parser.add_argument("--temperature-vq", type=float, default=0.6)
  parser.add_argument(
    "--sample-tokens",
    action=argparse.BooleanOptionalAction,
    default=False)
  parser.add_argument("--posterior-noise", type=float, default=0.0)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def batch_uid(batch: dict[str, object]) -> str:
  uid = batch["uid"]
  if isinstance(uid, (list, tuple)):
    return str(uid[0])
  return str(uid)


def child_split_targets(model, octree, parent_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
  child_indices = model._child_indices(octree, parent_depth)
  child_depth = parent_depth + 1
  child_split = octree.children[child_depth].ge(0).long()
  valid = child_indices >= 0
  safe = child_indices.clamp(min=0)
  targets = torch.where(valid, child_split[safe], torch.zeros_like(safe))
  return targets, valid


@torch.no_grad()
def teacher_forced_split_stats(model, octree, z: torch.Tensor, full_depth: int, depth_stop: int) -> tuple[dict[int, dict[str, float]], torch.Tensor]:
  decoder = model.decoder
  parent_hidden = decoder.bootstrap_hidden_from_z(
    octree, z, full_depth - 1, parallel=False)
  stats: dict[int, dict[str, float]] = {}
  for parent_depth in range(full_depth - 1, depth_stop - 1):
    target_depth = parent_depth + 1
    targets, valid = child_split_targets(decoder, octree, parent_depth)
    logits, child_hidden, child_indices = decoder.forward_split(
      octree, parent_depth, parent_hidden, parallel=False, z=z)
    pred = sample_binary_logits(
      logits, temperature=1.0, sample_tokens=False).long()
    probs = F.softmax(logits, dim=-1)[..., 1]
    valid_pred = pred[valid]
    valid_targets = targets[valid]
    stats[target_depth] = {
      "gt_nodes": float(int(octree.nnum[target_depth])),
      "gt_split": float(int(valid_targets.sum().item())),
      "gt_rate": float(valid_targets.float().mean().item()),
      "pred_split": float(int(valid_pred.sum().item())),
      "pred_rate": float(valid_pred.float().mean().item()),
      "prob_mean": float(probs[valid].mean().item()),
      "accuracy": float((valid_pred == valid_targets).float().mean().item()),
    }
    if target_depth <= depth_stop - 1:
      parent_hidden = decoder.scatter_child_hidden(
        child_hidden, child_indices, int(octree.nnum[target_depth]))
  return stats, parent_hidden


def format_depth_stats(stats: dict[int, dict[str, float]]) -> str:
  chunks = []
  for depth in sorted(stats):
    item = stats[depth]
    chunks.append(
      f"d{depth}:gt={int(item['gt_split'])}/{int(item['gt_nodes'])}"
      f" pred={int(item['pred_split'])} rate={item['pred_rate']:.3f}"
      f" pmean={item['prob_mean']:.3f} acc={item['accuracy']:.3f}")
  return " | ".join(chunks)


def format_free_stats(split_by_depth: dict[int, torch.Tensor], octree, depth_stop: int) -> str:
  chunks = []
  for depth in sorted(split_by_depth):
    split = split_by_depth[depth]
    chunks.append(
      f"d{depth}:nodes={int(split.numel())} split={int(split.sum())}"
      f" rate={float(split.float().mean().item()):.3f}")
  chunks.append(f"d{depth_stop}:nodes={int(octree.nnum[depth_stop])}")
  return " | ".join(chunks)


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  model, checkpoint = load_vae(args.vae_ckpt, device)
  saved = checkpoint.get("args", {})
  full_depth = int(saved.get("full_depth", 3))
  depth_stop = int(saved.get("depth_stop", 6))
  decode_depth = int(saved.get("depth", depth_stop))

  config = OctreeConfig(
    depth=decode_depth,
    full_depth=full_depth,
    points_scale=float(saved.get("points_scale", 1.0)),
    max_points=int(saved.get("max_points", 120000)),
    sample_seed=int(saved.get("sample_seed", 0)),
  )
  dataset = ShapeOctreeDataset(args.data, config=config, filelist=args.filelist)

  for index in range(min(args.num_samples, len(dataset))):
    batch = collate_shapes([dataset[index]])
    uid = batch_uid(batch)
    octree = batch["octree_gt"].to(device)
    vq_indices = load_cached_tokens(Path(args.vq_cache_dir), uid).to(device)
    with torch.no_grad():
      mu, logvar = model.encode(
        octree, split_by_depth(octree, full_depth, depth_stop), vq_indices, depth_stop)
      std = torch.exp(0.5 * logvar)
      z = mu
      if args.posterior_noise > 0:
        z = z + args.posterior_noise * std * torch.randn_like(std)
      teacher_stats, _ = teacher_forced_split_stats(
        model, octree, z, full_depth, depth_stop)
      free_octree, _, free_split = sample_structure_and_vq(
        model,
        z,
        full_depth,
        depth_stop,
        args.temperature_split,
        args.temperature_vq,
        args.sample_tokens,
        device,
      )
    print(
      f"sample={index} uid={uid} "
      f"mu_abs={float(mu.abs().mean().item()):.4f} "
      f"std={float(std.mean().item()):.4f}")
    print(f"  teacher: {format_depth_stats(teacher_stats)}")
    print(f"  free:    {format_free_stats(free_split, free_octree, depth_stop)}")


if __name__ == "__main__":
  main()
