from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"

from fractal3d import Fractal3DVAE, OctreeConfig, ShapeOctreeDataset, collate_shapes  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Inspect split predictions from a trained Fractal3D VAE.")
  parser.add_argument("--checkpoint", required=True)
  parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
  parser.add_argument("--filelist", default=None)
  parser.add_argument("--vq-cache-dir", required=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--max-samples", type=int, default=4)
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument(
    "--serial-child",
    action="store_true",
    help="Use serial child generation instead of the checkpoint/default parallel path.")
  return parser.parse_args()


def cfg_get(cfg: dict[str, object], key: str, default):
  return cfg[key] if key in cfg and cfg[key] is not None else default


def batch_uid(batch: dict[str, object]) -> str:
  uid = batch["uid"]
  if isinstance(uid, (list, tuple)):
    return str(uid[0])
  return str(uid)


def split_by_depth(octree, full_depth: int, depth_stop: int) -> dict[int, torch.Tensor]:
  return {
    depth: octree.children[depth].ge(0).long()
    for depth in range(full_depth, depth_stop)
  }


def child_split_targets(model, octree, parent_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
  child_indices = model._child_indices(octree, parent_depth)
  child_depth = parent_depth + 1
  child_split = octree.children[child_depth].ge(0).long()
  valid = child_indices >= 0
  safe = child_indices.clamp(min=0)
  targets = torch.where(valid, child_split[safe], torch.zeros_like(safe))
  return targets, valid


def load_cached_tokens(cache_dir: Path, uid: str, device: torch.device) -> torch.Tensor:
  path = cache_dir / f"{uid}.pt"
  if not path.exists():
    raise FileNotFoundError(f"Missing VQ cache for {uid}: {path}")
  return torch.load(path, map_location=device)["indices"].long()


def build_model(saved_args: dict[str, object], device: torch.device) -> Fractal3DVAE:
  model = Fractal3DVAE(
    dim=int(cfg_get(saved_args, "dim", 192)),
    z_dim=int(cfg_get(saved_args, "z_dim", 128)),
    encoder_layers=int(cfg_get(saved_args, "encoder_layers", 6)),
    decoder_layers=int(cfg_get(saved_args, "decoder_layers", 6)),
    encoder_patch_size=int(cfg_get(saved_args, "encoder_patch_size", 1024)),
    encoder_dilation=int(cfg_get(saved_args, "encoder_dilation", 8)),
    num_heads=int(cfg_get(saved_args, "heads", 6)),
    num_vq_embed=int(cfg_get(saved_args, "num_vq_embed", 32)),
    vq_groups=int(cfg_get(saved_args, "vq_groups", 32)),
    full_depth=int(cfg_get(saved_args, "full_depth", 3)),
    max_depth=int(cfg_get(saved_args, "depth", 8)),
    dropout=0.0,
  ).to(device)
  return model


def print_child_stats(
  targets: torch.Tensor,
  pred: torch.Tensor,
  prob: torch.Tensor,
  valid: torch.Tensor,
) -> None:
  for child_id in range(8):
    mask = valid[:, child_id]
    count = int(mask.sum().item())
    if count == 0:
      print(f"    child {child_id}: count=0")
      continue
    t = targets[:, child_id][mask].float()
    p = pred[:, child_id][mask].float()
    q = prob[:, child_id][mask]
    print(
      f"    child {child_id}: count={count:4d} "
      f"target_rate={float(t.mean().item()):.4f} "
      f"pred_rate={float(p.mean().item()):.4f} "
      f"prob_mean={float(q.mean().item()):.4f} "
      f"prob_min={float(q.min().item()):.4f} "
      f"prob_max={float(q.max().item()):.4f}")


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  checkpoint = torch.load(args.checkpoint, map_location=device)
  saved_args = checkpoint.get("args", {})
  if not isinstance(saved_args, dict):
    saved_args = vars(saved_args)

  depth = int(cfg_get(saved_args, "depth", 8))
  full_depth = int(cfg_get(saved_args, "full_depth", 3))
  depth_stop = int(cfg_get(saved_args, "depth_stop", 6))
  parallel_child_train = bool(cfg_get(saved_args, "parallel_child_train", True))
  if args.serial_child:
    parallel_child_train = False

  model = build_model(saved_args, device)
  missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
  if missing:
    print(f"missing checkpoint keys: {missing}")
  if unexpected:
    print(f"unexpected checkpoint keys: {unexpected}")
  model.eval()

  config = OctreeConfig(
    depth=depth,
    full_depth=full_depth,
    points_scale=float(cfg_get(saved_args, "points_scale", 1.0)),
    max_points=cfg_get(saved_args, "max_points", 120000),
    sample_seed=int(cfg_get(saved_args, "sample_seed", 0)),
  )
  dataset = ShapeOctreeDataset(args.data, config=config, filelist=args.filelist)
  loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_shapes)

  cache_dir = Path(args.vq_cache_dir)
  with torch.no_grad():
    for sample_idx, batch in enumerate(loader):
      if sample_idx >= args.max_samples:
        break
      uid = batch_uid(batch)
      octree = batch["octree_gt"].to(device)
      vq_indices = load_cached_tokens(cache_dir, uid, device)
      mu, logvar = model.encode(
        octree, split_by_depth(octree, full_depth, depth_stop), vq_indices, depth_stop)
      z = mu
      parent_hidden = model.decoder.bootstrap_hidden_from_z(
        octree, z, full_depth - 1, parallel=parallel_child_train)

      print(f"sample {sample_idx}: uid={uid}")
      print(
        f"  depth={depth} full_depth={full_depth} depth_stop={depth_stop} "
        f"parallel_child={parallel_child_train}")
      for parent_depth in range(full_depth - 1, depth_stop - 1):
        targets, valid = child_split_targets(model.decoder, octree, parent_depth)
        logits, child_hidden, child_indices = model.decoder.forward_split(
          octree, parent_depth, parent_hidden, parallel=parallel_child_train, z=z)
        prob = F.softmax(logits, dim=-1)[..., 1]
        pred = (prob >= args.threshold).long()

        valid_targets = targets[valid]
        valid_pred = pred[valid]
        total = int(valid.sum().item())
        target_pos = int((valid_targets == 1).sum().item())
        pred_pos = int((valid_pred == 1).sum().item())
        tp = int(((valid_pred == 1) & (valid_targets == 1)).sum().item())
        fp = int(((valid_pred == 1) & (valid_targets == 0)).sum().item())
        fn = int(((valid_pred == 0) & (valid_targets == 1)).sum().item())
        acc = float((valid_pred == valid_targets).float().mean().item()) if total else 0.0
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(
          f"  parent_depth={parent_depth} child_depth={parent_depth + 1} "
          f"parents={int(octree.nnum[parent_depth])} valid={total} "
          f"target_pos={target_pos} pred_pos={pred_pos} "
          f"acc={acc:.4f} precision={precision:.4f} recall={recall:.4f}")
        print_child_stats(targets, pred, prob, valid)

        if parent_depth + 1 <= depth_stop - 1:
          parent_hidden = model.decoder.scatter_child_hidden(
            child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))


if __name__ == "__main__":
  main()
