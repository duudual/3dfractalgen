from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from fractal3d import OctreeConfig, ShapeOctreeDataset, collate_shapes  # noqa: E402
from fractal3d.octgpt_vqvae import load_octgpt_vqvae  # noqa: E402
from ognn.octreed import OctreeD  # noqa: E402
from sample_vae import (  # noqa: E402
  load_vae,
  sample_structure_and_vq,
  write_ply,
)
from train_vae import batch_uid, load_cached_tokens, split_by_depth  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Debug whether bad VAE samples come from decode, VQ bits, or prior z.")
  parser.add_argument("--data", default="data/02691156")
  parser.add_argument("--filelist", default="outputs/debug16_filelist.txt")
  parser.add_argument("--vq-cache-dir", default="outputs/vq_cache_debug16_d6")
  parser.add_argument("--vae-ckpt", required=True)
  parser.add_argument(
    "--vqvae-ckpt", default="ckpt/vqvae_large_im5_uncond_bsq32.pth")
  parser.add_argument("--output-dir", default="outputs/debug_vae_decode")
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--temperature-split", type=float, default=0.7)
  parser.add_argument("--temperature-vq", type=float, default=0.7)
  parser.add_argument(
    "--sample-tokens",
    action=argparse.BooleanOptionalAction,
    default=False)
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def vq_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
  pred = pred.long()
  target = target.long()
  exact = (pred == target).all(dim=1)
  pred_one = pred == 1
  target_one = target == 1
  tp = (pred_one & target_one).sum().item()
  fp = (pred_one & ~target_one).sum().item()
  fn = (~pred_one & target_one).sum().item()
  return {
    "bit_accuracy": float((pred == target).float().mean().item()),
    "node_exact": float(exact.float().mean().item()),
    "pred_one_rate": float(pred_one.float().mean().item()),
    "target_one_rate": float(target_one.float().mean().item()),
    "one_precision": tp / max(tp + fp, 1),
    "one_recall": tp / max(tp + fn, 1),
  }


def node_counts(octree, max_depth: int) -> list[int]:
  return [int(octree.nnum[d]) for d in range(max_depth + 1)]


def predict_vq_on_octree(
  model,
  octree,
  z: torch.Tensor,
  full_depth: int,
  depth_stop: int,
  parallel_child: bool,
) -> torch.Tensor:
  decoder = model.decoder
  hidden = decoder.bootstrap_hidden_from_z(
    octree, z, full_depth - 1, parallel=parallel_child)
  for parent_depth in range(full_depth - 1, depth_stop - 1):
    _, child_hidden, child_indices = decoder.forward_split(
      octree, parent_depth, hidden, parallel=parallel_child, z=z)
    if parent_depth + 1 <= depth_stop - 1:
      hidden = decoder.scatter_child_hidden(
        child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))

  logits, _, child_indices = decoder.forward_vq(
    octree, depth_stop - 1, hidden, parallel=parallel_child, z=z)
  bits = logits.argmax(dim=-1).long()
  indices = torch.zeros(
    int(octree.nnum[depth_stop]), decoder.vq_groups, dtype=torch.long,
    device=octree.device)
  valid = child_indices >= 0
  indices[child_indices.clamp(min=0)[valid]] = bits[valid]
  return indices


@torch.no_grad()
def decode_surface(
  vqvae,
  codes: torch.Tensor,
  code_depth: int,
  decode_depth: int,
  octree,
  args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
  octree_for_decode = copy.deepcopy(octree)
  min_full_depth = code_depth - vqvae.decoder.encoder_stages + 1
  octree_for_decode.full_depth = min(int(octree.full_depth), min_full_depth)

  for depth in range(code_depth, decode_depth):
    split = torch.zeros(
      int(octree_for_decode.nnum[depth]), dtype=torch.int32,
      device=octree_for_decode.device)
    octree_for_decode.octree_split(split, depth)
    octree_for_decode.octree_grow(depth + 1, update_neigh=True)

  doctree = OctreeD(octree_for_decode)
  octree_out = copy.deepcopy(doctree)
  decoded = vqvae.decode_code(
    codes, code_depth, doctree, octree_out, pos=None, update_octree=True)
  neural_mpu = decoded["neural_mpu"]

  keep_points: list[torch.Tensor] = []
  keep_sdf: list[torch.Tensor] = []
  remaining = args.sdf_points
  while remaining > 0:
    count = min(args.chunk_size, remaining)
    xyz = torch.rand(count, 3, device=octree.device) * 2.0 - 1.0
    batch_id = torch.zeros(count, 1, device=octree.device)
    sdf = neural_mpu(torch.cat([xyz, batch_id], dim=1))
    if sdf.dim() > 1:
      sdf = sdf[:, 0]
    take = min(args.surface_points, count)
    _, idx = torch.topk(sdf.abs(), k=take, largest=False)
    keep_points.append(xyz[idx].detach().cpu())
    keep_sdf.append(sdf[idx].detach().cpu())
    remaining -= count

  points = torch.cat(keep_points, dim=0)
  sdf = torch.cat(keep_sdf, dim=0)
  take = min(args.surface_points, points.shape[0])
  _, idx = torch.topk(sdf.abs(), k=take, largest=False)

  split_stats = {}
  for depth, logits in decoded["logits"].items():
    split_stats[int(depth)] = {
      "nodes": int(logits.shape[0]),
      "pred_split_rate": float(logits.argmax(dim=1).float().mean().item()),
    }
  stats = {
    "decode_input_nnum": node_counts(octree_for_decode, decode_depth),
    "decode_output_nnum": node_counts(decoded["octree_out"], decode_depth),
    "vqvae_split_stats": split_stats,
    "sdf_min_abs": float(sdf.abs().min().item()),
    "sdf_median_abs": float(sdf.abs().median().item()),
  }
  return points[idx], sdf[idx], stats


def save_case(
  name: str,
  vqvae,
  indices: torch.Tensor,
  code_depth: int,
  decode_depth: int,
  octree,
  output_dir: Path,
  args: argparse.Namespace,
) -> dict[str, object]:
  codes = vqvae.quantizer.extract_code(indices)
  surface, surface_sdf, stats = decode_surface(
    vqvae, codes, code_depth, decode_depth, octree, args)
  write_ply(output_dir / f"{name}.ply", surface)
  torch.save({
    "indices": indices.detach().cpu(),
    "surface_points": surface,
    "surface_sdf": surface_sdf,
    "stats": stats,
  }, output_dir / f"{name}.pt")
  return stats


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model, checkpoint = load_vae(args.vae_ckpt, device)
  saved = checkpoint.get("args", {})
  full_depth = int(saved.get("full_depth", 3))
  depth_stop = int(saved.get("depth_stop", 6))
  decode_depth = int(saved.get("depth", depth_stop))
  z_dim = int(saved.get("z_dim", 128))
  parallel_child = bool(saved.get("parallel_child_train", True))

  config = OctreeConfig(
    depth=decode_depth,
    full_depth=full_depth,
    points_scale=float(saved.get("points_scale", 1.0)),
    max_points=int(saved.get("max_points", 120000)),
    sample_seed=int(saved.get("sample_seed", 0)),
  )
  dataset = ShapeOctreeDataset(args.data, config=config, filelist=args.filelist)
  batch = collate_shapes([dataset[args.index]])
  uid = batch_uid(batch)
  octree = batch["octree_gt"].to(device)
  cache_args = SimpleNamespace(
    depth=decode_depth, full_depth=full_depth, depth_stop=depth_stop,
    vq_groups=int(saved.get("vq_groups", 32)))
  cached = load_cached_tokens(Path(args.vq_cache_dir), uid, cache_args)
  gt_indices = cached["indices"].to(device).long()

  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)

  with torch.no_grad():
    mu, logvar = model.encode(
      octree, split_by_depth(octree, full_depth, depth_stop), gt_indices, depth_stop)
    recon_z = mu
    recon_indices = predict_vq_on_octree(
      model, octree, recon_z, full_depth, depth_stop, parallel_child)
    prior_z = torch.randn(1, z_dim, device=device)
    prior_octree, prior_indices, _ = sample_structure_and_vq(
      model,
      prior_z,
      full_depth,
      depth_stop,
      args.temperature_split,
      args.temperature_vq,
      args.sample_tokens,
      parallel_child,
      device,
    )

  print(f"uid={uid}")
  print(f"depth_stop={depth_stop} decode_depth={decode_depth}")
  print(f"gt_nnum={node_counts(octree, decode_depth)}")
  print(f"prior_nnum_to_depth_stop={node_counts(prior_octree, depth_stop)}")
  print(
    "posterior "
    f"mu_abs={float(mu.abs().mean().item()):.6f} "
    f"std_mean={float(torch.exp(0.5 * logvar).mean().item()):.6f} "
    f"kl_per_dim={float((-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean().item()):.6f}")
  print(f"recon_vq_metrics={vq_metrics(recon_indices, gt_indices)}")

  for name, indices, case_octree in [
    ("gt_vq", gt_indices, octree),
    ("recon_vq_on_gt_structure", recon_indices, octree),
    ("prior_sample", prior_indices, prior_octree),
  ]:
    stats = save_case(
      name, vqvae, indices, depth_stop, decode_depth, case_octree, output_dir, args)
    print(f"{name}_decode_stats={stats}")


if __name__ == "__main__":
  main()
