from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"
DEFAULT_VQ_CKPT = PROJECT_ROOT / "outputs" / "vq_train_cached" / "best.pt"
DEFAULT_SPLIT_CKPT = PROJECT_ROOT / "outputs" / "split_train" / "best.pt"
DEFAULT_VQVAE_CKPT = PROJECT_ROOT / "ckpt" / "vqvae_large_im5_uncond_bsq32.pth"

from fractal3d import Fractal3DGenerator, OctreeConfig, ShapeOctreeDataset  # noqa: E402
from fractal3d.octgpt_vqvae import encode_bsq_tokens, load_octgpt_vqvae  # noqa: E402
from ocnn.octree import Octree  # noqa: E402
from ognn.octreed import OctreeD  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Sample VQ-head predictions on a ground-truth octree structure.")
  parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
  parser.add_argument("--index", type=int, default=0)
  parser.add_argument("--output-dir", default="outputs/vq_samples")
  parser.add_argument("--vq-ckpt", default=str(DEFAULT_VQ_CKPT))
  parser.add_argument("--split-ckpt", default=str(DEFAULT_SPLIT_CKPT))
  parser.add_argument("--vqvae-ckpt", default=str(DEFAULT_VQVAE_CKPT))
  parser.add_argument(
    "--mode",
    choices=["gt_octree", "pred_split", "both"],
    default="both",
    help="Which structure to use before VQ prediction.")

  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--depth-stop", type=int, default=6)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)
  parser.add_argument("--sample-seed", type=int, default=0)

  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--sample-tokens", action="store_true")
  parser.add_argument(
    "--teacher-forced-vq",
    action="store_true",
    help="Also save deterministic VQ prediction on the ground-truth octree structure.")
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def model_args_from_checkpoint(checkpoint: dict, args: argparse.Namespace) -> dict:
  saved = checkpoint.get("args", {})
  return {
    "dim": int(saved.get("dim", 128)),
    "num_layers": int(saved.get("layers", 4)),
    "num_heads": int(saved.get("heads", 4)),
    "num_vq_embed": int(saved.get("num_vq_embed", 32)),
    "vq_groups": int(saved.get("vq_groups", 32)),
    "full_depth": int(saved.get("full_depth", args.full_depth)),
    "max_depth": int(saved.get("depth", args.depth)),
    "dropout": float(saved.get("dropout", 0.1)),
  }


def load_generator(
  path: str | Path,
  device: torch.device,
  args: argparse.Namespace,
  *,
  strict: bool = True,
):
  checkpoint = torch.load(path, map_location=device)
  model = Fractal3DGenerator(**model_args_from_checkpoint(checkpoint, args)).to(device)
  state = checkpoint["model"] if "model" in checkpoint else checkpoint
  if strict:
    model.load_state_dict(state)
  else:
    model_state = model.state_dict()
    compatible = {
      key: value for key, value in state.items()
      if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
  model.eval()
  return model, checkpoint


def predict_structure(
  model: Fractal3DGenerator,
  full_depth: int,
  code_depth: int,
  device: torch.device,
  sample_tokens: bool,
  temperature: float,
) -> tuple[Octree, dict[int, torch.Tensor]]:
  octree = Octree(depth=code_depth, full_depth=full_depth, device=device)
  for depth in range(full_depth + 1):
    octree.octree_grow_full(depth, update_neigh=True)

  split_by_depth: dict[int, torch.Tensor] = {}
  parent_depth = full_depth - 1
  parent_hidden = model.initial_hidden(octree, parent_depth)
  for split_depth in range(full_depth, code_depth):
    logits, child_hidden, child_indices = model.forward_split(
      octree, parent_depth, parent_hidden)
    if sample_tokens:
      probs = F.softmax(logits / temperature, dim=-1)
      sampled = torch.multinomial(probs.reshape(-1, 2), num_samples=1)
      sampled = sampled.view(logits.shape[0], 8)
    else:
      sampled = logits.argmax(dim=-1)
    valid = child_indices >= 0
    safe_idx = child_indices.clamp(min=0)
    split = torch.zeros(
      int(octree.nnum[split_depth]), dtype=torch.int32, device=device)
    split[safe_idx[valid]] = sampled[valid].int()
    split_by_depth[split_depth] = split.detach().cpu()
    octree.octree_split(split, split_depth)
    octree.octree_grow(split_depth + 1, update_neigh=True)
    parent_hidden = model.scatter_child_hidden(
      child_hidden, child_indices, int(octree.nnum[split_depth]))
    parent_depth = split_depth
  return octree, split_by_depth


def predict_depth_vq(
  model: Fractal3DGenerator,
  vqvae,
  octree,
  code_depth: int,
  temperature: float,
  sample_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  parent_depth = code_depth - 1
  n_code = int(octree.nnum[code_depth])
  pred_indices = torch.zeros(
    n_code, model.vq_groups, dtype=torch.long, device=octree.device)

  parent_hidden = model.initial_hidden(octree, model.full_depth - 1)
  for depth in range(model.full_depth - 1, parent_depth):
    _, child_hidden, child_idx = model.forward_split(octree, depth, parent_hidden)
    parent_hidden = model.scatter_child_hidden(
      child_hidden, child_idx, int(octree.nnum[depth + 1]))

  logits, _, child_idx = model.forward_vq(octree, parent_depth, parent_hidden)
  if sample_tokens:
    probs = F.softmax(logits / temperature, dim=-1)
    token = torch.multinomial(probs.reshape(-1, 2), num_samples=1)
    child_indices = token.view(logits.shape[0], 8, model.vq_groups)
  else:
    child_indices = logits.argmax(dim=-1)
  valid = child_idx >= 0
  safe_idx = child_idx.clamp(min=0)
  pred_indices[safe_idx[valid]] = child_indices[valid]
  pred_codes = vqvae.quantizer.extract_code(pred_indices)
  return pred_indices, pred_codes, valid


def predict_depth_vq_teacher_forced(
  model: Fractal3DGenerator,
  vqvae,
  octree,
  code_depth: int,
  gt_indices: torch.Tensor,
  gt_codes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  parent_depth = code_depth - 1
  parent_hidden = model.initial_hidden(octree, model.full_depth - 1)
  for depth in range(model.full_depth - 1, parent_depth):
    _, child_hidden, child_idx = model.forward_split(octree, depth, parent_hidden)
    parent_hidden = model.scatter_child_hidden(
      child_hidden, child_idx, int(octree.nnum[depth + 1]))
  logits, _, child_idx = model.forward_vq(octree, parent_depth, parent_hidden)
  pred_child = logits.argmax(dim=-1)
  pred_indices = torch.zeros_like(gt_indices)
  valid = child_idx >= 0
  safe_idx = child_idx.clamp(min=0)
  pred_indices[safe_idx[valid]] = pred_child[valid]
  pred_codes = vqvae.quantizer.extract_code(pred_indices)
  return pred_indices, pred_codes, valid


def split_structure_metrics(
  gt_octree,
  pred_octree,
  split_by_depth: dict[int, torch.Tensor],
  full_depth: int,
  code_depth: int,
) -> dict[int, dict[str, float]]:
  metrics: dict[int, dict[str, float]] = {}
  for depth in range(full_depth, code_depth):
    pred_split = split_by_depth[depth].to(pred_octree.device).long()
    gt_split = gt_octree.children[depth].ge(0).long()
    pred_keys = pred_octree.key(depth)
    idx_in_gt = gt_octree.search_key(pred_keys, depth)
    common = idx_in_gt >= 0
    if common.any():
      gt_on_pred = gt_split[idx_in_gt[common].long()]
      pred_on_common = pred_split[common]
      common_acc = (pred_on_common == gt_on_pred).float().mean().item()
      common_precision = (
        ((pred_on_common == 1) & (gt_on_pred == 1)).float().sum()
        / max((pred_on_common == 1).float().sum().item(), 1.0)
      ).item()
      common_recall = (
        ((pred_on_common == 1) & (gt_on_pred == 1)).float().sum()
        / max((gt_on_pred == 1).float().sum().item(), 1.0)
      ).item()
    else:
      common_acc = 0.0
      common_precision = 0.0
      common_recall = 0.0
    metrics[depth] = {
      "gt_nodes": float(int(gt_octree.nnum[depth])),
      "pred_nodes": float(int(pred_octree.nnum[depth])),
      "common_nodes": float(int(common.sum().item())),
      "gt_split": float(int(gt_split.sum().item())),
      "pred_split": float(int(pred_split.sum().item())),
      "common_acc": common_acc,
      "common_split_precision": common_precision,
      "common_split_recall": common_recall,
    }
  return metrics


def write_ply(path: Path, points: torch.Tensor) -> None:
  points = points.detach().cpu().float()
  with path.open("w", encoding="utf-8") as f:
    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write(f"element vertex {points.shape[0]}\n")
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("end_header\n")
    for x, y, z in points.tolist():
      f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def sdf_surface_points(
  vqvae,
  codes: torch.Tensor,
  code_depth: int,
  octree,
  num_query: int,
  num_surface: int,
  chunk_size: int,
  device: torch.device,
  update_octree: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  doctree = OctreeD(octree)
  octree_out = OctreeD(octree)
  decoded = vqvae.decode_code(
    codes, code_depth, doctree, octree_out, pos=None, update_octree=update_octree)
  neural_mpu = decoded["neural_mpu"]

  keep_points: list[torch.Tensor] = []
  keep_sdf: list[torch.Tensor] = []
  remaining = num_query
  while remaining > 0:
    count = min(chunk_size, remaining)
    xyz = torch.rand(count, 3, device=device) * 2.0 - 1.0
    batch_id = torch.zeros(count, 1, device=device)
    pos = torch.cat([xyz, batch_id], dim=1)
    sdf = neural_mpu(pos)
    if sdf.dim() > 1:
      sdf = sdf[:, 0]
    take = min(num_surface, count)
    _, idx = torch.topk(sdf.abs(), k=take, largest=False)
    keep_points.append(xyz[idx].detach().cpu())
    keep_sdf.append(sdf[idx].detach().cpu())
    remaining -= count

  points = torch.cat(keep_points, dim=0)
  sdf = torch.cat(keep_sdf, dim=0)
  take = min(num_surface, points.shape[0])
  _, idx = torch.topk(sdf.abs(), k=take, largest=False)
  return points[idx], sdf[idx]


def save_vq_sample(
  *,
  label: str,
  uid: str,
  args: argparse.Namespace,
  output_dir: Path,
  index: int,
  checkpoint: dict,
  vqvae,
  vq_model: Fractal3DGenerator,
  octree,
  gt_indices: torch.Tensor | None,
  gt_codes: torch.Tensor | None,
  split_by_depth: dict[int, torch.Tensor] | None,
  split_metrics: dict[int, dict[str, float]] | None,
  device: torch.device,
  gt_nnum_depth: int | None = None,
  update_octree: bool = False,
  teacher_forced: bool = False,
) -> None:
  with torch.no_grad():
    if teacher_forced:
      if gt_indices is None or gt_codes is None:
        raise ValueError("teacher_forced=True requires gt_indices and gt_codes")
      pred_indices, pred_codes, valid = predict_depth_vq_teacher_forced(
        vq_model, vqvae, octree, args.depth_stop, gt_indices, gt_codes)
    else:
      pred_indices, pred_codes, valid = predict_depth_vq(
        vq_model, vqvae, octree, args.depth_stop, args.temperature, args.sample_tokens)
    metrics: dict[str, float] = {}
    if gt_indices is not None and gt_indices.shape == pred_indices.shape:
      metrics["bit_accuracy"] = (pred_indices == gt_indices).float().mean().item()
      metrics["node_exact"] = (pred_indices == gt_indices).all(dim=1).float().mean().item()

    surface, surface_sdf = sdf_surface_points(
      vqvae=vqvae,
      codes=pred_codes,
      code_depth=args.depth_stop,
      octree=octree,
      num_query=args.sdf_points,
      num_surface=args.surface_points,
      chunk_size=args.chunk_size,
      device=device,
      update_octree=update_octree,
    )

  stem = f"{index:04d}_{uid}_{label}"
  token_path = output_dir / f"{stem}_tokens.pt"
  ply_path = output_dir / f"{stem}_surface.ply"
  torch.save({
    "uid": uid,
    "label": label,
    "teacher_forced": teacher_forced,
    "index": index,
    "checkpoint_epoch": checkpoint.get("epoch"),
    "checkpoint_best_val_loss": checkpoint.get("best_val_loss", math.nan),
    "code_depth": args.depth_stop,
    "pred_indices": pred_indices.cpu(),
    "gt_indices": None if gt_indices is None else gt_indices.cpu(),
    "valid": valid.cpu(),
    "split_by_depth": split_by_depth,
    "split_metrics": split_metrics,
    "nnum_depth": int(octree.nnum[args.depth_stop]),
    "gt_nnum_depth": gt_nnum_depth,
    "surface_points": surface,
    "surface_sdf": surface_sdf,
    **metrics,
  }, token_path)
  write_ply(ply_path, surface)

  metric_text = " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
  node_text = f"nodes={int(octree.nnum[args.depth_stop])}"
  if gt_nnum_depth is not None:
    node_text += f" gt_nodes={gt_nnum_depth}"
  print(f"[{label}] {node_text} {metric_text}")
  if split_metrics is not None:
    for depth, metric in split_metrics.items():
      print(
        f"[{label}] split_depth={depth} "
        f"gt_nodes={int(metric['gt_nodes'])} "
        f"pred_nodes={int(metric['pred_nodes'])} "
        f"common={int(metric['common_nodes'])} "
        f"gt_split={int(metric['gt_split'])} "
        f"pred_split={int(metric['pred_split'])} "
        f"common_acc={metric['common_acc']:.4f} "
        f"split_p={metric['common_split_precision']:.4f} "
        f"split_r={metric['common_split_recall']:.4f}")
  print(f"[{label}] saved_tokens={token_path}")
  print(f"[{label}] saved_surface_ply={ply_path}")


def main() -> None:
  args = parse_args()
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
  dataset = ShapeOctreeDataset(args.data, config=config)
  sample = dataset[args.index]
  uid = str(sample["uid"])
  octree = sample["octree_gt"].to(device)
  octree.construct_all_neigh()

  vq_model, checkpoint = load_generator(args.vq_ckpt, device, args)
  split_model = None
  if args.mode in {"pred_split", "both"}:
    split_model, _ = load_generator(args.split_ckpt, device, args, strict=False)
  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)
  with torch.no_grad():
    gt_indices, gt_codes, code_depth = encode_bsq_tokens(vqvae, octree)
    if code_depth != args.depth_stop:
      raise ValueError(
        f"VQVAE code depth is {code_depth}, expected {args.depth_stop}.")

  print(f"uid={uid}")
  print(f"vq_checkpoint={args.vq_ckpt}")
  if args.mode in {"gt_octree", "both"}:
    save_vq_sample(
      label="gt_octree",
      uid=uid,
      args=args,
      output_dir=output_dir,
      index=args.index,
      checkpoint=checkpoint,
      vqvae=vqvae,
      vq_model=vq_model,
      octree=octree,
      gt_indices=gt_indices,
      gt_codes=gt_codes,
      split_by_depth=None,
      split_metrics=None,
      device=device,
      gt_nnum_depth=int(octree.nnum[args.depth_stop]),
      update_octree=False,
    )
    if args.teacher_forced_vq:
      save_vq_sample(
        label="gt_octree_teacher_forced",
        uid=uid,
        args=args,
        output_dir=output_dir,
        index=args.index,
        checkpoint=checkpoint,
        vqvae=vqvae,
        vq_model=vq_model,
        octree=octree,
        gt_indices=gt_indices,
        gt_codes=gt_codes,
        split_by_depth=None,
        split_metrics=None,
        device=device,
        gt_nnum_depth=int(octree.nnum[args.depth_stop]),
        update_octree=False,
        teacher_forced=True,
      )

  if args.mode in {"pred_split", "both"}:
    assert split_model is not None
    pred_octree, split_by_depth = predict_structure(
      split_model,
      full_depth=args.full_depth,
      code_depth=args.depth_stop,
      device=device,
      sample_tokens=args.sample_tokens,
      temperature=args.temperature,
    )
    metrics = split_structure_metrics(
      octree, pred_octree, split_by_depth, args.full_depth, args.depth_stop)
    save_vq_sample(
      label="pred_split",
      uid=uid,
      args=args,
      output_dir=output_dir,
      index=args.index,
      checkpoint=checkpoint,
      vqvae=vqvae,
      vq_model=vq_model,
      octree=pred_octree,
      gt_indices=None,
      gt_codes=None,
      split_by_depth=split_by_depth,
      split_metrics=metrics,
      device=device,
      gt_nnum_depth=int(octree.nnum[args.depth_stop]),
      update_octree=True,
    )


if __name__ == "__main__":
  main()
