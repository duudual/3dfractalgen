from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"
DEFAULT_VQVAE_CKPT = PROJECT_ROOT / "ckpt" / "vqvae_large_im5_uncond_bsq32.pth"

from fractal3d import (  # noqa: E402
  Fractal3DGenerator,
  OctreeConfig,
  ShapeOctreeDataset,
  collate_shapes,
)
from fractal3d.config import parse_args_with_config  # noqa: E402
from fractal3d.octgpt_vqvae import encode_bsq_tokens, load_octgpt_vqvae  # noqa: E402


def shape_dirs(root: Path, filelist: str | Path | None) -> list[Path]:
  if filelist is None:
    return sorted(p for p in root.iterdir() if (p / "pointcloud.npz").exists())

  filelist_path = Path(filelist)
  dirs: list[Path] = []
  for line in filelist_path.read_text(encoding="utf-8").splitlines():
    item = line.strip()
    if not item:
      continue
    path = Path(item)
    if not path.is_absolute():
      path = root / item
    dirs.append(path)
  return dirs


class ShapeUidDataset(Dataset):
  def __init__(self, root: str | Path, filelist: str | Path | None = None) -> None:
    self.root = Path(root)
    self.shape_dirs = shape_dirs(self.root, filelist)
    if len(self.shape_dirs) == 0:
      raise RuntimeError(f"No pointcloud.npz files found under {self.root}")

  def __len__(self) -> int:
    return len(self.shape_dirs)

  def __getitem__(self, index: int) -> dict[str, str]:
    return {"uid": self.shape_dirs[index].name}


def collate_uids(batch: list[dict[str, str]]) -> dict[str, list[str]]:
  return {"uid": [item["uid"] for item in batch]}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train 3DFractalGen VQ token prediction with frozen OctGPT VQVAE.")
  parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
  parser.add_argument("--filelist", default=None)
  parser.add_argument("--val-filelist", default=None)
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--output-dir", default="outputs/vq_train")
  parser.add_argument("--resume", default=None)
  parser.add_argument(
      "--init-split-ckpt",
      default=None,
      help="Optional split checkpoint to initialize the shared model.")
  parser.add_argument(
      "--vq-cache-dir",
      default=None,
      help="Optional directory of cached VQVAE tokens from cache_vq_tokens.py.")

  parser.add_argument("--vqvae-ckpt", default=str(DEFAULT_VQVAE_CKPT))

  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--depth-stop", type=int, default=6)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)
  parser.add_argument("--sample-seed", type=int, default=0)

  parser.add_argument("--dim", type=int, default=128)
  parser.add_argument("--layers", type=int, default=4)
  parser.add_argument("--heads", type=int, default=4)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument(
      "--parallel-child-train",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Predict all 8 child hidden states in one causal transformer pass.")
  parser.add_argument(
      "--amp",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Use CUDA automatic mixed precision during training/evaluation.")
  parser.add_argument("--num-vq-embed", type=int, default=32)
  parser.add_argument("--vq-groups", type=int, default=32)
  parser.add_argument(
      "--freeze-split-head",
      action="store_true",
      default=True,
      help="Freeze split_head during VQ-only training.")

  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=1)
  parser.add_argument("--num-workers", type=int, default=0)
  parser.add_argument("--lr", type=float, default=1e-4)
  parser.add_argument("--weight-decay", type=float, default=0.05)
  parser.add_argument("--grad-clip", type=float, default=1.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--log-every", type=int, default=10)
  parser.add_argument("--val-every", type=int, default=1)
  return parse_args_with_config(parser)


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
  if args.vq_cache_dir is not None:
    train_set = ShapeUidDataset(args.data, filelist=args.filelist)
    if args.val_filelist:
      val_set = ShapeUidDataset(args.data, filelist=args.val_filelist)
    elif args.val_fraction > 0 and len(train_set) > 1:
      val_len = max(1, int(round(len(train_set) * args.val_fraction)))
      train_len = len(train_set) - val_len
      generator = torch.Generator().manual_seed(args.seed)
      train_set, val_set = random_split(train_set, [train_len, val_len], generator)
    else:
      val_set = None

    train_loader = DataLoader(
      train_set,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      collate_fn=collate_uids,
    )
    val_loader = None
    if val_set is not None:
      val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_uids,
      )
    return train_loader, val_loader

  config = OctreeConfig(
    depth=args.depth,
    full_depth=args.full_depth,
    points_scale=args.points_scale,
    max_points=args.max_points,
    sample_seed=args.sample_seed,
  )
  root = Path(args.data)
  train_set = ShapeOctreeDataset(root, config=config, filelist=args.filelist)

  if args.val_filelist:
    val_set = ShapeOctreeDataset(root, config=config, filelist=args.val_filelist)
  elif args.val_fraction > 0 and len(train_set) > 1:
    val_len = max(1, int(round(len(train_set) * args.val_fraction)))
    train_len = len(train_set) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(train_set, [train_len, val_len], generator)
  else:
    val_set = None

  train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    collate_fn=collate_shapes,
  )
  val_loader = None
  if val_set is not None:
    val_loader = DataLoader(
      val_set,
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      collate_fn=collate_shapes,
    )
  return train_loader, val_loader


def vq_targets_for_depth(
  model: Fractal3DGenerator,
  octree,
  indices: torch.Tensor,
  codes: torch.Tensor,
  parent_depth: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  child_indices = model._child_indices(octree, parent_depth)
  valid = child_indices >= 0
  safe_idx = child_indices.clamp(min=0)
  target_indices = indices[safe_idx]
  target_codes = codes[safe_idx]
  target_indices = torch.where(
    valid.unsqueeze(-1), target_indices, torch.zeros_like(target_indices))
  target_codes = torch.where(
    valid.unsqueeze(-1), target_codes, torch.zeros_like(target_codes))
  return target_indices.long(), target_codes, valid


def hidden_at_depth(
  model: Fractal3DGenerator,
  octree,
  start_depth: int,
  target_depth: int,
  parallel_child_train: bool,
) -> torch.Tensor:
  hidden = model.initial_hidden(octree, start_depth)
  for parent_depth in range(start_depth, target_depth):
    _, child_hidden, child_indices = model.forward_split(
      octree, parent_depth, hidden, parallel=parallel_child_train)
    hidden = model.scatter_child_hidden(
      child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))
  return hidden


def vq_loss_for_batch(
  model: Fractal3DGenerator,
  vqvae: torch.nn.Module | None,
  octree,
  expected_code_depth: int,
  cached_tokens: dict[str, torch.Tensor | int] | None = None,
  parallel_child_train: bool = True,
) -> tuple[torch.Tensor | None, dict[str, float]]:
  if cached_tokens is None:
    if vqvae is None:
      raise ValueError("vqvae is required when cached_tokens is not provided")
    with torch.no_grad():
      indices, codes, code_depth = encode_bsq_tokens(vqvae, octree)
  else:
    indices = cached_tokens["indices"].to(octree.device)
    codes = cached_tokens["codes"].to(octree.device)
    code_depth = int(cached_tokens["code_depth"])

  if code_depth != expected_code_depth:
    raise ValueError(
      f"VQVAE code depth is {code_depth}, expected {expected_code_depth}.")

  parent_depth = code_depth - 1
  target_indices, _, valid = vq_targets_for_depth(
    model, octree, indices, codes, parent_depth)
  valid_bits = valid.unsqueeze(-1).expand_as(target_indices)
  total_bits = int(valid_bits.sum().item())
  if total_bits == 0:
    return None, {"bits": 0.0}

  parent_hidden = hidden_at_depth(
    model, octree, model.full_depth - 1, parent_depth, parallel_child_train)
  logits, _, _ = model.forward_vq(
    octree=octree,
    parent_depth=parent_depth,
    parent_hidden=parent_hidden,
    parallel=parallel_child_train,
  )
  loss = F.cross_entropy(
    logits[valid_bits].reshape(-1, 2),
    target_indices[valid_bits].reshape(-1),
  )

  pred = logits.argmax(dim=-1)
  correct_bits = int(((pred == target_indices) & valid_bits).sum().item())
  exact_nodes = ((pred == target_indices) | ~valid_bits).all(dim=-1)
  exact_nodes = exact_nodes & valid
  node_total = int(valid.sum().item())
  one_bits = target_indices[valid_bits] == 1
  pred_one = pred[valid_bits] == 1
  tp = int((pred_one & one_bits).sum().item())
  fp = int((pred_one & ~one_bits).sum().item())
  fn = int((~pred_one & one_bits).sum().item())
  stats = {
    "loss_sum": float(loss.detach().item()) * total_bits,
    "bits": float(total_bits),
    "bit_correct": float(correct_bits),
    "nodes": float(node_total),
    "node_exact": float(exact_nodes.sum().item()),
    "tp": float(tp),
    "fp": float(fp),
    "fn": float(fn),
  }
  return loss, stats


def empty_stats() -> dict[str, float]:
  return {
    "loss_sum": 0.0,
    "bits": 0.0,
    "bit_correct": 0.0,
    "nodes": 0.0,
    "node_exact": 0.0,
    "tp": 0.0,
    "fp": 0.0,
    "fn": 0.0,
  }


def add_stats(total: dict[str, float], stats: dict[str, float]) -> None:
  for key, value in stats.items():
    total[key] = total.get(key, 0.0) + float(value)


def finalize_stats(total: dict[str, float]) -> dict[str, float]:
  bits = max(total["bits"], 1.0)
  nodes = max(total["nodes"], 1.0)
  precision_den = max(total["tp"] + total["fp"], 1.0)
  recall_den = max(total["tp"] + total["fn"], 1.0)
  precision = total["tp"] / precision_den
  recall = total["tp"] / recall_den
  f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
  return {
    "loss": total["loss_sum"] / bits,
    "bit_accuracy": total["bit_correct"] / bits,
    "node_exact_accuracy": total["node_exact"] / nodes,
    "one_precision": precision,
    "one_recall": recall,
    "one_f1": f1,
    "bits": total["bits"],
    "nodes": total["nodes"],
  }


def batch_uid(batch: dict[str, object]) -> str:
  uid = batch["uid"]
  if isinstance(uid, (list, tuple)):
    return str(uid[0])
  return str(uid)


def load_cached_tokens(
  cache_dir: Path,
  uid: str,
  expected_meta: dict[str, object],
) -> dict[str, object]:
  cache_path = cache_dir / f"{uid}.pt"
  if not cache_path.exists():
    raise FileNotFoundError(f"Missing VQ cache for {uid}: {cache_path}")
  cached = torch.load(cache_path, map_location="cpu")
  for key, expected in expected_meta.items():
    actual = cached.get(key)
    if actual != expected:
      raise ValueError(
        f"{cache_path} was built with {key}={actual}, expected {expected}.")
  return cached


def run_epoch(
  model: Fractal3DGenerator,
  vqvae: torch.nn.Module | None,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  code_depth: int,
  grad_clip: float,
  vq_cache_dir: Path | None = None,
  cache_meta: dict[str, object] | None = None,
  writer: SummaryWriter | None = None,
  global_step: int = 0,
  log_every: int = 10,
  epoch: int = 0,
  phase: str = "train",
  parallel_child_train: bool = True,
  use_amp: bool = False,
  scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, float], int]:
  training = optimizer is not None
  model.train(training)
  if vqvae is not None:
    vqvae.eval()
  total_stats = empty_stats()

  pbar = tqdm(loader, desc=f"{phase} epoch {epoch:04d}", dynamic_ncols=True, leave=False)
  for batch in pbar:
    octree = batch["octree_gt"].to(device)
    cached_tokens = None
    if vq_cache_dir is not None:
      cached_tokens = load_cached_tokens(
        vq_cache_dir, batch_uid(batch), cache_meta or {})
    with torch.cuda.amp.autocast(enabled=use_amp):
      loss, stats = vq_loss_for_batch(
        model, vqvae, octree, code_depth, cached_tokens, parallel_child_train)
    if loss is None:
      continue

    if training:
      assert scaler is not None
      optimizer.zero_grad(set_to_none=True)
      scaler.scale(loss).backward()
      if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
          [p for p in model.parameters() if p.requires_grad], grad_clip)
      scaler.step(optimizer)
      scaler.update()
      global_step += 1

      if writer is not None and global_step % log_every == 0:
        step_metrics = finalize_stats(stats)
        for key, value in step_metrics.items():
          writer.add_scalar(f"train/{key}_step", value, global_step)

    add_stats(total_stats, stats)
    metrics_so_far = finalize_stats(total_stats)
    pbar.set_postfix(
      loss=f"{metrics_so_far['loss']:.4f}",
      bit_acc=f"{metrics_so_far['bit_accuracy']:.4f}",
      exact=f"{metrics_so_far['node_exact_accuracy']:.4f}",
    )

  return finalize_stats(total_stats), global_step


def save_checkpoint(
  path: Path,
  model: Fractal3DGenerator,
  optimizer: torch.optim.Optimizer,
  epoch: int,
  best_val_loss: float,
  args: argparse.Namespace,
) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best_val_loss,
    "args": vars(args),
  }, path)


def load_train_checkpoint(
  path: Path,
  model: Fractal3DGenerator,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[int, float]:
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", math.inf))


def load_model_checkpoint(path: Path, model: Fractal3DGenerator, device: torch.device) -> None:
  checkpoint = torch.load(path, map_location=device)
  state = checkpoint["model"] if "model" in checkpoint else checkpoint
  model_state = model.state_dict()
  compatible = {
    key: value for key, value in state.items()
    if key in model_state and model_state[key].shape == value.shape
  }
  skipped = sorted(set(state.keys()) - set(compatible.keys()))
  missing, unexpected = model.load_state_dict(compatible, strict=False)
  if skipped:
    print(f"Skipped shape-incompatible init keys: {skipped}")
  if missing:
    print(f"Missing keys after init load: {missing}")
  if unexpected:
    print(f"Unexpected keys after init load: {unexpected}")


def write_epoch_metrics(
  writer: SummaryWriter,
  phase: str,
  metrics: dict[str, float],
  epoch: int,
) -> None:
  for key, value in metrics.items():
    writer.add_scalar(f"{phase}/{key}_epoch", value, epoch)


def format_metrics(prefix: str, metrics: dict[str, float]) -> str:
  return (
    f"{prefix}_loss={metrics['loss']:.6f} "
    f"{prefix}_bit_acc={metrics['bit_accuracy']:.4f} "
    f"{prefix}_exact={metrics['node_exact_accuracy']:.4f} "
    f"{prefix}_one_p={metrics['one_precision']:.4f} "
    f"{prefix}_one_r={metrics['one_recall']:.4f} "
    f"{prefix}_one_f1={metrics['one_f1']:.4f}"
  )


def main() -> None:
  args = parse_args()
  set_seed(args.seed)
  device = torch.device(args.device)

  if args.depth_stop > args.depth:
    raise ValueError("--depth-stop must be <= --depth")
  if args.vq_cache_dir is not None and args.batch_size != 1:
    raise ValueError("--vq-cache-dir currently requires --batch-size 1")
  code_depth = args.depth_stop
  parent_depth = code_depth - 1
  print(
    f"Training VQ tokens at depth {code_depth} "
    f"from parent depth {parent_depth}.")

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(output_dir / "tensorboard")

  train_loader, val_loader = build_loaders(args)
  model = Fractal3DGenerator(
    dim=args.dim,
    num_layers=args.layers,
    num_heads=args.heads,
    num_vq_embed=args.num_vq_embed,
    vq_groups=args.vq_groups,
    full_depth=args.full_depth,
    max_depth=args.depth,
    dropout=args.dropout,
  ).to(device)

  if args.init_split_ckpt:
    load_model_checkpoint(Path(args.init_split_ckpt), model, device)

  if args.freeze_split_head:
    for param in model.split_head.parameters():
      param.requires_grad_(False)

  trainable = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(
    trainable, lr=args.lr, weight_decay=args.weight_decay)
  use_amp = bool(args.amp and device.type == "cuda")
  scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

  vq_cache_dir = Path(args.vq_cache_dir) if args.vq_cache_dir is not None else None
  vqvae = None if vq_cache_dir is not None else load_octgpt_vqvae(args.vqvae_ckpt, device)
  cache_meta = {
    "depth": args.depth,
    "full_depth": args.full_depth,
    "depth_stop": args.depth_stop,
    "points_scale": args.points_scale,
    "max_points": args.max_points,
    "sample_seed": args.sample_seed,
    "dim": args.dim,
    "num_vq_embed": args.num_vq_embed,
    "vq_groups": args.vq_groups,
  }

  start_epoch = 0
  best_val_loss = math.inf
  if args.resume:
    start_epoch, best_val_loss = load_train_checkpoint(
      Path(args.resume), model, optimizer, device)

  for epoch in range(start_epoch, args.epochs):
    train_metrics, global_step = run_epoch(
      model=model,
      vqvae=vqvae,
      loader=train_loader,
      optimizer=optimizer,
      device=device,
      code_depth=code_depth,
      grad_clip=args.grad_clip,
      vq_cache_dir=vq_cache_dir,
      cache_meta=cache_meta,
      writer=writer,
      global_step=epoch * max(len(train_loader), 1),
      log_every=args.log_every,
      epoch=epoch,
      phase="train",
      parallel_child_train=args.parallel_child_train,
      use_amp=use_amp,
      scaler=scaler,
    )
    write_epoch_metrics(writer, "train", train_metrics, epoch)

    monitor_loss = train_metrics["loss"]
    if val_loader is not None and (epoch + 1) % args.val_every == 0:
      with torch.no_grad():
        val_metrics, _ = run_epoch(
          model=model,
          vqvae=vqvae,
          loader=val_loader,
          optimizer=None,
          device=device,
          code_depth=code_depth,
          grad_clip=args.grad_clip,
          vq_cache_dir=vq_cache_dir,
          cache_meta=cache_meta,
          epoch=epoch,
          phase="val",
          parallel_child_train=args.parallel_child_train,
          use_amp=use_amp,
        )
      write_epoch_metrics(writer, "val", val_metrics, epoch)
      monitor_loss = val_metrics["loss"]
      print(f"epoch {epoch:04d} {format_metrics('train', train_metrics)} "
            f"{format_metrics('val', val_metrics)}")
    else:
      print(f"epoch {epoch:04d} {format_metrics('train', train_metrics)}")

    save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val_loss, args)
    if monitor_loss < best_val_loss:
      best_val_loss = monitor_loss
      save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val_loss, args)
      writer.add_scalar("checkpoint/best_val_loss", best_val_loss, epoch)

  writer.close()


if __name__ == "__main__":
  main()
