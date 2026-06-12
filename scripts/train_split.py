from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"

from fractal3d import (  # noqa: E402
  Fractal3DGenerator,
  OctreeConfig,
  ShapeOctreeDataset,
  collate_shapes,
)
from fractal3d.config import parse_args_with_config  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Train 3DFractalGen split prediction with TensorBoard logging.")
  parser.add_argument(
      "--data",
      default=str(DEFAULT_DATA_DIR),
      help="Dataset root directory.")
  parser.add_argument("--filelist", default=None, help="Optional train file list.")
  parser.add_argument("--val-filelist", default=None, help="Optional val file list.")
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--output-dir", default="outputs/split_train")
  parser.add_argument("--resume", default=None)

  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument(
      "--depth-stop",
      type=int,
      default=6,
      help="OctGPT-style latent stop depth. Split labels are trained for "
           "depths [full_depth, depth_stop).")
  parser.add_argument("--train-depth-low", type=int, default=None)
  parser.add_argument("--train-depth-high", type=int, default=None)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)

  parser.add_argument("--dim", type=int, default=128)
  parser.add_argument("--layers", type=int, default=4)
  parser.add_argument("--heads", type=int, default=4)
  parser.add_argument("--dropout", type=float, default=0.1)

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
  config = OctreeConfig(
    depth=args.depth,
    full_depth=args.full_depth,
    points_scale=args.points_scale,
    max_points=args.max_points,
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


def child_split_targets(
  model: Fractal3DGenerator,
  octree,
  parent_depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  child_indices = model._child_indices(octree, parent_depth)
  child_depth = parent_depth + 1
  child_split = octree.children[child_depth].ge(0).long()
  valid = child_indices >= 0
  safe_indices = child_indices.clamp(min=0)
  targets = child_split[safe_indices]
  targets = torch.where(valid, targets, torch.zeros_like(targets))
  return targets, valid


def split_loss_for_depth(
  model: Fractal3DGenerator,
  octree,
  parent_depth: int,
  parent_hidden: torch.Tensor,
) -> tuple[torch.Tensor | None, dict[str, int], torch.Tensor | None]:
  targets, valid = child_split_targets(model, octree, parent_depth)
  total = int(valid.sum().item())
  if total == 0:
    return None, {"tokens": 0}, None

  logits, child_hidden, child_indices = model.forward_split(
    octree=octree,
    parent_depth=parent_depth,
    parent_hidden=parent_hidden,
  )
  loss_per_token = F.cross_entropy(
    logits.reshape(-1, 2),
    targets.reshape(-1),
    reduction="none",
  ).view_as(targets)
  loss = (loss_per_token * valid.float()).sum() / valid.sum().clamp_min(1)

  pred = logits.argmax(dim=-1)
  valid_targets = targets[valid]
  valid_pred = pred[valid]
  split_mask = valid_targets == 1
  leaf_mask = valid_targets == 0
  pred_split = valid_pred == 1

  stats = {
    "target_depth": parent_depth + 1,
    "tokens": total,
    "correct": int((valid_pred == valid_targets).sum().item()),
    "split_total": int(split_mask.sum().item()),
    "split_correct": int(((valid_pred == valid_targets) & split_mask).sum().item()),
    "leaf_total": int(leaf_mask.sum().item()),
    "leaf_correct": int(((valid_pred == valid_targets) & leaf_mask).sum().item()),
    "tp": int((pred_split & split_mask).sum().item()),
    "fp": int((pred_split & leaf_mask).sum().item()),
    "fn": int(((~pred_split) & split_mask).sum().item()),
  }
  next_hidden = model.scatter_child_hidden(
    child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))
  return loss, stats, next_hidden


def empty_stats() -> dict[str, float]:
  return {
    "loss_sum": 0.0,
    "tokens": 0.0,
    "correct": 0.0,
    "split_total": 0.0,
    "split_correct": 0.0,
    "leaf_total": 0.0,
    "leaf_correct": 0.0,
    "tp": 0.0,
    "fp": 0.0,
    "fn": 0.0,
  }


def add_stats(
  total: dict[str, float],
  loss_value: float,
  stats: dict[str, int],
) -> None:
  tokens = float(stats.get("tokens", 0))
  total["loss_sum"] += loss_value * tokens
  for key in [
    "tokens", "correct", "split_total", "split_correct", "leaf_total",
    "leaf_correct", "tp", "fp", "fn",
  ]:
    total[key] += float(stats.get(key, 0))


def finalize_stats(total: dict[str, float]) -> dict[str, float]:
  tokens = max(total["tokens"], 1.0)
  split_total = max(total["split_total"], 1.0)
  leaf_total = max(total["leaf_total"], 1.0)
  precision_den = max(total["tp"] + total["fp"], 1.0)
  recall_den = max(total["tp"] + total["fn"], 1.0)
  precision = total["tp"] / precision_den
  recall = total["tp"] / recall_den
  f1_den = max(precision + recall, 1e-12)
  return {
    "loss": total["loss_sum"] / tokens,
    "accuracy": total["correct"] / tokens,
    "split_accuracy": total["split_correct"] / split_total,
    "leaf_accuracy": total["leaf_correct"] / leaf_total,
    "split_precision": precision,
    "split_recall": recall,
    "split_f1": 2.0 * precision * recall / f1_den,
    "tokens": total["tokens"],
    "split_tokens": total["split_total"],
    "leaf_tokens": total["leaf_total"],
  }


def run_epoch(
  model: Fractal3DGenerator,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  depth_low: int,
  depth_high: int,
  grad_clip: float,
  writer: SummaryWriter | None = None,
  global_step: int = 0,
  log_every: int = 10,
  epoch: int = 0,
  phase: str = "train",
) -> tuple[dict[str, float], int]:
  training = optimizer is not None
  model.train(training)

  total_stats = empty_stats()
  depth_stats: dict[int, dict[str, float]] = {}

  pbar = tqdm(
    loader,
    desc=f"{phase} epoch {epoch:04d}",
    dynamic_ncols=True,
    leave=False,
  )
  for batch_idx, batch in enumerate(pbar):
    octree = batch["octree_gt"].to(device)
    losses: list[torch.Tensor] = []
    batch_stats = empty_stats()
    hidden_by_depth: dict[int, torch.Tensor] = {
      depth_low: model.initial_hidden(octree, depth_low)
    }

    for parent_depth in range(depth_low, depth_high + 1):
      if parent_depth + 1 > octree.depth:
        continue
      if parent_depth not in hidden_by_depth:
        continue
      loss, stats, next_hidden = split_loss_for_depth(
        model, octree, parent_depth, hidden_by_depth[parent_depth])
      if loss is None:
        continue
      if next_hidden is not None:
        hidden_by_depth[parent_depth + 1] = next_hidden
      loss_value = float(loss.detach().item())
      losses.append(loss * stats["tokens"])
      add_stats(batch_stats, loss_value, stats)

      target_depth = stats["target_depth"]
      if target_depth not in depth_stats:
        depth_stats[target_depth] = empty_stats()
      add_stats(depth_stats[target_depth], loss_value, stats)

    if batch_stats["tokens"] == 0:
      continue

    loss = torch.stack(losses).sum() / batch_stats["tokens"]
    if training:
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()
      global_step += 1

      if writer is not None and global_step % log_every == 0:
        step_metrics = finalize_stats(batch_stats)
        writer.add_scalar("train/loss_step", float(loss.item()), global_step)
        writer.add_scalar("train/accuracy_step", step_metrics["accuracy"], global_step)
        writer.add_scalar(
          "train/split_accuracy_step", step_metrics["split_accuracy"], global_step)
        writer.add_scalar(
          "train/leaf_accuracy_step", step_metrics["leaf_accuracy"], global_step)
        writer.add_scalar(
          "train/split_recall_step", step_metrics["split_recall"], global_step)

    add_stats(total_stats, float(loss.item()), {k: int(v) for k, v in batch_stats.items()})
    metrics_so_far = finalize_stats(total_stats)
    pbar.set_postfix(
      loss=f"{metrics_so_far['loss']:.4f}",
      acc=f"{metrics_so_far['accuracy']:.4f}",
      split_rec=f"{metrics_so_far['split_recall']:.4f}",
      tokens=int(metrics_so_far["tokens"]),
    )

  metrics = finalize_stats(total_stats)
  metrics["per_depth"] = {
    depth: finalize_stats(stats) for depth, stats in sorted(depth_stats.items())
  }
  return metrics, global_step


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


def load_checkpoint(
  path: Path,
  model: Fractal3DGenerator,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[int, float]:
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", math.inf))


def write_epoch_metrics(
  writer: SummaryWriter,
  phase: str,
  metrics: dict,
  epoch: int,
) -> None:
  for key, value in metrics.items():
    if key == "per_depth":
      continue
    writer.add_scalar(f"{phase}/{key}_epoch", value, epoch)

  for depth, depth_metrics in metrics.get("per_depth", {}).items():
    for key, value in depth_metrics.items():
      writer.add_scalar(f"{phase}_by_depth/depth{depth}/{key}", value, epoch)


def format_metrics(prefix: str, metrics: dict) -> str:
  return (
    f"{prefix}_loss={metrics['loss']:.6f} "
    f"{prefix}_acc={metrics['accuracy']:.4f} "
    f"{prefix}_split_acc={metrics['split_accuracy']:.4f} "
    f"{prefix}_leaf_acc={metrics['leaf_accuracy']:.4f} "
    f"{prefix}_split_p={metrics['split_precision']:.4f} "
    f"{prefix}_split_r={metrics['split_recall']:.4f} "
    f"{prefix}_split_f1={metrics['split_f1']:.4f}"
  )


def main() -> None:
  args = parse_args()
  set_seed(args.seed)

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(output_dir / "tensorboard")

  device = torch.device(args.device)
  train_loader, val_loader = build_loaders(args)
  depth_low = args.train_depth_low
  if depth_low is None:
    depth_low = args.full_depth - 1
  depth_high = args.train_depth_high
  if depth_high is None:
    depth_high = args.depth_stop - 2

  if args.depth_stop > args.depth:
    raise ValueError("--depth-stop must be <= --depth")
  if depth_low < 0 or depth_high < depth_low:
    raise ValueError("Invalid train depth range")
  print(
    f"Training split labels for target depths "
    f"{depth_low + 1}..{depth_high + 1} "
    f"(parent depths {depth_low}..{depth_high}).")

  model = Fractal3DGenerator(
    dim=args.dim,
    num_layers=args.layers,
    num_heads=args.heads,
    full_depth=args.full_depth,
    max_depth=args.depth,
    dropout=args.dropout,
  ).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

  start_epoch = 0
  best_val_loss = math.inf
  if args.resume:
    start_epoch, best_val_loss = load_checkpoint(
      Path(args.resume), model, optimizer, device)

  for epoch in range(start_epoch, args.epochs):
    train_metrics, global_step = run_epoch(
      model=model,
      loader=train_loader,
      optimizer=optimizer,
      device=device,
      depth_low=depth_low,
      depth_high=depth_high,
      grad_clip=args.grad_clip,
      writer=writer,
      global_step=epoch * max(len(train_loader), 1),
      log_every=args.log_every,
      epoch=epoch,
      phase="train",
    )
    write_epoch_metrics(writer, "train", train_metrics, epoch)

    monitor_loss = train_metrics["loss"]
    if val_loader is not None and (epoch + 1) % args.val_every == 0:
      with torch.no_grad():
        val_metrics, _ = run_epoch(
          model=model,
          loader=val_loader,
          optimizer=None,
          device=device,
          depth_low=depth_low,
          depth_high=depth_high,
          grad_clip=args.grad_clip,
          epoch=epoch,
          phase="val",
        )
      write_epoch_metrics(writer, "val", val_metrics, epoch)
      monitor_loss = val_metrics["loss"]
      print(f"epoch {epoch:04d} {format_metrics('train', train_metrics)} "
            f"{format_metrics('val', val_metrics)}")
    else:
      print(f"epoch {epoch:04d} {format_metrics('train', train_metrics)}")

    save_checkpoint(
      output_dir / "last.pt", model, optimizer, epoch, best_val_loss, args)
    if monitor_loss < best_val_loss:
      best_val_loss = monitor_loss
      save_checkpoint(
        output_dir / "best.pt", model, optimizer, epoch, best_val_loss, args)
      writer.add_scalar("checkpoint/best_val_loss", best_val_loss, epoch)

  writer.close()


if __name__ == "__main__":
  main()
