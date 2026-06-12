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
  parser.add_argument("--train-depth-low", type=int, default=None)
  parser.add_argument("--train-depth-high", type=int, default=None)
  parser.add_argument("--points-scale", type=float, default=0.5)
  parser.add_argument("--max-points", type=int, default=120000)

  parser.add_argument("--dim", type=int, default=256)
  parser.add_argument("--layers", type=int, default=6)
  parser.add_argument("--heads", type=int, default=8)
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
  return parser.parse_args()


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
) -> tuple[torch.Tensor | None, int, int]:
  targets, valid = child_split_targets(model, octree, parent_depth)
  total = int(valid.sum().item())
  if total == 0:
    return None, 0, 0

  logits, _ = model.forward_split(
    octree=octree,
    parent_depth=parent_depth,
    split_targets=targets,
  )
  loss_per_token = F.cross_entropy(
    logits.reshape(-1, 2),
    targets.reshape(-1),
    reduction="none",
  ).view_as(targets)
  loss = (loss_per_token * valid.float()).sum() / valid.sum().clamp_min(1)

  pred = logits.argmax(dim=-1)
  correct = int(((pred == targets) & valid).sum().item())
  return loss, correct, total


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

  total_loss_sum = 0.0
  total_tokens = 0
  total_correct = 0

  pbar = tqdm(
    loader,
    desc=f"{phase} epoch {epoch:04d}",
    dynamic_ncols=True,
    leave=False,
  )
  for batch_idx, batch in enumerate(pbar):
    octree = batch["octree_gt"].to(device)
    losses: list[torch.Tensor] = []
    batch_tokens = 0
    batch_correct = 0

    for parent_depth in range(depth_low, depth_high + 1):
      if parent_depth + 1 > octree.depth:
        continue
      loss, correct, tokens = split_loss_for_depth(model, octree, parent_depth)
      if loss is None:
        continue
      losses.append(loss * tokens)
      batch_tokens += tokens
      batch_correct += correct

    if batch_tokens == 0:
      continue

    loss = torch.stack(losses).sum() / batch_tokens
    if training:
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()
      global_step += 1

      if writer is not None and global_step % log_every == 0:
        writer.add_scalar("train/loss_step", float(loss.item()), global_step)
        writer.add_scalar(
          "train/accuracy_step", batch_correct / max(batch_tokens, 1), global_step)

    total_loss_sum += float(loss.item()) * batch_tokens
    total_tokens += batch_tokens
    total_correct += batch_correct
    pbar.set_postfix(
      loss=f"{total_loss_sum / max(total_tokens, 1):.4f}",
      acc=f"{total_correct / max(total_tokens, 1):.4f}",
      tokens=total_tokens,
    )

  metrics = {
    "loss": total_loss_sum / max(total_tokens, 1),
    "accuracy": total_correct / max(total_tokens, 1),
    "tokens": float(total_tokens),
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
    depth_low = args.full_depth
  depth_high = args.train_depth_high
  if depth_high is None:
    depth_high = args.depth - 1

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
    writer.add_scalar("train/loss_epoch", train_metrics["loss"], epoch)
    writer.add_scalar("train/accuracy_epoch", train_metrics["accuracy"], epoch)
    writer.add_scalar("train/tokens_epoch", train_metrics["tokens"], epoch)

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
      writer.add_scalar("val/loss_epoch", val_metrics["loss"], epoch)
      writer.add_scalar("val/accuracy_epoch", val_metrics["accuracy"], epoch)
      writer.add_scalar("val/tokens_epoch", val_metrics["tokens"], epoch)
      monitor_loss = val_metrics["loss"]
      print(
        f"epoch {epoch:04d} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"train_acc={train_metrics['accuracy']:.4f} "
        f"val_loss={val_metrics['loss']:.6f} "
        f"val_acc={val_metrics['accuracy']:.4f}")
    else:
      print(
        f"epoch {epoch:04d} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"train_acc={train_metrics['accuracy']:.4f}")

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
