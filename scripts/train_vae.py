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
DEFAULT_VQVAE_CKPT = PROJECT_ROOT / "ckpt" / "vqvae_large_im5_uncond_bsq32.pth"

from fractal3d import Fractal3DVAE, OctreeConfig, ShapeOctreeDataset, collate_shapes  # noqa: E402
from fractal3d.config import parse_args_with_config  # noqa: E402
from fractal3d.octgpt_vqvae import encode_bsq_tokens, load_octgpt_vqvae  # noqa: E402


def set_seed(seed: int) -> None:
  random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def batch_uid(batch: dict[str, object]) -> str:
  uid = batch["uid"]
  if isinstance(uid, (list, tuple)):
    if len(uid) != 1:
      raise ValueError("Cached VQ VAE training currently expects batch_size=1.")
    return str(uid[0])
  return str(uid)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Train OctFormer-VAE 3DFractalGen.")
  parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
  parser.add_argument("--filelist", default=None)
  parser.add_argument("--val-filelist", default=None)
  parser.add_argument("--val-fraction", type=float, default=0.2)
  parser.add_argument("--output-dir", default="outputs/vae_train")
  parser.add_argument("--resume", default=None)
  parser.add_argument("--vq-cache-dir", default=None)
  parser.add_argument("--vqvae-ckpt", default=str(DEFAULT_VQVAE_CKPT))

  parser.add_argument("--depth", type=int, default=8)
  parser.add_argument("--full-depth", type=int, default=3)
  parser.add_argument("--depth-stop", type=int, default=6)
  parser.add_argument("--points-scale", type=float, default=1.0)
  parser.add_argument("--max-points", type=int, default=120000)
  parser.add_argument("--sample-seed", type=int, default=0)

  parser.add_argument("--dim", type=int, default=192)
  parser.add_argument("--z-dim", type=int, default=128)
  parser.add_argument("--encoder-layers", type=int, default=6)
  parser.add_argument("--decoder-layers", type=int, default=6)
  parser.add_argument("--encoder-patch-size", type=int, default=1024)
  parser.add_argument("--encoder-dilation", type=int, default=8)
  parser.add_argument("--heads", type=int, default=6)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--num-vq-embed", type=int, default=32)
  parser.add_argument("--vq-groups", type=int, default=32)
  parser.add_argument(
    "--parallel-child-train",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

  parser.add_argument("--lambda-vq", type=float, default=1.0)
  parser.add_argument("--beta-max", type=float, default=1e-4)
  parser.add_argument("--beta-warmup-epochs", type=int, default=10)

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


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
  config = OctreeConfig(
    depth=args.depth,
    full_depth=args.full_depth,
    points_scale=args.points_scale,
    max_points=args.max_points,
    sample_seed=args.sample_seed,
  )
  train_set = ShapeOctreeDataset(args.data, config=config, filelist=args.filelist)

  if args.val_filelist:
    val_set = ShapeOctreeDataset(args.data, config=config, filelist=args.val_filelist)
  elif args.val_fraction > 0 and len(train_set) > 1:
    val_len = max(1, int(round(len(train_set) * args.val_fraction)))
    train_len = len(train_set) - val_len
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(train_set, [train_len, val_len], generator)
  else:
    val_set = None

  loader_kwargs = {
    "num_workers": args.num_workers,
    "collate_fn": collate_shapes,
    "pin_memory": str(args.device).startswith("cuda"),
  }
  if args.num_workers > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2
  train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
  val_loader = None
  if val_set is not None:
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
  return train_loader, val_loader


def load_cached_tokens(cache_dir: Path, uid: str, args: argparse.Namespace) -> dict[str, object]:
  path = cache_dir / f"{uid}.pt"
  if not path.exists():
    raise FileNotFoundError(f"Missing VQ cache for {uid}: {path}")
  cached = torch.load(path, map_location="cpu")
  for key in ["depth", "full_depth", "depth_stop", "vq_groups"]:
    expected = getattr(args, key)
    if cached.get(key) != expected:
      raise ValueError(f"{path} has {key}={cached.get(key)}, expected {expected}")
  return cached


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


def split_losses(model, octree, z: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
  parent_hidden = model.bootstrap_hidden_from_z(
    octree, z, args.full_depth - 1, parallel=args.parallel_child_train)
  losses: list[torch.Tensor] = []
  total_tokens = 0
  total_correct = 0
  split_tokens = 0
  pred_split_tokens = 0
  tp = 0
  fp = 0
  fn = 0
  for parent_depth in range(args.full_depth - 1, args.depth_stop - 1):
    targets, valid = child_split_targets(model, octree, parent_depth)
    if int(valid.sum().item()) == 0:
      continue
    logits, child_hidden, child_indices = model.forward_split(
      octree, parent_depth, parent_hidden, parallel=args.parallel_child_train)
    loss_per = F.cross_entropy(
      logits.reshape(-1, 2), targets.reshape(-1), reduction="none").view_as(targets)
    tokens = int(valid.sum().item())
    losses.append((loss_per * valid.float()).sum())
    total_tokens += tokens
    pred = logits.argmax(dim=-1)
    total_correct += int(((pred == targets) & valid).sum().item())
    valid_targets = targets[valid]
    valid_pred = pred[valid]
    target_split = valid_targets == 1
    pred_split = valid_pred == 1
    split_tokens += int(target_split.sum().item())
    pred_split_tokens += int(pred_split.sum().item())
    tp += int((pred_split & target_split).sum().item())
    fp += int((pred_split & ~target_split).sum().item())
    fn += int((~pred_split & target_split).sum().item())
    if parent_depth + 1 <= args.depth_stop - 1:
      parent_hidden = model.scatter_child_hidden(
        child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))

  if total_tokens == 0:
    zero = z.sum() * 0.0
    return zero, {
      "split_tokens": 0.0,
      "split_accuracy": 0.0,
      "split_positive": 0.0,
      "split_pred_positive": 0.0,
      "split_target_rate": 0.0,
      "split_pred_rate": 0.0,
      "split_precision": 0.0,
      "split_recall": 0.0,
    }, parent_hidden
  loss = torch.stack(losses).sum() / total_tokens
  precision = tp / max(tp + fp, 1)
  recall = tp / max(tp + fn, 1)
  return loss, {
    "split_tokens": float(total_tokens),
    "split_accuracy": total_correct / max(total_tokens, 1),
    "split_positive": float(split_tokens),
    "split_pred_positive": float(pred_split_tokens),
    "split_target_rate": split_tokens / max(total_tokens, 1),
    "split_pred_rate": pred_split_tokens / max(total_tokens, 1),
    "split_precision": precision,
    "split_recall": recall,
  }, parent_hidden


def vq_loss(model, octree, parent_hidden: torch.Tensor, vq_indices: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
  parent_depth = args.depth_stop - 1
  child_indices = model._child_indices(octree, parent_depth)
  valid = child_indices >= 0
  safe = child_indices.clamp(min=0)
  targets = torch.where(
    valid.unsqueeze(-1), vq_indices[safe], torch.zeros_like(vq_indices[safe]))
  logits, _, _ = model.forward_vq(
    octree, parent_depth, parent_hidden, parallel=args.parallel_child_train)
  valid_bits = valid.unsqueeze(-1).expand_as(targets)
  if int(valid_bits.sum().item()) == 0:
    zero = parent_hidden.sum() * 0.0
    return zero, {
      "vq_bits": 0.0,
      "vq_bit_accuracy": 0.0,
      "vq_node_exact": 0.0,
      "vq_target_one_rate": 0.0,
      "vq_pred_one_rate": 0.0,
      "vq_one_precision": 0.0,
      "vq_one_recall": 0.0,
    }
  loss = F.cross_entropy(
    logits[valid_bits].reshape(-1, 2),
    targets[valid_bits].reshape(-1).long())
  pred = logits.argmax(dim=-1)
  correct = int(((pred == targets) & valid_bits).sum().item())
  exact = ((pred == targets) | ~valid_bits).all(dim=-1) & valid
  target_ones = targets[valid_bits] == 1
  pred_ones = pred[valid_bits] == 1
  one_tp = int((pred_ones & target_ones).sum().item())
  one_fp = int((pred_ones & ~target_ones).sum().item())
  one_fn = int((~pred_ones & target_ones).sum().item())
  total_bits = int(valid_bits.sum().item())
  return loss, {
    "vq_bits": float(total_bits),
    "vq_bit_accuracy": correct / max(total_bits, 1),
    "vq_node_exact": float(exact.sum().item()) / max(int(valid.sum().item()), 1),
    "vq_target_one_rate": float(target_ones.float().mean().item()),
    "vq_pred_one_rate": float(pred_ones.float().mean().item()),
    "vq_one_precision": one_tp / max(one_tp + one_fp, 1),
    "vq_one_recall": one_tp / max(one_tp + one_fn, 1),
  }


def load_batch_payload(
  batch: dict[str, object],
  args: argparse.Namespace,
  device: torch.device,
  vqvae,
) -> tuple[object, torch.Tensor]:
  if args.vq_cache_dir is not None:
    cached = load_cached_tokens(Path(args.vq_cache_dir), batch_uid(batch), args)
    octree = batch["octree_gt"].to(device)
    return octree, cached["indices"].to(device).long()

  octree = batch["octree_gt"].to(device)
  if vqvae is None:
    raise ValueError("vqvae is required without --vq-cache-dir")
  with torch.no_grad():
    indices, _, code_depth = encode_bsq_tokens(vqvae, octree)
  if code_depth != args.depth_stop:
    raise ValueError(f"VQVAE code depth is {code_depth}, expected {args.depth_stop}.")
  return octree, indices.long()


def run_epoch(
  model: Fractal3DVAE,
  vqvae,
  loader: DataLoader,
  optimizer: torch.optim.Optimizer | None,
  device: torch.device,
  args: argparse.Namespace,
  epoch: int,
  writer: SummaryWriter | None,
  global_step: int,
  scaler: torch.amp.GradScaler | None,
) -> tuple[dict[str, float], int]:
  training = optimizer is not None
  model.train(training)
  if vqvae is not None:
    vqvae.eval()
  use_amp = bool(args.amp and device.type == "cuda")
  beta = args.beta_max * min(1.0, (epoch + 1) / max(args.beta_warmup_epochs, 1))

  totals = {key: 0.0 for key in [
    "loss", "split_loss", "vq_loss", "kl_loss", "split_accuracy",
    "split_target_rate", "split_pred_rate", "split_precision", "split_recall",
    "vq_bit_accuracy", "vq_node_exact", "vq_target_one_rate",
    "vq_pred_one_rate", "vq_one_precision", "vq_one_recall",
    "z_mu_abs_mean", "z_std_mean", "kl_per_dim", "count",
  ]}
  pbar = tqdm(loader, desc=("train" if training else "val") + f" {epoch:04d}", dynamic_ncols=True, leave=False)
  for batch in pbar:
    octree, vq_indices = load_batch_payload(batch, args, device, vqvae)
    with torch.amp.autocast("cuda", enabled=use_amp):
      mu, logvar = model.encode(octree, split_by_depth(octree, args.full_depth, args.depth_stop), vq_indices, args.depth_stop)
      z = model.reparameterize(mu, logvar) if training else mu
      s_loss, s_stats, parent_hidden = split_losses(model.decoder, octree, z, args)
      q_loss, q_stats = vq_loss(model.decoder, octree, parent_hidden, vq_indices, args)
      k_loss = model.kl_loss(mu, logvar)
      loss = s_loss + args.lambda_vq * q_loss + beta * k_loss
      z_std = torch.exp(0.5 * torch.clamp(logvar, -30.0, 20.0))

    if training:
      assert optimizer is not None and scaler is not None
      optimizer.zero_grad(set_to_none=True)
      scaler.scale(loss).backward()
      if args.grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      scaler.step(optimizer)
      scaler.update()
      global_step += 1
      if writer is not None and global_step % args.log_every == 0:
        writer.add_scalar("train/loss_total_step", float(loss.item()), global_step)
        writer.add_scalar("train/split_loss_step", float(s_loss.detach().item()), global_step)
        writer.add_scalar("train/vq_loss_step", float(q_loss.detach().item()), global_step)
        writer.add_scalar("train/kl_loss_step", float(k_loss.detach().item()), global_step)
        writer.add_scalar("train/split_precision_step", s_stats["split_precision"], global_step)
        writer.add_scalar("train/split_recall_step", s_stats["split_recall"], global_step)
        writer.add_scalar("train/vq_one_precision_step", q_stats["vq_one_precision"], global_step)
        writer.add_scalar("train/vq_one_recall_step", q_stats["vq_one_recall"], global_step)
        writer.add_scalar("train/z_mu_abs_mean_step", float(mu.detach().abs().mean().item()), global_step)
        writer.add_scalar("train/z_std_mean_step", float(z_std.detach().mean().item()), global_step)
        writer.add_scalar("train/beta_step", beta, global_step)

    totals["loss"] += float(loss.detach().item())
    totals["split_loss"] += float(s_loss.detach().item())
    totals["vq_loss"] += float(q_loss.detach().item())
    totals["kl_loss"] += float(k_loss.detach().item())
    totals["split_accuracy"] += float(s_stats["split_accuracy"])
    totals["split_target_rate"] += float(s_stats["split_target_rate"])
    totals["split_pred_rate"] += float(s_stats["split_pred_rate"])
    totals["split_precision"] += float(s_stats["split_precision"])
    totals["split_recall"] += float(s_stats["split_recall"])
    totals["vq_bit_accuracy"] += float(q_stats["vq_bit_accuracy"])
    totals["vq_node_exact"] += float(q_stats["vq_node_exact"])
    totals["vq_target_one_rate"] += float(q_stats["vq_target_one_rate"])
    totals["vq_pred_one_rate"] += float(q_stats["vq_pred_one_rate"])
    totals["vq_one_precision"] += float(q_stats["vq_one_precision"])
    totals["vq_one_recall"] += float(q_stats["vq_one_recall"])
    totals["z_mu_abs_mean"] += float(mu.detach().abs().mean().item())
    totals["z_std_mean"] += float(z_std.detach().mean().item())
    totals["kl_per_dim"] += float(k_loss.detach().item()) / max(mu.shape[-1], 1)
    totals["count"] += 1.0
    pbar.set_postfix(loss=f"{totals['loss'] / max(totals['count'], 1):.4f}")

  count = max(totals.pop("count"), 1.0)
  metrics = {key: value / count for key, value in totals.items()}
  metrics["beta"] = beta
  return metrics, global_step


def save_checkpoint(path: Path, model: Fractal3DVAE, optimizer, epoch: int, best: float, args: argparse.Namespace) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch,
    "best_val_loss": best,
    "args": vars(args),
  }, path)


def load_checkpoint(path: Path, model: Fractal3DVAE, optimizer, device: torch.device) -> tuple[int, float]:
  checkpoint = torch.load(path, map_location=device)
  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  return int(checkpoint["epoch"]) + 1, float(checkpoint.get("best_val_loss", math.inf))


def main() -> None:
  args = parse_args()
  if args.depth_stop > args.depth:
    raise ValueError("--depth-stop must be <= --depth")
  if args.vq_cache_dir is not None and args.batch_size != 1:
    raise ValueError("--vq-cache-dir currently requires --batch-size 1")
  set_seed(args.seed)
  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(output_dir / "tensorboard")
  train_loader, val_loader = build_loaders(args)

  model = Fractal3DVAE(
    dim=args.dim,
    z_dim=args.z_dim,
    encoder_layers=args.encoder_layers,
    decoder_layers=args.decoder_layers,
    encoder_patch_size=args.encoder_patch_size,
    encoder_dilation=args.encoder_dilation,
    num_heads=args.heads,
    num_vq_embed=args.num_vq_embed,
    vq_groups=args.vq_groups,
    full_depth=args.full_depth,
    max_depth=args.depth,
    dropout=args.dropout,
  ).to(device)
  vqvae = None if args.vq_cache_dir is not None else load_octgpt_vqvae(args.vqvae_ckpt, device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
  scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))

  start_epoch = 0
  best_val = math.inf
  if args.resume:
    start_epoch, best_val = load_checkpoint(Path(args.resume), model, optimizer, device)

  global_step = 0
  for epoch in range(start_epoch, args.epochs):
    train_metrics, global_step = run_epoch(
      model, vqvae, train_loader, optimizer, device, args, epoch, writer, global_step, scaler)
    for key, value in train_metrics.items():
      writer.add_scalar(f"train/{key}_epoch", value, epoch)
    print("train", epoch, " ".join(f"{k}={v:.6f}" for k, v in train_metrics.items()))

    val_metrics = None
    if val_loader is not None and epoch % args.val_every == 0:
      with torch.no_grad():
        val_metrics, _ = run_epoch(
          model, vqvae, val_loader, None, device, args, epoch, writer, global_step, scaler)
      for key, value in val_metrics.items():
        writer.add_scalar(f"val/{key}_epoch", value, epoch)
      print("val", epoch, " ".join(f"{k}={v:.6f}" for k, v in val_metrics.items()))

    score = train_metrics["loss"] if val_metrics is None else val_metrics["loss"]
    save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_val, args)
    if score < best_val:
      best_val = score
      save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_val, args)

  writer.close()


if __name__ == "__main__":
  main()
