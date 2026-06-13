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

DEFAULT_VAE_CKPT = PROJECT_ROOT / "outputs" / "vae_train" / "best.pt"
DEFAULT_VQVAE_CKPT = PROJECT_ROOT / "ckpt" / "vqvae_large_im5_uncond_bsq32.pth"

from fractal3d import Fractal3DVAE  # noqa: E402
from fractal3d.octgpt_vqvae import load_octgpt_vqvae  # noqa: E402
from ocnn.octree import Octree  # noqa: E402
from ognn.octreed import OctreeD  # noqa: E402


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Sample OctFormer-VAE 3DFractalGen.")
  parser.add_argument("--vae-ckpt", default=str(DEFAULT_VAE_CKPT))
  parser.add_argument("--vqvae-ckpt", default=str(DEFAULT_VQVAE_CKPT))
  parser.add_argument("--output-dir", default="outputs/vae_samples")
  parser.add_argument("--num-samples", type=int, default=4)
  parser.add_argument("--temperature-split", type=float, default=1.0)
  parser.add_argument("--temperature-vq", type=float, default=1.0)
  parser.add_argument(
    "--sample-tokens",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def model_args_from_checkpoint(checkpoint: dict) -> dict:
  saved = checkpoint.get("args", {})
  return {
    "dim": int(saved.get("dim", 192)),
    "z_dim": int(saved.get("z_dim", 128)),
    "encoder_layers": int(saved.get("encoder_layers", 6)),
    "decoder_layers": int(saved.get("decoder_layers", 6)),
    "encoder_patch_size": int(saved.get("encoder_patch_size", 1024)),
    "encoder_dilation": int(saved.get("encoder_dilation", 8)),
    "num_heads": int(saved.get("heads", 6)),
    "num_vq_embed": int(saved.get("num_vq_embed", 32)),
    "vq_groups": int(saved.get("vq_groups", 32)),
    "full_depth": int(saved.get("full_depth", 3)),
    "max_depth": int(saved.get("depth", 8)),
    "dropout": float(saved.get("dropout", 0.1)),
  }


def load_vae(path: str | Path, device: torch.device) -> tuple[Fractal3DVAE, dict]:
  checkpoint = torch.load(path, map_location=device)
  model = Fractal3DVAE(**model_args_from_checkpoint(checkpoint)).to(device)
  state = checkpoint["model"] if "model" in checkpoint else checkpoint
  model.load_state_dict(state)
  model.eval()
  return model, checkpoint


def sample_binary_logits(logits: torch.Tensor, temperature: float, sample_tokens: bool) -> torch.Tensor:
  if sample_tokens:
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.reshape(-1, logits.shape[-1]), num_samples=1).view(logits.shape[:-1])
  return logits.argmax(dim=-1)


def init_octree(full_depth: int, depth_stop: int, device: torch.device) -> Octree:
  octree = Octree(depth=depth_stop, full_depth=full_depth, device=device)
  for depth in range(full_depth + 1):
    octree.octree_grow_full(depth, update_neigh=True)
  return octree


@torch.no_grad()
def sample_structure_and_vq(
  model: Fractal3DVAE,
  z: torch.Tensor,
  full_depth: int,
  depth_stop: int,
  temperature_split: float,
  temperature_vq: float,
  sample_tokens: bool,
  device: torch.device,
) -> tuple[Octree, torch.Tensor, dict[int, torch.Tensor]]:
  decoder = model.decoder
  octree = init_octree(full_depth, depth_stop, device)
  split_by_depth: dict[int, torch.Tensor] = {}
  hidden = decoder.bootstrap_hidden_from_z(
    octree, z, full_depth - 1, parallel=False)

  for parent_depth in range(full_depth - 1, depth_stop - 1):
    logits, child_hidden, child_indices = decoder.forward_split(
      octree, parent_depth, hidden, parallel=False)
    sampled = sample_binary_logits(logits, temperature_split, sample_tokens).int()
    target_depth = parent_depth + 1
    split = torch.zeros(int(octree.nnum[target_depth]), dtype=torch.int32, device=device)
    valid = child_indices >= 0
    safe = child_indices.clamp(min=0)
    split[safe[valid]] = sampled[valid]
    split_by_depth[target_depth] = split.detach().cpu()
    octree.octree_split(split, target_depth)
    if target_depth < depth_stop:
      octree.octree_grow(target_depth + 1, update_neigh=True)
      hidden = decoder.scatter_child_hidden(
        child_hidden, child_indices, int(octree.nnum[target_depth]))

  logits, _, child_indices = decoder.forward_vq(
    octree, depth_stop - 1, hidden, parallel=False)
  sampled_vq = sample_binary_logits(logits, temperature_vq, sample_tokens).long()
  indices = torch.zeros(
    int(octree.nnum[depth_stop]), decoder.vq_groups, dtype=torch.long, device=device)
  valid = child_indices >= 0
  safe = child_indices.clamp(min=0)
  indices[safe[valid]] = sampled_vq[valid]
  return octree, indices, split_by_depth


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


@torch.no_grad()
def sdf_surface_points(
  vqvae,
  codes: torch.Tensor,
  code_depth: int,
  octree,
  num_query: int,
  num_surface: int,
  chunk_size: int,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  for depth in range(code_depth, int(octree.depth)):
    split = torch.zeros(int(octree.nnum[depth]), dtype=torch.int32, device=device)
    octree.octree_split(split, depth)
    if depth + 1 <= int(octree.depth):
      octree.octree_grow(depth + 1, update_neigh=True)

  doctree = OctreeD(octree)
  decoded = vqvae.decode_code(
    codes, code_depth, doctree, OctreeD(octree), pos=None, update_octree=True)
  neural_mpu = decoded["neural_mpu"]

  keep_points: list[torch.Tensor] = []
  keep_sdf: list[torch.Tensor] = []
  remaining = num_query
  while remaining > 0:
    count = min(chunk_size, remaining)
    xyz = torch.rand(count, 3, device=device) * 2.0 - 1.0
    batch_id = torch.zeros(count, 1, device=device)
    sdf = neural_mpu(torch.cat([xyz, batch_id], dim=1))
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


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  model, checkpoint = load_vae(args.vae_ckpt, device)
  saved = checkpoint.get("args", {})
  full_depth = int(saved.get("full_depth", 3))
  depth_stop = int(saved.get("depth_stop", 6))
  z_dim = int(saved.get("z_dim", 128))
  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)

  for index in range(args.num_samples):
    z = torch.randn(1, z_dim, device=device)
    with torch.no_grad():
      octree, vq_indices, split_by_depth = sample_structure_and_vq(
        model,
        z,
        full_depth,
        depth_stop,
        args.temperature_split,
        args.temperature_vq,
        args.sample_tokens,
        device,
      )
      codes = vqvae.quantizer.extract_code(vq_indices)
      surface, surface_sdf = sdf_surface_points(
        vqvae,
        codes,
        depth_stop,
        octree,
        args.sdf_points,
        args.surface_points,
        args.chunk_size,
        device,
      )

    stem = f"{index:04d}"
    torch.save({
      "z": z.detach().cpu(),
      "vq_indices": vq_indices.detach().cpu(),
      "split_by_depth": split_by_depth,
      "surface_points": surface,
      "surface_sdf": surface_sdf,
      "checkpoint_epoch": checkpoint.get("epoch"),
      "checkpoint_best_val_loss": checkpoint.get("best_val_loss", math.nan),
    }, output_dir / f"{stem}_sample.pt")
    write_ply(output_dir / f"{stem}_surface.ply", surface)
    print(
      f"sample={index} nodes_depth_stop={int(octree.nnum[depth_stop])} "
      f"saved={output_dir / f'{stem}_surface.ply'}")


if __name__ == "__main__":
  main()
