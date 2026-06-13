from __future__ import annotations

import argparse
import copy
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

from fractal3d import Fractal3DVAE, OctreeConfig, ShapeOctreeDataset, collate_shapes  # noqa: E402
from fractal3d.octgpt_vqvae import load_octgpt_vqvae  # noqa: E402
from ocnn.octree import Octree, init_octree as ocnn_init_octree  # noqa: E402
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
    "--z-scale",
    type=float,
    default=1.0,
    help="Scale standard-normal latent samples before decoding.")
  parser.add_argument(
    "--posterior-data",
    default=None,
    help="Dataset root used to build a posterior latent bank for sampling.")
  parser.add_argument(
    "--posterior-filelist",
    default=None,
    help="Filelist used to build the posterior latent bank.")
  parser.add_argument(
    "--posterior-vq-cache-dir",
    default=None,
    help="Cached VQ tokens used to encode posterior bank samples.")
  parser.add_argument(
    "--posterior-noise",
    type=float,
    default=0.0,
    help="Sample z = mu + posterior_noise * std * eps from the posterior bank.")
  parser.add_argument(
    "--sample-tokens",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument(
    "--decode-sdf",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Decode sampled VQ codes with the frozen VQ-VAE and export surface PLY.")
  parser.add_argument(
    "--vqvae-update-octree",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Allow the frozen VQ-VAE decoder to update octree structure during SDF decoding.")
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
  missing, unexpected = model.load_state_dict(state, strict=False)
  if missing:
    print(f"missing checkpoint keys: {missing}")
  if unexpected:
    print(f"unexpected checkpoint keys: {unexpected}")
  model.eval()
  return model, checkpoint


def load_cached_tokens(cache_dir: Path, uid: str) -> torch.Tensor:
  cached = torch.load(cache_dir / f"{uid}.pt", map_location="cpu", weights_only=False)
  return cached["indices"].long()


def split_by_depth(octree, full_depth: int, depth_stop: int) -> dict[int, torch.Tensor]:
  return {
    depth: octree.children[depth].ge(0).long()
    for depth in range(full_depth, depth_stop)
  }


def build_posterior_bank(
  model: Fractal3DVAE,
  data: str,
  filelist: str | None,
  cache_dir: str,
  full_depth: int,
  depth_stop: int,
  decode_depth: int,
  saved_args: dict,
  device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
  config = OctreeConfig(
    depth=decode_depth,
    full_depth=full_depth,
    points_scale=float(saved_args.get("points_scale", 1.0)),
    max_points=int(saved_args.get("max_points", 120000)),
    sample_seed=int(saved_args.get("sample_seed", 0)),
  )
  dataset = ShapeOctreeDataset(data, config=config, filelist=filelist)
  mus: list[torch.Tensor] = []
  stds: list[torch.Tensor] = []
  cache_path = Path(cache_dir)
  with torch.no_grad():
    for sample in dataset:
      batch = collate_shapes([sample])
      uid = batch["uid"]
      if isinstance(uid, (list, tuple)):
        uid = uid[0]
      octree = batch["octree_gt"].to(device)
      vq_indices = load_cached_tokens(cache_path, str(uid)).to(device)
      mu, logvar = model.encode(
        octree, split_by_depth(octree, full_depth, depth_stop), vq_indices, depth_stop)
      mus.append(mu.detach().cpu())
      stds.append(torch.exp(0.5 * logvar).detach().cpu())
  return torch.cat(mus, dim=0), torch.cat(stds, dim=0)


def sample_latent(
  z_dim: int,
  device: torch.device,
  z_scale: float,
  posterior_bank: tuple[torch.Tensor, torch.Tensor] | None,
  posterior_noise: float,
) -> torch.Tensor:
  if posterior_bank is None:
    return torch.randn(1, z_dim, device=device) * z_scale
  mus, stds = posterior_bank
  idx = torch.randint(mus.shape[0], (1,)).item()
  z = mus[idx:idx + 1].to(device)
  if posterior_noise > 0:
    z = z + posterior_noise * stds[idx:idx + 1].to(device) * torch.randn_like(z)
  return z


def sample_binary_logits(logits: torch.Tensor, temperature: float, sample_tokens: bool) -> torch.Tensor:
  if sample_tokens:
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.reshape(-1, logits.shape[-1]), num_samples=1).view(logits.shape[:-1])
  return logits.argmax(dim=-1)


def init_generated_octree(full_depth: int, depth_stop: int, device: torch.device) -> Octree:
  return ocnn_init_octree(
    depth=depth_stop, full_depth=full_depth, batch_size=1, device=device)


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
  octree = init_generated_octree(full_depth, depth_stop, device)
  split_by_depth: dict[int, torch.Tensor] = {}
  hidden = decoder.bootstrap_hidden_from_z(
    octree, z, full_depth - 1, parallel=False)

  for parent_depth in range(full_depth - 1, depth_stop - 1):
    logits, child_hidden, child_indices = decoder.forward_split(
      octree, parent_depth, hidden, parallel=False, z=z)
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
    octree, depth_stop - 1, hidden, parallel=False, z=z)
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
  decode_depth: int,
  octree,
  num_query: int,
  num_surface: int,
  chunk_size: int,
  device: torch.device,
  update_octree: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
  octree_for_decode = copy.deepcopy(octree)
  min_full_depth = code_depth - vqvae.decoder.encoder_stages + 1
  if min_full_depth < 0:
    raise ValueError(
      f"Invalid VQ-VAE code depth {code_depth} for "
      f"{vqvae.decoder.encoder_stages} encoder stages.")
  octree_for_decode.full_depth = min(int(octree.full_depth), min_full_depth)

  # OctGPT's VQ-VAE is trained with code_depth = final_depth - delta_depth.
  # The decoder still needs placeholder graph levels up to final depth; with
  # update_octree=True it predicts the missing fine split structure itself.
  for depth in range(code_depth, decode_depth):
    split = torch.zeros(
      int(octree_for_decode.nnum[depth]), dtype=torch.int32, device=device)
    octree_for_decode.octree_split(split, depth)
    octree_for_decode.octree_grow(depth + 1, update_neigh=True)

  doctree = OctreeD(octree_for_decode)
  octree_out = copy.deepcopy(doctree)
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
  decode_depth = int(saved.get("depth", depth_stop))
  z_dim = int(saved.get("z_dim", 128))
  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)
  posterior_bank = None
  if args.posterior_data is not None:
    if args.posterior_vq_cache_dir is None:
      raise ValueError("--posterior-vq-cache-dir is required with --posterior-data")
    posterior_bank = build_posterior_bank(
      model,
      args.posterior_data,
      args.posterior_filelist,
      args.posterior_vq_cache_dir,
      full_depth,
      depth_stop,
      decode_depth,
      saved,
      device,
    )
    print(
      f"posterior_bank_size={posterior_bank[0].shape[0]} "
      f"mu_abs={posterior_bank[0].abs().mean().item():.6f} "
      f"std_mean={posterior_bank[1].mean().item():.6f}")

  for index in range(args.num_samples):
    z = sample_latent(
      z_dim, device, args.z_scale, posterior_bank, args.posterior_noise)
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
      surface = torch.empty(0, 3)
      surface_sdf = torch.empty(0)
      if args.decode_sdf:
        surface, surface_sdf = sdf_surface_points(
          vqvae,
          codes,
          depth_stop,
          decode_depth,
          octree,
          args.sdf_points,
          args.surface_points,
          args.chunk_size,
          device,
          args.vqvae_update_octree,
        )

    stem = f"{index:04d}"
    torch.save({
      "z": z.detach().cpu(),
      "vq_indices": vq_indices.detach().cpu(),
      "split_by_depth": split_by_depth,
      "code_depth": depth_stop,
      "decode_depth": decode_depth,
      "surface_points": surface,
      "surface_sdf": surface_sdf,
      "checkpoint_epoch": checkpoint.get("epoch"),
      "checkpoint_best_val_loss": checkpoint.get("best_val_loss", math.nan),
      "posterior_sampling": posterior_bank is not None,
      "posterior_noise": args.posterior_noise,
    }, output_dir / f"{stem}_sample.pt")
    if args.decode_sdf:
      write_ply(output_dir / f"{stem}_surface.ply", surface)
    print(
      f"sample={index} nodes_depth_stop={int(octree.nnum[depth_stop])} "
      f"splits={{{', '.join(f'{depth}: {int(split.sum())}' for depth, split in split_by_depth.items())}}} "
      f"saved={output_dir / f'{stem}_sample.pt'}")


if __name__ == "__main__":
  main()
