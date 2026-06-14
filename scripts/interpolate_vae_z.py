from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(SCRIPTS_ROOT) not in sys.path:
  sys.path.insert(0, str(SCRIPTS_ROOT))

from fractal3d.octgpt_vqvae import load_octgpt_vqvae  # noqa: E402
from sample_vae import (  # noqa: E402
  build_posterior_bank,
  create_mesh,
  decode_vqvae,
  load_vae,
  sample_binary_logits,
  sample_structure_and_vq,
  sdf_surface_points_from_mpu,
  write_ply,
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Interpolate between two posterior VAE latents and decode samples.")
  parser.add_argument("--vae-ckpt", required=True)
  parser.add_argument(
    "--vqvae-ckpt", default="ckpt/vqvae_large_im5_uncond_bsq32.pth")
  parser.add_argument("--data", default="data/02691156")
  parser.add_argument("--filelist", required=True)
  parser.add_argument("--vq-cache-dir", required=True)
  parser.add_argument("--output-dir", default="outputs/vae_z_interp")
  parser.add_argument("--index-a", type=int, default=0)
  parser.add_argument("--index-b", type=int, default=1)
  parser.add_argument("--uid-a", default=None)
  parser.add_argument("--uid-b", default=None)
  parser.add_argument("--steps", type=int, default=9)
  parser.add_argument("--temperature-split", type=float, default=0.7)
  parser.add_argument("--temperature-vq", type=float, default=0.7)
  parser.add_argument(
    "--sample-tokens",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Sample split/VQ tokens. Default is deterministic argmax.")
  parser.add_argument("--posterior-noise", type=float, default=0.0)
  parser.add_argument("--sdf-points", type=int, default=200000)
  parser.add_argument("--surface-points", type=int, default=20000)
  parser.add_argument("--chunk-size", type=int, default=20000)
  parser.add_argument(
    "--export-ply",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Also export top-k SDF surface point PLY files.")
  parser.add_argument(
    "--export-mesh",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--mesh-resolution", type=int, default=300)
  parser.add_argument("--mesh-level", type=float, default=0.002)
  parser.add_argument("--sdf-scale", type=float, default=0.9)
  parser.add_argument("--mesh-scale", type=float, default=None)
  parser.add_argument(
    "--vqvae-update-octree",
    action=argparse.BooleanOptionalAction,
    default=True)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return parser.parse_args()


def resolve_index(uids: list[str], uid: str | None, fallback: int) -> int:
  if uid is None:
    return fallback
  if uid not in uids:
    raise ValueError(f"uid {uid!r} is not in the posterior filelist")
  return uids.index(uid)


def main() -> None:
  args = parse_args()
  if args.steps < 2:
    raise ValueError("--steps must be at least 2")

  device = torch.device(args.device)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  model, checkpoint = load_vae(args.vae_ckpt, device)
  saved = checkpoint.get("args", {})
  full_depth = int(saved.get("full_depth", 3))
  depth_stop = int(saved.get("depth_stop", 6))
  decode_depth = int(saved.get("depth", depth_stop))
  parallel_child = bool(saved.get("parallel_child_train", True))
  mesh_scale = args.mesh_scale
  if mesh_scale is None:
    mesh_scale = float(saved.get("points_scale", 1.0))

  posterior_bank = build_posterior_bank(
    model,
    args.data,
    args.filelist,
    args.vq_cache_dir,
    full_depth,
    depth_stop,
    decode_depth,
    saved,
    device,
  )
  mus, stds, uids = posterior_bank
  index_a = resolve_index(uids, args.uid_a, args.index_a)
  index_b = resolve_index(uids, args.uid_b, args.index_b)
  if index_a < 0 or index_a >= mus.shape[0]:
    raise ValueError(f"--index-a {index_a} is out of range [0, {mus.shape[0]})")
  if index_b < 0 or index_b >= mus.shape[0]:
    raise ValueError(f"--index-b {index_b} is out of range [0, {mus.shape[0]})")

  mu_a = mus[index_a:index_a + 1]
  mu_b = mus[index_b:index_b + 1]
  std_a = stds[index_a:index_a + 1]
  std_b = stds[index_b:index_b + 1]
  vqvae = load_octgpt_vqvae(args.vqvae_ckpt, device)

  print(f"interp_a index={index_a} uid={uids[index_a]}")
  print(f"interp_b index={index_b} uid={uids[index_b]}")
  print(
    f"mu_l1_distance={(mu_a - mu_b).abs().mean().item():.6f} "
    f"std_a={std_a.mean().item():.6f} std_b={std_b.mean().item():.6f}")

  for step in range(args.steps):
    t = step / (args.steps - 1)
    z_cpu = (1.0 - t) * mu_a + t * mu_b
    if args.posterior_noise > 0:
      std_cpu = (1.0 - t) * std_a + t * std_b
      z_cpu = z_cpu + args.posterior_noise * std_cpu * torch.randn_like(z_cpu)
    z = z_cpu.to(device)

    with torch.no_grad():
      octree, vq_indices, split_by_depth = sample_structure_and_vq(
        model,
        z,
        full_depth,
        depth_stop,
        args.temperature_split,
        args.temperature_vq,
        args.sample_tokens,
        parallel_child,
        device,
      )
      codes = vqvae.quantizer.extract_code(vq_indices)
      decoded = decode_vqvae(
        vqvae,
        codes,
        depth_stop,
        decode_depth,
        octree,
        device,
        args.vqvae_update_octree,
      )
      surface = torch.empty(0, 3)
      surface_sdf = torch.empty(0)
      if args.export_ply:
        surface, surface_sdf = sdf_surface_points_from_mpu(
          decoded["neural_mpu"],
          args.sdf_points,
          args.surface_points,
          args.chunk_size,
          device,
        )

    stem = f"{step:02d}_t{t:.3f}"
    if args.export_mesh:
      create_mesh(
        decoded["neural_mpu"],
        output_dir / f"{stem}.obj",
        args.mesh_resolution,
        args.mesh_level,
        -args.sdf_scale,
        args.sdf_scale,
        mesh_scale,
        device,
      )
    if args.export_ply:
      write_ply(output_dir / f"{stem}_surface.ply", surface)

    torch.save({
      "z": z_cpu,
      "t": t,
      "uid_a": uids[index_a],
      "uid_b": uids[index_b],
      "index_a": index_a,
      "index_b": index_b,
      "vq_indices": vq_indices.detach().cpu(),
      "split_by_depth": split_by_depth,
      "surface_points": surface,
      "surface_sdf": surface_sdf,
      "code_depth": depth_stop,
      "decode_depth": decode_depth,
    }, output_dir / f"{stem}_sample.pt")

    print(
      f"step={step} t={t:.3f} nodes_depth_stop={int(octree.nnum[depth_stop])} "
      f"splits={{{', '.join(f'{d}: {int(s.sum())}' for d, s in split_by_depth.items())}}} "
      f"saved={output_dir / f'{stem}.obj'}")


if __name__ == "__main__":
  main()
