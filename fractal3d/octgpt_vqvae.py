from __future__ import annotations

import sys
from pathlib import Path

import torch


class AttrDict(dict):
  """Small dict that also supports attribute access for OctGPT builders."""

  def __getattr__(self, key: str):
    try:
      return self[key]
    except KeyError as exc:
      raise AttributeError(key) from exc


def load_octgpt_vqvae(
  octgpt_root: str | Path,
  ckpt_path: str | Path,
  device: torch.device | str,
) -> torch.nn.Module:
  octgpt_root = Path(octgpt_root).resolve()
  ckpt_path = Path(ckpt_path).resolve()
  if str(octgpt_root) not in sys.path:
    sys.path.insert(0, str(octgpt_root))

  from utils import builder  # type: ignore

  flags = AttrDict({
    "name": "vqvae_large",
    "in_channels": 4,
    "embedding_channels": 32,
    "quantizer_type": "bsq",
    "feature": "ND",
  })
  vqvae = builder.build_vae_model(flags)
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
  vqvae.load_state_dict(checkpoint)
  vqvae.to(device)
  vqvae.eval()
  for param in vqvae.parameters():
    param.requires_grad_(False)
  return vqvae


@torch.no_grad()
def encode_bsq_tokens(vqvae: torch.nn.Module, octree) -> tuple[torch.Tensor, torch.Tensor, int]:
  code = vqvae.extract_code(octree)
  zq, indices, _ = vqvae.quantizer(code)
  code_depth = octree.depth - vqvae.encoder.delta_depth
  return indices.long(), zq, int(code_depth)
