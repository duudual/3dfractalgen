from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_embedding import AbsPosEmb


ROLE_PARENT = 0
ROLE_UNCLE = 1
ROLE_CHILD = 2


@dataclass
class FractalBlockBatch:
  """Batched parent/uncle/child sequences for one octree depth.

  The sequence layout is fixed:
    [parent] + [up to 7 uncles] + [8 children]
  """

  parent_indices: torch.Tensor
  child_indices: torch.Tensor
  embeddings: torch.Tensor
  xyz: torch.Tensor
  depth_idx: torch.Tensor
  role_ids: torch.Tensor
  padding_mask: torch.Tensor
  child_offset: int = 8

  @property
  def child_embeddings(self) -> torch.Tensor:
    return self.embeddings[:, self.child_offset:self.child_offset + 8]


class ChildARTransformer(nn.Module):
  """Small AR transformer over parent/uncle context and 8 child slots."""

  def __init__(
    self,
    dim: int,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    layer = nn.TransformerEncoderLayer(
      d_model=dim,
      nhead=num_heads,
      dim_feedforward=int(dim * mlp_ratio),
      dropout=dropout,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.layers = nn.TransformerEncoder(layer, num_layers=num_layers)
    self.norm = nn.LayerNorm(dim)
    self.register_buffer("attn_mask", self._build_attn_mask(), persistent=False)

  @staticmethod
  def _build_attn_mask() -> torch.Tensor:
    seq_len = 16
    child_offset = 8
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    # Context tokens are conditioning information only; do not let them read
    # child tokens during training, otherwise teacher-forced child labels leak
    # back into the conditioning states.
    mask[:child_offset, child_offset:] = True

    for query in range(child_offset, seq_len):
      child_query = query - child_offset
      for key in range(child_offset, seq_len):
        child_key = key - child_offset
        if child_key > child_query:
          mask[query, key] = True
    return mask

  def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    attn_mask = self.attn_mask.to(device=x.device)
    x = self.layers(x, mask=attn_mask, src_key_padding_mask=padding_mask)
    return self.norm(x)


class Fractal3DGenerator(nn.Module):
  """Parent-uncle-child AR generator for octree split and VQ tokens.

  This module predicts labels for the 8 children of each parent node at
  `parent_depth`. The caller is responsible for passing an octree that already
  contains `parent_depth + 1` nodes, which is true for teacher-forced training
  on ground-truth octrees and for generation after the previous grow step.
  """

  def __init__(
    self,
    dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    num_vq_embed: int = 256,
    vq_groups: Optional[int] = None,
    full_depth: int = 3,
    max_depth: int = 8,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_vq_embed = num_vq_embed
    self.vq_groups = vq_groups or num_vq_embed
    self.full_depth = full_depth
    self.max_depth = max_depth

    self.split_emb = nn.Embedding(2, dim)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
    self.vq_proj = nn.Linear(num_vq_embed, dim)
    self.role_emb = nn.Embedding(3, dim)
    self.pos_emb = AbsPosEmb(dim, full_depth=full_depth, max_depth=max_depth)
    self.transformer = ChildARTransformer(
      dim=dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    self.split_head = nn.Linear(dim, 2)
    self.vq_head = nn.Linear(dim, self.vq_groups * 2)
    self._init_weights()

  def _init_weights(self) -> None:
    nn.init.normal_(self.mask_token, std=0.02)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

  def _depth_index(self, depth: int, device: torch.device) -> torch.Tensor:
    value = max(0, min(depth - self.full_depth, self.max_depth - self.full_depth))
    return torch.tensor(value, dtype=torch.long, device=device)

  def _rescale_xyz(self, xyz: torch.Tensor, depth: int) -> torch.Tensor:
    max_scale = 2 ** (self.max_depth + 1)
    scale = 2 ** depth
    xyz = xyz.long() * max_scale // scale
    xyz = xyz + max_scale // scale // 2
    return xyz.float()

  def _xyz_at_depth(self, octree, depth: int) -> torch.Tensor:
    x, y, z, _ = octree.xyzb(depth)
    xyz = torch.stack([x, y, z], dim=1)
    return self._rescale_xyz(xyz, depth)

  def _child_xyz(self, octree, parent_depth: int) -> torch.Tensor:
    x, y, z, _ = octree.xyzb(parent_depth)
    parent = torch.stack([x, y, z], dim=1).long()
    offsets = torch.tensor(
      [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
       [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
      dtype=torch.long,
      device=parent.device,
    )
    child = parent[:, None, :] * 2 + offsets[None, :, :]
    return self._rescale_xyz(child.reshape(-1, 3), parent_depth + 1).view(-1, 8, 3)

  def _sibling_indices(self, octree, depth: int) -> torch.Tensor:
    nnum = int(octree.nnum[depth])
    device = octree.device
    if depth <= self.full_depth:
      starts = torch.arange(0, nnum, 8, device=device)
    else:
      prev_children = octree.children[depth - 1]
      starts = prev_children[prev_children >= 0].long()
      starts = starts[starts < nnum]
    group = torch.empty((nnum, 8), dtype=torch.long, device=device).fill_(-1)
    for start in starts.tolist():
      end = min(start + 8, nnum)
      idx = torch.arange(start, end, dtype=torch.long, device=device)
      group[idx, :idx.numel()] = idx
    missing = group[:, 0] < 0
    if missing.any():
      group[missing, 0] = torch.arange(nnum, device=device)[missing]
    return group

  def _child_indices(self, octree, parent_depth: int) -> torch.Tensor:
    n_parent = int(octree.nnum[parent_depth])
    n_child = int(octree.nnum[parent_depth + 1])
    device = octree.device
    children = octree.children[parent_depth]
    out = torch.empty((n_parent, 8), dtype=torch.long, device=device).fill_(-1)
    valid_parent = children >= 0
    for parent_idx in torch.nonzero(valid_parent, as_tuple=False).flatten().tolist():
      start = int(children[parent_idx].item())
      end = min(start + 8, n_child)
      if start < n_child:
        out[parent_idx, :end - start] = torch.arange(start, end, device=device)
    return out

  def _teacher_forced_child_embeddings(
    self,
    split_targets: Optional[torch.Tensor] = None,
    vq_codes: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if split_targets is None and vq_codes is None:
      raise ValueError("split_targets or vq_codes must be provided")
    if split_targets is not None:
      bsz = split_targets.shape[0]
      embeddings = self.mask_token.expand(bsz, 8, -1).clone()
      if split_targets.shape[1] > 1:
        prev = split_targets[:, :-1].clamp(min=0)
        embeddings[:, 1:] = self.split_emb(prev)
      return embeddings

    assert vq_codes is not None
    bsz = vq_codes.shape[0]
    embeddings = self.mask_token.expand(bsz, 8, -1).clone()
    if vq_codes.shape[1] > 1:
      embeddings[:, 1:] = self.vq_proj(vq_codes[:, :-1])
    return embeddings

  def build_blocks(
    self,
    octree,
    parent_depth: int,
    parent_tokens: Optional[torch.Tensor] = None,
    child_input_embeddings: Optional[torch.Tensor] = None,
  ) -> FractalBlockBatch:
    if parent_depth + 1 > octree.depth:
      raise ValueError("octree must already contain parent_depth + 1 nodes")

    device = octree.device
    n_parent = int(octree.nnum[parent_depth])
    if parent_tokens is None:
      parent_tokens = torch.zeros(n_parent, self.dim, device=device)
    if child_input_embeddings is None:
      child_input_embeddings = self.mask_token.expand(n_parent, 8, -1).clone()

    sibling_idx = self._sibling_indices(octree, parent_depth)
    parent_idx = torch.arange(n_parent, dtype=torch.long, device=device)
    uncle_idx = torch.empty((n_parent, 7), dtype=torch.long, device=device).fill_(-1)
    for row in range(n_parent):
      sibs = sibling_idx[row]
      sibs = sibs[(sibs >= 0) & (sibs != row)]
      uncle_idx[row, :min(7, sibs.numel())] = sibs[:7]

    seq = torch.zeros(n_parent, 16, self.dim, device=device)
    seq[:, 0] = parent_tokens[parent_idx]
    valid_uncle = uncle_idx >= 0
    safe_uncle = uncle_idx.clamp(min=0)
    seq[:, 1:8] = parent_tokens[safe_uncle]
    seq[:, 1:8] = torch.where(valid_uncle.unsqueeze(-1), seq[:, 1:8], 0)
    seq[:, 8:16] = child_input_embeddings

    parent_xyz = self._xyz_at_depth(octree, parent_depth)
    child_xyz = self._child_xyz(octree, parent_depth)
    xyz = torch.zeros(n_parent, 16, 3, device=device)
    xyz[:, 0] = parent_xyz[parent_idx]
    xyz[:, 1:8] = parent_xyz[safe_uncle]
    xyz[:, 1:8] = torch.where(valid_uncle.unsqueeze(-1), xyz[:, 1:8], 0)
    xyz[:, 8:16] = child_xyz

    depth_idx = torch.empty(n_parent, 16, dtype=torch.long, device=device)
    depth_idx[:, :8] = self._depth_index(parent_depth, device)
    depth_idx[:, 8:] = self._depth_index(parent_depth + 1, device)

    role_ids = torch.empty(n_parent, 16, dtype=torch.long, device=device)
    role_ids[:, 0] = ROLE_PARENT
    role_ids[:, 1:8] = ROLE_UNCLE
    role_ids[:, 8:] = ROLE_CHILD

    padding_mask = torch.zeros(n_parent, 16, dtype=torch.bool, device=device)
    padding_mask[:, 1:8] = ~valid_uncle

    child_idx = self._child_indices(octree, parent_depth)
    return FractalBlockBatch(
      parent_indices=parent_idx,
      child_indices=child_idx,
      embeddings=seq,
      xyz=xyz,
      depth_idx=depth_idx,
      role_ids=role_ids,
      padding_mask=padding_mask,
    )

  def encode_blocks(self, blocks: FractalBlockBatch) -> torch.Tensor:
    bsz, seq_len, _ = blocks.embeddings.shape
    pos_ctx = SimpleNamespace(
      xyz=blocks.xyz.reshape(bsz * seq_len, 3),
      depth_idx=blocks.depth_idx.reshape(bsz * seq_len),
    )
    pos = self.pos_emb(blocks.embeddings.reshape(bsz * seq_len, self.dim), pos_ctx)
    pos = pos.view(bsz, seq_len, self.dim)
    x = blocks.embeddings + pos + self.role_emb(blocks.role_ids)
    return self.transformer(x, blocks.padding_mask)

  def forward_split(
    self,
    octree,
    parent_depth: int,
    split_targets: Optional[torch.Tensor] = None,
    parent_tokens: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, FractalBlockBatch]:
    child_inputs = None
    if split_targets is not None:
      child_inputs = self._teacher_forced_child_embeddings(split_targets=split_targets)
    blocks = self.build_blocks(octree, parent_depth, parent_tokens, child_inputs)
    hidden = self.encode_blocks(blocks)
    logits = self.split_head(hidden[:, blocks.child_offset:blocks.child_offset + 8])
    return logits, blocks

  def forward_vq(
    self,
    octree,
    parent_depth: int,
    vq_codes: Optional[torch.Tensor] = None,
    parent_tokens: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, FractalBlockBatch]:
    child_inputs = None
    if vq_codes is not None:
      child_inputs = self._teacher_forced_child_embeddings(vq_codes=vq_codes)
    blocks = self.build_blocks(octree, parent_depth, parent_tokens, child_inputs)
    hidden = self.encode_blocks(blocks)
    logits = self.vq_head(hidden[:, blocks.child_offset:blocks.child_offset + 8])
    logits = logits.view(logits.shape[0], 8, self.vq_groups, 2)
    return logits, blocks

  def split_loss(
    self,
    octree,
    parent_depth: int,
    split_targets: torch.Tensor,
    parent_tokens: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    logits, _ = self.forward_split(octree, parent_depth, split_targets, parent_tokens)
    return F.cross_entropy(logits.reshape(-1, 2), split_targets.reshape(-1).long())

  def vq_loss(
    self,
    octree,
    parent_depth: int,
    vq_indices: torch.Tensor,
    vq_codes: torch.Tensor,
    parent_tokens: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    logits, _ = self.forward_vq(octree, parent_depth, vq_codes, parent_tokens)
    return F.cross_entropy(logits.reshape(-1, 2), vq_indices.reshape(-1).long())

  @torch.no_grad()
  def generate_split_step(
    self,
    octree,
    parent_depth: int,
    parent_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    n_parent = int(octree.nnum[parent_depth])
    child_inputs = self.mask_token.expand(n_parent, 8, -1).clone()
    sampled = torch.empty(n_parent, 8, dtype=torch.long, device=octree.device)
    for child_id in range(8):
      blocks = self.build_blocks(octree, parent_depth, parent_tokens, child_inputs)
      hidden = self.encode_blocks(blocks)
      logits = self.split_head(hidden[:, blocks.child_offset + child_id])
      probs = F.softmax(logits / temperature, dim=-1)
      token = torch.multinomial(probs, num_samples=1).squeeze(1)
      sampled[:, child_id] = token
      child_inputs[:, child_id] = self.split_emb(token)
    return sampled, child_inputs

  @torch.no_grad()
  def generate_vq_step(
    self,
    octree,
    parent_depth: int,
    vqvae,
    parent_tokens: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_parent = int(octree.nnum[parent_depth])
    child_inputs = self.mask_token.expand(n_parent, 8, -1).clone()
    vq_indices = torch.empty(
      n_parent, 8, self.vq_groups, dtype=torch.long, device=octree.device)
    vq_codes = torch.empty(
      n_parent, 8, self.num_vq_embed, dtype=child_inputs.dtype, device=octree.device)
    for child_id in range(8):
      blocks = self.build_blocks(octree, parent_depth, parent_tokens, child_inputs)
      hidden = self.encode_blocks(blocks)
      logits = self.vq_head(hidden[:, blocks.child_offset + child_id])
      logits = logits.view(n_parent, self.vq_groups, 2)
      probs = F.softmax(logits / temperature, dim=-1)
      token = torch.multinomial(probs.reshape(-1, 2), num_samples=1)
      token = token.view(n_parent, self.vq_groups)
      zq = vqvae.quantizer.extract_code(token)
      vq_indices[:, child_id] = token
      vq_codes[:, child_id] = zq
      child_inputs[:, child_id] = self.vq_proj(zq)
    return vq_indices, vq_codes, child_inputs
