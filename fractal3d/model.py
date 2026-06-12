from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


ROLE_PARENT = 0
ROLE_UNCLE = 1


class LocalOctantPosEmb(nn.Module):
  """Learned local octant position embedding for OCNN child order."""

  def __init__(self, dim: int) -> None:
    super().__init__()
    self.emb_x = nn.Parameter(torch.zeros(dim))
    self.emb_y = nn.Parameter(torch.zeros(dim))
    self.emb_z = nn.Parameter(torch.zeros(dim))

  @staticmethod
  def octant_bits(child_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_bit = child_id & 1
    y_bit = (child_id >> 1) & 1
    z_bit = (child_id >> 2) & 1
    return x_bit, y_bit, z_bit

  def forward(self, child_id: torch.Tensor) -> torch.Tensor:
    x_bit, y_bit, z_bit = self.octant_bits(child_id.long())
    sx = (2 * x_bit - 1).to(dtype=self.emb_x.dtype).unsqueeze(-1)
    sy = (2 * y_bit - 1).to(dtype=self.emb_y.dtype).unsqueeze(-1)
    sz = (2 * z_bit - 1).to(dtype=self.emb_z.dtype).unsqueeze(-1)
    return sx * self.emb_x + sy * self.emb_y + sz * self.emb_z


class ChildARTransformer(nn.Module):
  """Small transformer used as an AR hidden-state transition."""

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

  def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    x = self.layers(x, src_key_padding_mask=padding_mask)
    return self.norm(x)


class Fractal3DGenerator(nn.Module):
  """Parent/sibling hidden-state AR generator for octree split and VQ tokens."""

  def __init__(
    self,
    dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
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

    self.seed_token = nn.Parameter(torch.zeros(1, dim))
    self.role_emb = nn.Embedding(2, dim)
    self.local_pos_emb = LocalOctantPosEmb(dim)
    self.transformer = ChildARTransformer(
      dim=dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    self.split_head = nn.Linear(dim, 2)
    self.vq_head = nn.Linear(dim, self.vq_groups * 2)
    self._init_weights()

  def _init_weights(self) -> None:
    nn.init.normal_(self.seed_token, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_x, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_y, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_z, std=0.02)
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

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

  def initial_hidden(self, octree, depth: int | None = None) -> torch.Tensor:
    depth = self.full_depth if depth is None else depth
    nnum = int(octree.nnum[depth])
    device = octree.device
    child_id = torch.arange(nnum, device=device) % 8
    return self.seed_token.to(device).expand(nnum, -1) + self.local_pos_emb(child_id)

  def _context_tokens(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    n_parent = int(octree.nnum[parent_depth])
    if parent_hidden.shape[0] != n_parent:
      raise ValueError(
        f"parent_hidden has {parent_hidden.shape[0]} nodes, expected {n_parent}.")

    sibling_idx = self._sibling_indices(octree, parent_depth)
    parent_idx = torch.arange(n_parent, dtype=torch.long, device=octree.device)
    uncle_idx = torch.empty((n_parent, 7), dtype=torch.long, device=octree.device).fill_(-1)
    for row in range(n_parent):
      sibs = sibling_idx[row]
      sibs = sibs[(sibs >= 0) & (sibs != row)]
      uncle_idx[row, :min(7, sibs.numel())] = sibs[:7]

    valid_uncle = uncle_idx >= 0
    safe_uncle = uncle_idx.clamp(min=0)
    context = torch.zeros(n_parent, 8, self.dim, device=octree.device)
    context[:, 0] = parent_hidden[parent_idx] + self.role_emb.weight[ROLE_PARENT]
    context[:, 1:8] = parent_hidden[safe_uncle] + self.role_emb.weight[ROLE_UNCLE]
    context[:, 1:8] = torch.where(valid_uncle.unsqueeze(-1), context[:, 1:8], 0)

    padding_mask = torch.zeros(n_parent, 8, dtype=torch.bool, device=octree.device)
    padding_mask[:, 1:8] = ~valid_uncle
    return context, padding_mask

  def forward_children(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    context, context_padding = self._context_tokens(octree, parent_depth, parent_hidden)
    n_parent = context.shape[0]
    child_outputs: list[torch.Tensor] = []
    child_id = torch.arange(8, dtype=torch.long, device=octree.device)
    queries = self.local_pos_emb(child_id)

    for idx in range(8):
      query = queries[idx].expand(n_parent, 1, -1)
      if idx == 0:
        seq = torch.cat([context, query], dim=1)
        padding = torch.cat([
          context_padding,
          torch.zeros(n_parent, 1, dtype=torch.bool, device=octree.device),
        ], dim=1)
      else:
        prev = torch.stack(child_outputs, dim=1)
        seq = torch.cat([context, prev, query], dim=1)
        padding = torch.cat([
          context_padding,
          torch.zeros(n_parent, idx + 1, dtype=torch.bool, device=octree.device),
        ], dim=1)
      hidden = self.transformer(seq, padding)
      child_outputs.append(hidden[:, -1])
    child_hidden = torch.stack(child_outputs, dim=1)
    return child_hidden, self._child_indices(octree, parent_depth)

  def forward_split(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    child_hidden, child_indices = self.forward_children(octree, parent_depth, parent_hidden)
    logits = self.split_head(child_hidden)
    return logits, child_hidden, child_indices

  def forward_vq(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    child_hidden, child_indices = self.forward_children(octree, parent_depth, parent_hidden)
    logits = self.vq_head(child_hidden)
    logits = logits.view(logits.shape[0], 8, self.vq_groups, 2)
    return logits, child_hidden, child_indices

  def scatter_child_hidden(
    self,
    child_hidden: torch.Tensor,
    child_indices: torch.Tensor,
    n_child: int,
  ) -> torch.Tensor:
    hidden = torch.zeros(n_child, self.dim, dtype=child_hidden.dtype, device=child_hidden.device)
    valid = child_indices >= 0
    hidden[child_indices[valid].long()] = child_hidden[valid]
    return hidden

  def split_loss(
    self,
    octree,
    parent_depth: int,
    split_targets: torch.Tensor,
    parent_hidden: torch.Tensor,
  ) -> torch.Tensor:
    logits, _, child_indices = self.forward_split(octree, parent_depth, parent_hidden)
    valid = child_indices >= 0
    loss = F.cross_entropy(
      logits.reshape(-1, 2), split_targets.reshape(-1).long(), reduction="none")
    return (loss.view_as(split_targets) * valid.float()).sum() / valid.sum().clamp_min(1)

  def vq_loss(
    self,
    octree,
    parent_depth: int,
    vq_indices: torch.Tensor,
    parent_hidden: torch.Tensor,
  ) -> torch.Tensor:
    logits, _, child_indices = self.forward_vq(octree, parent_depth, parent_hidden)
    valid = child_indices >= 0
    valid_bits = valid.unsqueeze(-1).expand_as(vq_indices)
    return F.cross_entropy(
      logits[valid_bits].reshape(-1, 2), vq_indices[valid_bits].reshape(-1).long())
