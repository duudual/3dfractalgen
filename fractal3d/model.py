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

  def forward(
    self,
    x: torch.Tensor,
    padding_mask: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    x = self.layers(x, mask=attn_mask, src_key_padding_mask=padding_mask)
    return self.norm(x)


class LocalOctreeAttention(nn.Module):
  """Windowed sequence attention used by the VAE token encoder."""

  def __init__(
    self,
    dim: int,
    num_heads: int,
    patch_size: int = 1024,
    dilation: int = 1,
    dropout: float = 0.1,
    use_swin: bool = False,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_heads = num_heads
    self.patch_size = patch_size
    self.dilation = dilation
    self.use_swin = use_swin
    self.qkv = nn.Linear(dim, dim * 3)
    self.proj = nn.Linear(dim, dim)
    self.dropout = dropout

  def _attention_pass(
    self,
    x: torch.Tensor,
    padding: torch.Tensor,
    dilation: int,
  ) -> torch.Tensor:
    bsz, seq_len, dim = x.shape
    block = self.patch_size * dilation
    pad_len = (block - seq_len % block) % block
    if pad_len:
      x = torch.cat([x, x.new_zeros(bsz, pad_len, dim)], dim=1)
      padding = torch.cat([
        padding,
        torch.ones(bsz, pad_len, dtype=torch.bool, device=x.device),
      ], dim=1)
    padded_len = x.shape[1]

    qkv = self.qkv(x).view(bsz, padded_len, 3, self.num_heads, dim // self.num_heads)
    qkv = qkv.permute(2, 0, 1, 3, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if dilation > 1:
      q = q.view(bsz, -1, self.patch_size, dilation, self.num_heads, dim // self.num_heads)
      k = k.view_as(q)
      v = v.view_as(q)
      q = q.transpose(2, 3).reshape(-1, self.patch_size, self.num_heads, dim // self.num_heads)
      k = k.transpose(2, 3).reshape_as(q)
      v = v.transpose(2, 3).reshape_as(q)
      mask = padding.view(bsz, -1, self.patch_size, dilation)
      mask = mask.transpose(2, 3).reshape(-1, self.patch_size)
    else:
      q = q.view(-1, self.patch_size, self.num_heads, dim // self.num_heads)
      k = k.view_as(q)
      v = v.view_as(q)
      mask = padding.view(-1, self.patch_size)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    attn_mask = mask[:, None, None, :]
    out = F.scaled_dot_product_attention(
      q,
      k,
      v,
      attn_mask=attn_mask.logical_not(),
      dropout_p=self.dropout if self.training else 0.0,
    )
    out = out.transpose(1, 2).reshape(-1, self.patch_size, dim)

    if dilation > 1:
      out = out.view(bsz, -1, dilation, self.patch_size, dim)
      out = out.transpose(2, 3).reshape(bsz, padded_len, dim)
    else:
      out = out.view(bsz, padded_len, dim)
    return self.proj(out[:, :seq_len])

  def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    if self.use_swin:
      shift = self.patch_size // 2
      x_shift = torch.roll(x, shifts=-shift, dims=1)
      pad_shift = torch.roll(padding, shifts=-shift, dims=1)
      out = self._attention_pass(x_shift, pad_shift, self.dilation)
      return torch.roll(out, shifts=shift, dims=1)
    return self._attention_pass(x, padding, self.dilation)


class LocalOctreeEncoderBlock(nn.Module):
  def __init__(
    self,
    dim: int,
    num_heads: int,
    patch_size: int,
    dilation: int,
    dropout: float,
    use_swin: bool,
  ) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(dim)
    self.attn = LocalOctreeAttention(
      dim, num_heads, patch_size, dilation, dropout, use_swin)
    self.norm2 = nn.LayerNorm(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, dim * 4),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(dim * 4, dim),
      nn.Dropout(dropout),
    )

  def forward(self, x: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.norm1(x), padding)
    x = x + self.mlp(self.norm2(x))
    return x


class OctreeTokenEncoder(nn.Module):
  """OctFormer-style encoder that maps GT split/VQ tokens to a shape latent."""

  def __init__(
    self,
    dim: int,
    z_dim: int = 128,
    num_layers: int = 6,
    num_heads: int = 6,
    vq_groups: int = 32,
    max_depth: int = 8,
    patch_size: int = 1024,
    dilation: int = 8,
    use_swin: bool = True,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.layers = nn.ModuleList([
      LocalOctreeEncoderBlock(
        dim=dim,
        num_heads=num_heads,
        patch_size=patch_size,
        dilation=1 if i % 2 == 0 else dilation,
        dropout=dropout,
        use_swin=use_swin and ((i // 2) % 2 == 1),
      )
      for i in range(num_layers)
    ])
    self.norm = nn.LayerNorm(dim)
    self.split_emb = nn.Embedding(2, dim)
    self.vq_proj = nn.Linear(vq_groups, dim)
    self.type_emb = nn.Embedding(2, dim)
    self.depth_emb = nn.Embedding(max_depth + 1, dim)
    self.local_pos_emb = LocalOctantPosEmb(dim)
    self.mu_head = nn.Linear(dim, z_dim)
    self.logvar_head = nn.Linear(dim, z_dim)
    self.vq_groups = vq_groups
    self.max_depth = max_depth
    self._init_weights()

  def _init_weights(self) -> None:
    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_x, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_y, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_z, std=0.02)

  def _batch_ids(self, octree, depth: int) -> torch.Tensor:
    return octree.batch_id(depth).long()

  def _depth_tokens(
    self,
    octree,
    depth: int,
    token_data: torch.Tensor,
    token_type: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    device = token_data.device
    nnum = token_data.shape[0]
    child_id = torch.arange(nnum, device=device) % 8
    depth_id = torch.full((nnum,), depth, dtype=torch.long, device=device)
    depth_id = depth_id.clamp(max=self.max_depth)

    if token_type == 0:
      token = self.split_emb(token_data.long())
    else:
      token = self.vq_proj(token_data.float())
    token = token + self.type_emb.weight[token_type]
    token = token + self.depth_emb(depth_id)
    token = token + self.local_pos_emb(child_id)
    return token, self._batch_ids(octree, depth)

  def forward(
    self,
    octree,
    split_by_depth: dict[int, torch.Tensor],
    vq_indices: torch.Tensor,
    depth_stop: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    tokens: list[torch.Tensor] = []
    batch_ids: list[torch.Tensor] = []
    for depth in sorted(split_by_depth):
      token, batch_id = self._depth_tokens(
        octree, depth, split_by_depth[depth].to(octree.device), token_type=0)
      tokens.append(token)
      batch_ids.append(batch_id)

    token, batch_id = self._depth_tokens(
      octree, depth_stop, vq_indices.to(octree.device), token_type=1)
    tokens.append(token)
    batch_ids.append(batch_id)

    flat = torch.cat(tokens, dim=0)
    flat_batch = torch.cat(batch_ids, dim=0)
    batch_size = int(octree.batch_size)
    max_len = max(int((flat_batch == i).sum().item()) for i in range(batch_size))
    seq = flat.new_zeros(batch_size, max_len, flat.shape[-1])
    padding = torch.ones(batch_size, max_len, dtype=torch.bool, device=flat.device)

    for batch_idx in range(batch_size):
      mask = flat_batch == batch_idx
      count = int(mask.sum().item())
      if count == 0:
        continue
      seq[batch_idx, :count] = flat[mask]
      padding[batch_idx, :count] = False

    encoded = seq
    for layer in self.layers:
      encoded = layer(encoded, padding)
    encoded = self.norm(encoded)
    valid = (~padding).to(encoded.dtype).unsqueeze(-1)
    pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return self.mu_head(pooled), self.logvar_head(pooled)


class Fractal3DGenerator(nn.Module):
  """Parent/sibling hidden-state AR generator for octree split and VQ tokens."""

  def __init__(
    self,
    dim: int = 192,
    num_layers: int = 6,
    num_heads: int = 6,
    num_vq_embed: int = 256,
    vq_groups: Optional[int] = None,
    z_dim: int = 128,
    full_depth: int = 3,
    max_depth: int = 8,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.dim = dim
    self.num_vq_embed = num_vq_embed
    self.vq_groups = vq_groups or num_vq_embed
    self.z_dim = z_dim
    self.full_depth = full_depth
    self.max_depth = max_depth

    self.z_proj = nn.Linear(z_dim, dim)
    self.z_cond_proj = nn.Linear(z_dim, dim)
    self.role_emb = nn.Embedding(2, dim)
    self.local_pos_emb = LocalOctantPosEmb(dim)
    self.path_pos_emb = nn.Parameter(torch.zeros(max_depth, 3, dim))
    self.transformer = ChildARTransformer(
      dim=dim, num_layers=num_layers, num_heads=num_heads, dropout=dropout)
    self.split_head = nn.Linear(dim, 2)
    self.vq_head = nn.Linear(dim, self.vq_groups * 2)
    self.register_buffer(
      "parallel_child_attn_mask",
      self._build_parallel_child_attn_mask(),
      persistent=False,
    )
    self._init_weights()

  @staticmethod
  def _build_parallel_child_attn_mask() -> torch.Tensor:
    seq_len = 16
    child_offset = 8
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    mask[:child_offset, child_offset:] = True
    for query in range(child_offset, seq_len):
      child_query = query - child_offset
      for key in range(child_offset, seq_len):
        child_key = key - child_offset
        if child_key > child_query:
          mask[query, key] = True
    return mask

  def _init_weights(self) -> None:
    nn.init.normal_(self.local_pos_emb.emb_x, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_y, std=0.02)
    nn.init.normal_(self.local_pos_emb.emb_z, std=0.02)
    nn.init.normal_(self.path_pos_emb, std=0.02)
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
    idx = torch.arange(nnum, dtype=torch.long, device=device)
    starts = (idx // 8) * 8
    offsets = torch.arange(8, dtype=torch.long, device=device)
    group = starts[:, None] + offsets[None, :]
    group = torch.where(group < nnum, group, torch.full_like(group, -1))
    return group

  def _child_indices(self, octree, parent_depth: int) -> torch.Tensor:
    n_parent = int(octree.nnum[parent_depth])
    n_child = int(octree.nnum[parent_depth + 1])
    device = octree.device
    children = octree.children[parent_depth].long()
    offsets = torch.arange(8, dtype=torch.long, device=device)
    out = children[:, None] + offsets[None, :]
    valid = (children[:, None] >= 0) & (out < n_child)
    out = torch.where(valid, out, torch.full((n_parent, 8), -1, dtype=torch.long, device=device))
    return out

  def _node_path_pos_emb(self, octree, depth: int) -> torch.Tensor:
    nnum = int(octree.nnum[depth])
    device = octree.device
    if depth <= 0:
      return torch.zeros(nnum, self.dim, dtype=self.path_pos_emb.dtype, device=device)

    if hasattr(octree, "keys"):
      key = octree.keys[depth].long()
      morton = key & ((1 << (3 * depth)) - 1)
      pos = torch.zeros(nnum, self.dim, dtype=self.path_pos_emb.dtype, device=device)
      for level in range(min(depth, self.max_depth)):
        child_id = (morton >> (3 * level)) & 7
        x_bit = child_id & 1
        y_bit = (child_id >> 1) & 1
        z_bit = (child_id >> 2) & 1
        sx = (2 * x_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
        sy = (2 * y_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
        sz = (2 * z_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
        pos = (
          pos
          + sx * self.path_pos_emb[level, 0]
          + sy * self.path_pos_emb[level, 1]
          + sz * self.path_pos_emb[level, 2]
        )
      return pos

    idx = torch.arange(nnum, dtype=torch.long, device=device)
    pos = torch.zeros(nnum, self.dim, dtype=self.path_pos_emb.dtype, device=device)
    for level in range(min(depth, self.max_depth)):
      child_id = (idx >> (3 * level)) & 7
      x_bit = child_id & 1
      y_bit = (child_id >> 1) & 1
      z_bit = (child_id >> 2) & 1
      sx = (2 * x_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
      sy = (2 * y_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
      sz = (2 * z_bit - 1).to(dtype=pos.dtype).unsqueeze(-1)
      pos = (
        pos
        + sx * self.path_pos_emb[level, 0]
        + sy * self.path_pos_emb[level, 1]
        + sz * self.path_pos_emb[level, 2]
      )
    return pos

  def _child_path_pos_emb(self, octree, parent_depth: int) -> torch.Tensor:
    child_indices = self._child_indices(octree, parent_depth)
    child_pos = self._node_path_pos_emb(octree, parent_depth + 1)
    safe = child_indices.clamp(min=0)
    out = child_pos[safe]
    return torch.where((child_indices >= 0).unsqueeze(-1), out, torch.zeros_like(out))

  def _z_condition_for_depth(self, octree, z: torch.Tensor, depth: int) -> torch.Tensor:
    batch_ids = octree.batch_id(depth).long()
    return self.z_cond_proj(z)[batch_ids]

  def _root_children_from_z(self, octree, z: torch.Tensor, parallel: bool = False) -> torch.Tensor:
    batch_size = int(octree.batch_size)
    if z.shape[0] != batch_size:
      raise ValueError(f"z has batch {z.shape[0]}, expected {batch_size}.")
    root = self.z_proj(z)
    z_cond = self.z_cond_proj(z)
    child_id = torch.arange(8, dtype=torch.long, device=octree.device)
    queries = self.local_pos_emb(child_id)
    child_path_pos = self._node_path_pos_emb(octree, 1).view(batch_size, 8, self.dim)

    if parallel:
      child_queries = (
        queries.unsqueeze(0).expand(batch_size, -1, -1)
        + child_path_pos
        + z_cond.unsqueeze(1)
      )
      seq = torch.cat([root.unsqueeze(1), child_queries], dim=1)
      padding = torch.zeros(batch_size, 9, dtype=torch.bool, device=octree.device)
      attn_mask = torch.zeros(9, 9, dtype=torch.bool, device=octree.device)
      attn_mask[0, 1:] = True
      for query in range(1, 9):
        for key in range(1, 9):
          if key > query:
            attn_mask[query, key] = True
      hidden = self.transformer(seq, padding, attn_mask=attn_mask)
      return hidden[:, 1:9].reshape(batch_size * 8, self.dim)

    child_outputs: list[torch.Tensor] = []
    for idx in range(8):
      query = (
        queries[idx].expand(batch_size, 1, -1)
        + child_path_pos[:, idx:idx + 1]
        + z_cond.unsqueeze(1)
      )
      if idx == 0:
        seq = torch.cat([root.unsqueeze(1), query], dim=1)
      else:
        prev = torch.stack(child_outputs, dim=1)
        seq = torch.cat([root.unsqueeze(1), prev, query], dim=1)
      padding = torch.zeros(seq.shape[:2], dtype=torch.bool, device=octree.device)
      hidden = self.transformer(seq, padding)
      child_outputs.append(hidden[:, -1])
    return torch.stack(child_outputs, dim=1).reshape(batch_size * 8, self.dim)

  def bootstrap_hidden_from_z(
    self,
    octree,
    z: torch.Tensor,
    target_depth: int,
    parallel: bool = False,
  ) -> torch.Tensor:
    if target_depth < 1:
      raise ValueError("target_depth must be >= 1 for z bootstrap.")
    hidden = self._root_children_from_z(octree, z, parallel=parallel)
    if int(octree.nnum[1]) != hidden.shape[0]:
      raise ValueError(
        f"Octree depth 1 has {int(octree.nnum[1])} nodes, "
        f"but z bootstrap produced {hidden.shape[0]}.")
    for parent_depth in range(1, target_depth):
      _, child_hidden, child_indices = self.forward_split(
        octree, parent_depth, hidden, parallel=parallel, z=z)
      hidden = self.scatter_child_hidden(
        child_hidden, child_indices, int(octree.nnum[parent_depth + 1]))
    return hidden

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

    parent_idx = torch.arange(n_parent, dtype=torch.long, device=octree.device)
    sibling_idx = self._sibling_indices(octree, parent_depth)
    local_id = parent_idx % 8
    keep_uncle = torch.arange(8, dtype=torch.long, device=octree.device)[None, :] != local_id[:, None]
    uncle_idx = sibling_idx[keep_uncle].view(n_parent, 7)

    valid_uncle = uncle_idx >= 0
    safe_uncle = uncle_idx.clamp(min=0)
    path_pos = self._node_path_pos_emb(octree, parent_depth)
    context = torch.zeros(n_parent, 8, self.dim, device=octree.device)
    context[:, 0] = (
      parent_hidden[parent_idx]
      + path_pos[parent_idx]
      + self.role_emb.weight[ROLE_PARENT]
    )
    context[:, 1:8] = (
      parent_hidden[safe_uncle]
      + path_pos[safe_uncle]
      + self.role_emb.weight[ROLE_UNCLE]
    )
    context[:, 1:8] = torch.where(valid_uncle.unsqueeze(-1), context[:, 1:8], 0)

    padding_mask = torch.zeros(n_parent, 8, dtype=torch.bool, device=octree.device)
    padding_mask[:, 1:8] = ~valid_uncle
    return context, padding_mask

  def forward_children(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
    parallel: bool = False,
    z: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    if parallel:
      return self.forward_children_parallel(octree, parent_depth, parent_hidden, z=z)

    context, context_padding = self._context_tokens(octree, parent_depth, parent_hidden)
    n_parent = context.shape[0]
    child_outputs: list[torch.Tensor] = []
    child_id = torch.arange(8, dtype=torch.long, device=octree.device)
    queries = self.local_pos_emb(child_id)
    child_path_pos = self._child_path_pos_emb(octree, parent_depth)
    z_cond = None
    if z is not None:
      z_cond = self._z_condition_for_depth(octree, z, parent_depth).unsqueeze(1)

    for idx in range(8):
      query = queries[idx].expand(n_parent, 1, -1) + child_path_pos[:, idx:idx + 1]
      if z_cond is not None:
        query = query + z_cond
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

  def forward_children_parallel(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
    z: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    context, context_padding = self._context_tokens(octree, parent_depth, parent_hidden)
    n_parent = context.shape[0]
    child_id = torch.arange(8, dtype=torch.long, device=octree.device)
    queries = self.local_pos_emb(child_id).unsqueeze(0).expand(n_parent, -1, -1)
    queries = queries + self._child_path_pos_emb(octree, parent_depth)
    if z is not None:
      queries = queries + self._z_condition_for_depth(octree, z, parent_depth).unsqueeze(1)
    seq = torch.cat([context, queries], dim=1)
    padding = torch.cat([
      context_padding,
      torch.zeros(n_parent, 8, dtype=torch.bool, device=octree.device),
    ], dim=1)
    attn_mask = self.parallel_child_attn_mask.to(device=octree.device)
    hidden = self.transformer(seq, padding, attn_mask=attn_mask)
    return hidden[:, 8:16], self._child_indices(octree, parent_depth)

  def forward_split(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
    parallel: bool = False,
    z: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    child_hidden, child_indices = self.forward_children(
      octree, parent_depth, parent_hidden, parallel=parallel, z=z)
    logits = self.split_head(child_hidden)
    return logits, child_hidden, child_indices

  def forward_vq(
    self,
    octree,
    parent_depth: int,
    parent_hidden: torch.Tensor,
    parallel: bool = False,
    z: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    child_hidden, child_indices = self.forward_children(
      octree, parent_depth, parent_hidden, parallel=parallel, z=z)
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


class Fractal3DVAE(nn.Module):
  """VAE wrapper around an OctFormer-style encoder and Fractal3D decoder."""

  def __init__(
    self,
    dim: int = 192,
    z_dim: int = 128,
    encoder_layers: int = 6,
    decoder_layers: int = 6,
    num_heads: int = 6,
    num_vq_embed: int = 32,
    vq_groups: int = 32,
    full_depth: int = 3,
    max_depth: int = 8,
    encoder_patch_size: int = 1024,
    encoder_dilation: int = 8,
    dropout: float = 0.1,
  ) -> None:
    super().__init__()
    self.encoder = OctreeTokenEncoder(
      dim=dim,
      z_dim=z_dim,
      num_layers=encoder_layers,
      num_heads=num_heads,
      vq_groups=vq_groups,
      max_depth=max_depth,
      patch_size=encoder_patch_size,
      dilation=encoder_dilation,
      dropout=dropout,
    )
    self.decoder = Fractal3DGenerator(
      dim=dim,
      num_layers=decoder_layers,
      num_heads=num_heads,
      num_vq_embed=num_vq_embed,
      vq_groups=vq_groups,
      z_dim=z_dim,
      full_depth=full_depth,
      max_depth=max_depth,
      dropout=dropout,
    )

  @staticmethod
  def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, -30.0, 20.0)
    std = torch.exp(0.5 * logvar)
    return mu + std * torch.randn_like(std)

  @staticmethod
  def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, -30.0, 20.0)
    return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()

  def encode(
    self,
    octree,
    split_by_depth: dict[int, torch.Tensor],
    vq_indices: torch.Tensor,
    depth_stop: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    return self.encoder(octree, split_by_depth, vq_indices, depth_stop)
