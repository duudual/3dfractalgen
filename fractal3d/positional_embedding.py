from __future__ import annotations

import numpy as np
import torch


FULL_DEPTH = 3
MAX_DEPTH = 6


class AbsPosEmb(torch.nn.Module):
  """OctGPT-style absolute 3D position embedding.

  This is copied into 3dfractalgen so the model does not depend on importing
  modules from the sibling octgpt project.
  """

  def __init__(
    self,
    num_embed: int,
    full_depth: int = FULL_DEPTH,
    max_depth: int = MAX_DEPTH,
  ):
    super().__init__()
    self.num_embed = num_embed
    self.full_depth = full_depth
    self.max_depth = max_depth
    self.absolute_emb = torch.nn.Parameter(self.init_absolute_emb())
    self.depth_emb = torch.nn.Embedding(
      self.max_depth - self.full_depth + 1, num_embed)

  def get_emb(self, sin_inp):
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

  def init_absolute_emb(self):
    xyz = torch.arange(0, 2 ** (self.max_depth + 1)).repeat(3, 1).t()
    pos_x, pos_y, pos_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    channels = int(np.ceil(self.num_embed / 6) * 2)
    if channels % 2:
      channels += 1

    inv_freq = 1.0 / (100 ** (torch.arange(0, channels, 2).float() / channels))

    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
    sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
    emb_x = self.get_emb(sin_inp_x)
    emb_y = self.get_emb(sin_inp_y)
    emb_z = self.get_emb(sin_inp_z)
    emb = torch.zeros((pos_x.shape[0], channels * 3))
    column_index = torch.arange(0, channels * 3, 3)
    emb[:, column_index] = emb_x
    emb[:, column_index + 1] = emb_y
    emb[:, column_index + 2] = emb_z

    return emb

  def get_3d_pos_emb(self, xyz):
    pos_x, pos_y, pos_z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    device = xyz.device

    channels = int(np.ceil(self.num_embed / 6) * 2)
    if channels % 2:
      channels += 1

    column_index = torch.arange(0, channels * 3, 3, device=device)
    index_x = torch.meshgrid(pos_x.long(), column_index, indexing="ij")
    index_y = torch.meshgrid(pos_y.long(), column_index + 1, indexing="ij")
    index_z = torch.meshgrid(pos_z.long(), column_index + 2, indexing="ij")
    emb = torch.zeros((pos_x.shape[0], channels * 3), device=device)
    emb[:, column_index] = self.absolute_emb[index_x]
    emb[:, column_index + 1] = self.absolute_emb[index_y]
    emb[:, column_index + 2] = self.absolute_emb[index_z]
    return emb[:, :self.num_embed]

  def forward(self, data: torch.Tensor, octree):
    depth_embedding = self.depth_emb(octree.depth_idx)
    position_embeddings = self.get_3d_pos_emb(octree.xyz)
    position_embeddings += depth_embedding
    return position_embeddings
