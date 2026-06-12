from __future__ import annotations

from pathlib import Path
from typing import List

import ocnn
import ognn
import torch
import torch.nn.functional as F
from ognn import mpu
from ognn.octreed import OctreeD
from ocnn.octree import Octree


class Encoder(torch.nn.Module):
  def __init__(
    self,
    in_channels: int,
    channels: List[int],
    resblk_nums: List[int],
    bottleneck: int = 1,
  ) -> None:
    super().__init__()
    groups = 32
    self.stage_num = len(channels)
    self.delta_depth = self.stage_num - 1
    self.conv1 = ocnn.modules.OctreeConvGnRelu(in_channels, channels[0], groups)
    self.blocks = torch.nn.ModuleList([
      ocnn.modules.OctreeResBlocks(
        channels[i], channels[i], resblk_nums[i], bottleneck,
        nempty=False, resblk=ocnn.modules.OctreeResBlockGn,
        use_checkpoint=True)
      for i in range(self.stage_num)
    ])
    self.downsample = torch.nn.ModuleList([
      ocnn.modules.OctreeConvGnRelu(
        channels[i], channels[i + 1], groups, kernel_size=[2], stride=2)
      for i in range(self.stage_num - 1)
    ])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int) -> torch.Tensor:
    out = self.conv1(data, octree, depth)
    for i in range(self.stage_num):
      di = depth - i
      out = self.blocks[i](out, octree, di)
      if i < self.stage_num - 1:
        out = self.downsample[i](out, octree, di)
    return out


class Decoder(torch.nn.Module):
  def __init__(
    self,
    n_node_type: int,
    encoder_channels: List[int],
    encoder_blk_nums: List[int],
    decoder_channels: List[int],
    decoder_blk_nums: List[int],
    mpu_stage_nums: int = 3,
    pred_stage_nums: int = 3,
    bottleneck: int = 1,
  ) -> None:
    super().__init__()
    self.n_edge_type = 7
    self.head_channel = 64
    self.use_checkpoint = True
    self.act_type = "relu"
    self.resblk_type = "basic"
    self.norm_type = "group_norm"
    self.n_node_type = n_node_type
    self.encoder_blk_nums = encoder_blk_nums
    self.decoder_blk_nums = decoder_blk_nums
    self.encoder_channels = encoder_channels
    self.decoder_channels = decoder_channels

    self.encoder_stages = len(self.encoder_blk_nums)
    self.decoder_stages = len(self.decoder_blk_nums)
    self.graph_pad = ognn.nn.GraphPad()

    n_node_types = [self.n_node_type - i for i in range(self.encoder_stages)]
    self.encoder = torch.nn.ModuleList([
      ognn.nn.GraphResBlocks(
        self.encoder_channels[i], self.encoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type, self.act_type,
        bottleneck, self.encoder_blk_nums[i], self.resblk_type)
      for i in range(self.encoder_stages)
    ])
    self.downsample = torch.nn.ModuleList([
      ognn.nn.GraphDownsample(
        self.encoder_channels[i], self.encoder_channels[i + 1],
        self.norm_type, self.act_type)
      for i in range(self.encoder_stages - 1)
    ])

    n_node_type = self.n_node_type - self.encoder_stages + 1
    n_node_types = [n_node_type + i for i in range(self.decoder_stages)]
    self.upsample = torch.nn.ModuleList([
      ognn.nn.GraphUpsample(
        self.decoder_channels[i - 1], self.decoder_channels[i],
        self.norm_type, self.act_type)
      for i in range(1, self.decoder_stages)
    ])
    self.decoder = torch.nn.ModuleList([
      ognn.nn.GraphResBlocks(
        self.decoder_channels[i], self.decoder_channels[i],
        self.n_edge_type, n_node_types[i], self.norm_type, self.act_type,
        bottleneck, self.decoder_blk_nums[i], self.resblk_type)
      for i in range(self.decoder_stages)
    ])

    self.start_pred = self.decoder_stages - pred_stage_nums
    self.predict = torch.nn.ModuleList([
      ognn.nn.Prediction(
        self.decoder_channels[i], self.head_channel, 2,
        self.norm_type, self.act_type)
      for i in range(self.start_pred, self.decoder_stages)
    ])
    self.start_mpu = self.decoder_stages - mpu_stage_nums
    self.regress = torch.nn.ModuleList([
      ognn.nn.Prediction(
        self.decoder_channels[i], self.head_channel, 4,
        self.norm_type, self.act_type)
      for i in range(self.start_mpu, self.decoder_stages)
    ])

  def _octree_align(
    self,
    value: torch.Tensor,
    octree: OctreeD,
    octree_query: OctreeD,
    depth: int,
  ) -> torch.Tensor:
    key = octree.graphs[depth].key
    query = octree_query.graphs[depth].key
    return ocnn.nn.search_value(value, key, query)

  def octree_encoder(self, code: torch.Tensor, octree: OctreeD, depth: int) -> dict:
    convs = {depth: code}
    for i in range(self.encoder_stages):
      d = depth - i
      convs[d] = self.encoder[i](convs[d], octree, d)
      if i < self.encoder_stages - 1:
        convs[d - 1] = self.downsample[i](convs[d], octree, d)
    return convs

  def octree_decoder(
    self,
    convs: dict,
    octree_in: OctreeD,
    octree_out: OctreeD,
    depth: int,
    update_octree: bool = False,
  ) -> dict:
    logits, signals = {}, {}
    deconv = convs[depth]
    for i in range(self.decoder_stages):
      d = depth + i
      if i > 0:
        deconv = self.upsample[i - 1](deconv, octree_out, d - 1)
        if d in convs:
          if i >= self.start_pred:
            skip = self._octree_align(convs[d], octree_in, octree_out, d)
          else:
            skip = convs[d]
          deconv = deconv + skip
      deconv = self.decoder[i](deconv, octree_out, d)

      if i >= self.start_pred:
        j = i - self.start_pred
        logit = self.predict[j](deconv, octree_out, d)
        nnum = octree_out.nnum[d]
        logits[d] = logit[-nnum:]

      if i >= self.start_mpu:
        j = i - self.start_mpu
        signal = self.regress[j](deconv, octree_out, d)
        signals[d] = self.graph_pad(signal, octree_out, d)

      if update_octree and i >= self.start_pred:
        split = logits[d].argmax(1).int()
        octree_out.octree_split(split, d)
        if i < self.decoder_stages - 1:
          octree_out.octree_grow(d + 1)

    return {"logits": logits, "signals": signals, "octree_out": octree_out}

  def forward(
    self,
    code: torch.Tensor,
    depth: int,
    octree_in: OctreeD,
    octree_out: OctreeD,
    pos: torch.Tensor | None = None,
    update_octree: bool = False,
  ) -> dict:
    convs = self.octree_encoder(code, octree_in, depth)
    d = depth - self.encoder_stages + 1
    output = self.octree_decoder(convs, octree_in, octree_out, d, update_octree)
    depth_out = octree_out.depth
    neural_mpu = mpu.NeuralMPU(output["signals"], octree_out, depth_out)
    if pos is not None:
      output["mpus"] = neural_mpu(pos)
    output["neural_mpu"] = lambda p: neural_mpu(p)[depth_out]
    return output


class BinarySphericalQuantizer(torch.nn.Module):
  def __init__(
    self,
    D: int,
    gamma0: float = 1.0,
    gamma1: float = 1.0,
    inv_temperature: float = 1.0,
    rnd_flip: float = 0.0,
    **kwargs,
  ) -> None:
    super().__init__()
    self.embed_dim = D
    self.gamma0 = gamma0
    self.gamma1 = gamma1
    self.rnd_flip = rnd_flip
    self.inv_temperature = inv_temperature
    self.register_buffer("basis", 2 ** torch.arange(D - 1, -1, -1))

  def quantize(self, z: torch.Tensor) -> torch.Tensor:
    zhat = (z > 0) * 2 - 1
    if self.training and self.rnd_flip > 0:
      ratio = torch.rand(1).item() * self.rnd_flip
      flip = (torch.rand_like(z) > ratio) * 2 - 1
      zhat = zhat * flip
    return z + (zhat - z).detach()

  def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = F.normalize(z, p=2.0, dim=-1)
    persample_entropy, cb_entropy = self.soft_entropy_loss(z)
    entropy_penalty = self.gamma0 * persample_entropy - self.gamma1 * cb_entropy
    zq = self.quantize(z)
    indices = self.code2index(zq.detach())
    zq = zq * (1.0 / self.embed_dim ** 0.5)
    return zq, indices, entropy_penalty / self.inv_temperature

  def soft_entropy_loss(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5 * self.inv_temperature))
    prob = torch.stack([p, 1 - p], dim=-1)
    per_sample_entropy = self.get_entropy(prob, dim=-1).sum(dim=-1).mean()
    avg_prob = torch.mean(prob, dim=0)
    codebook_entropy = self.get_entropy(avg_prob, dim=-1).sum()
    return per_sample_entropy, codebook_entropy

  def get_entropy(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-8)).sum(dim=dim)

  def code2index(self, zhat: torch.Tensor) -> torch.Tensor:
    return ((zhat + 1) / 2).long()

  def index2code(self, indices: torch.Tensor) -> torch.Tensor:
    return indices * 2.0 - 1.0

  def extract_code(self, indices: torch.Tensor) -> torch.Tensor:
    return self.index2code(indices) * (1.0 / self.embed_dim ** 0.5)


class VQVAELargeBSQ32(torch.nn.Module):
  """Standalone OctGPT-compatible ShapeNet VQVAE large bsq32."""

  def __init__(self) -> None:
    super().__init__()
    self.feature = "ND"
    self.encoder = Encoder(
      in_channels=4,
      channels=[32, 32, 64],
      resblk_nums=[2, 2, 2],
      bottleneck=1,
    )
    self.decoder = Decoder(
      n_node_type=7,
      encoder_channels=[64, 128, 256, 512],
      encoder_blk_nums=[2, 4, 8, 2],
      decoder_channels=[512, 256, 128, 64, 32, 32],
      decoder_blk_nums=[2, 4, 8, 2, 2, 2],
      mpu_stage_nums=3,
      pred_stage_nums=3,
      bottleneck=1,
    )
    self.quantizer = BinarySphericalQuantizer(D=32)
    self.pre_proj = torch.nn.Linear(64, 32, bias=True)
    self.post_proj = torch.nn.Linear(32, 64, bias=True)

  def extract_code(self, octree_in: Octree) -> torch.Tensor:
    depth = octree_in.depth
    data = octree_in.get_input_feature(feature=self.feature)
    conv = self.encoder(data, octree_in, depth)
    return self.pre_proj(conv)

  def decode_code(
    self,
    code: torch.Tensor,
    code_depth: int,
    octree_in: OctreeD,
    octree_out: OctreeD,
    pos: torch.Tensor | None = None,
    update_octree: bool = False,
  ) -> dict:
    data = self.post_proj(code)
    data = octree_in.pad_zeros(data, code_depth)
    return self.decoder(data, code_depth, octree_in, octree_out, pos, update_octree)

  def forward(
    self,
    octree_in: Octree,
    octree_out: OctreeD,
    pos: torch.Tensor | None = None,
    update_octree: bool = False,
  ) -> dict:
    code = self.extract_code(octree_in)
    zq, _, vq_loss = self.quantizer(code)
    doctree_in = OctreeD(octree_in)
    code_depth = doctree_in.depth - self.encoder.delta_depth
    output = self.decode_code(zq, code_depth, doctree_in, octree_out, pos, update_octree)
    output["vae_loss"] = vq_loss
    return output


def load_octgpt_vqvae(
  ckpt_path: str | Path,
  device: torch.device | str,
) -> VQVAELargeBSQ32:
  ckpt_path = Path(ckpt_path).resolve()
  vqvae = VQVAELargeBSQ32()
  checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
  vqvae.load_state_dict(checkpoint)
  vqvae.to(device)
  vqvae.eval()
  for param in vqvae.parameters():
    param.requires_grad_(False)
  return vqvae


@torch.no_grad()
def encode_bsq_tokens(
  vqvae: VQVAELargeBSQ32,
  octree,
) -> tuple[torch.Tensor, torch.Tensor, int]:
  code = vqvae.extract_code(octree)
  zq, indices, _ = vqvae.quantizer(code)
  code_depth = octree.depth - vqvae.encoder.delta_depth
  return indices.long(), zq, int(code_depth)
