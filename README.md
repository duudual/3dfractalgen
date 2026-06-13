# 3DFractalGen

3DFractalGen explores a VAE-style, top-down fractal octree generator for 3D
shape generation. The current implementation uses:

- an OctFormer-style local-window encoder over GT octree split tokens and
  OctGPT VQ/BSQ tokens;
- a shape-level latent variable `z`;
- a parent/sibling hidden-state autoregressive decoder;
- the frozen OctGPT VQ-VAE decoder to convert generated VQ codes into an
  implicit SDF/MPU field.

The decoder starts from:

```text
root hidden = z_proj(z)
```

Then it generates the 8 root-child hidden states and recursively expands hidden
states down the octree. There is no `seed_token` or `initial_hidden` path in the
current VAE workflow.

## Data Preparation

The raw ShapeNet category directory is expected to contain samples like:

```text
data/02691156/<uid>/models/model_normalized.obj
```

Convert OBJ meshes to OctGPT-compatible point clouds:

```bash
cd 3dfractalgen
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --points 200000
```

This writes `pointcloud.npz` into each shape directory:

- `points`: sampled surface points, shape `[N, 3]`, `float16`
- `normals`: corresponding face normals, shape `[N, 3]`, `float16`

Verify OCNN octree construction:

```bash
python scripts/inspect_octree.py --data data/02691156 --depth 8 --full-depth 3
```

For a quick smoke test:

```bash
python scripts/prepare_shapenet_pointclouds.py \
  --data data/02691156 \
  --points 4096 \
  --limit 2 \
  --output-root outputs/airplane_pointcloud_smoke \
  --overwrite

python scripts/inspect_octree.py \
  --data outputs/airplane_pointcloud_smoke \
  --depth 8 \
  --full-depth 3
```

## VQ Token Cache

VAE training needs GT VQ/BSQ tokens from the frozen OctGPT VQ-VAE. Caching them
is recommended.

Expected checkpoint:

```text
ckpt/vqvae_large_im5_uncond_bsq32.pth
```

Cache tokens:

```bash
python scripts/cache_vq_tokens.py \
  --data data/02691156 \
  --output-dir outputs/vq_cache \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --depth 8 \
  --full-depth 3 \
  --depth-stop 6 \
  --num-vq-embed 32 \
  --vq-groups 32
```

Each cached file stores:

- `indices`: BSQ bit indices, shape `[num_code_nodes, 32]`
- `codes`: quantized VQ code vectors
- `code_depth`: expected to equal `--depth-stop`
- metadata for depth, full depth, data scale, and VQ dimensions

## Train VAE

Run joint VAE training:

```bash
python scripts/train_vae.py \
  --data data/02691156 \
  --vq-cache-dir outputs/vq_cache \
  --output-dir outputs/vae_train \
  --depth 8 \
  --full-depth 3 \
  --depth-stop 6 \
  --dim 192 \
  --z-dim 128 \
  --encoder-layers 6 \
  --decoder-layers 6 \
  --encoder-patch-size 1024 \
  --encoder-dilation 8 \
  --heads 6 \
  --vq-groups 32 \
  --num-vq-embed 32 \
  --lambda-vq 1.0 \
  --beta-max 1e-4 \
  --beta-warmup-epochs 10 \
  --batch-size 1 \
  --epochs 100
```

Without a cache, `train_vae.py` can encode VQ tokens online with the frozen
VQ-VAE:

```bash
python scripts/train_vae.py \
  --data data/02691156 \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --output-dir outputs/vae_train_online \
  --depth 8 \
  --full-depth 3 \
  --depth-stop 6
```

The training loss is:

```text
total = split_loss + lambda_vq * vq_loss + beta * KL(q(z|x) || N(0,I))
```

`beta` linearly warms up to `--beta-max` over `--beta-warmup-epochs`.

TensorBoard logs are written to:

```text
outputs/vae_train/tensorboard
```

Checkpoints:

```text
outputs/vae_train/last.pt
outputs/vae_train/best.pt
```

## Small-Data Overfit Debug

```bash
python scripts/make_debug_filelist.py \
  --data data/02691156 \
  --output outputs/debug_filelist.txt \
  --count 4
```

Cache VQ tokens for only those examples:

```bash
python scripts/cache_vq_tokens.py \
  --data data/02691156 \
  --filelist outputs/debug_filelist.txt \
  --output-dir outputs/vq_cache_debug \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --depth 8 \
  --full-depth 3 \
  --depth-stop 6 \
  --overwrite
```

Run beta-free reconstruction training:

```bash
python scripts/train_vae.py \
  --data data/02691156 \
  --filelist outputs/debug_filelist.txt \
  --val-fraction 0 \
  --vq-cache-dir outputs/vq_cache_debug \
  --output-dir outputs/vae_overfit_debug \
  --depth 8 \
  --full-depth 3 \
  --depth-stop 6 \
  --beta-max 0 \
  --batch-size 1 \
  --epochs 200
```

Expected signs of a healthy implementation on 1-4 shapes:

- `split_loss` should keep dropping and `split_accuracy` should approach 1.
- `vq_loss` should drop below the random baseline near `0.69`.
- `vq_bit_accuracy` should clearly exceed `0.55`.
- `vq_node_exact` should become nonzero and increase.
- With `--beta-max 0`, `kl_loss` can be large; that is acceptable during
  reconstruction-only debugging.

If this tiny run cannot overfit, inspect implementation alignment before trying
larger training.

For an even smaller split-first debug target, stop at `depth_stop=4`. The frozen
OctGPT VQ-VAE used here has `code_depth = input_depth - 2`, so the matching
cache command must use `--depth 6`, not `--depth 8`:

```bash
python scripts/cache_vq_tokens.py \
  --data data/02691156 \
  --filelist outputs/debug_filelist.txt \
  --output-dir outputs/vq_cache_debug_d4 \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --depth 6 \
  --full-depth 3 \
  --depth-stop 4 \
  --overwrite
```

Then run a deterministic, beta-free split-focused overfit:

```bash
python scripts/train_vae.py \
  --data data/02691156 \
  --filelist outputs/debug_filelist.txt \
  --val-fraction 0 \
  --vq-cache-dir outputs/vq_cache_debug_d4 \
  --output-dir outputs/vae_overfit_split_d4 \
  --depth 6 \
  --full-depth 3 \
  --depth-stop 4 \
  --beta-max 0 \
  --lambda-vq 0 \
  --split-pos-weight 2.0 \
  --deterministic-z \
  --batch-size 1 \
  --epochs 100
```

## Diagnostic Logs

`train_vae.py` logs the following metrics per epoch and to TensorBoard:

- losses: `loss`, `split_loss`, `vq_loss`, `kl_loss`, `kl_per_dim`
- split: `split_accuracy`, `split_target_rate`, `split_pred_rate`,
  `split_precision`, `split_recall`
- VQ bits: `vq_bit_accuracy`, `vq_node_exact`, `vq_target_one_rate`,
  `vq_pred_one_rate`, `vq_one_precision`, `vq_one_recall`
- latent: `z_mu_abs_mean`, `z_std_mean`, `beta`

Useful interpretations:

- `split_pred_rate` close to 0 or 1 while `split_recall`/`precision` is poor
  means the split head is mostly predicting a majority class.
- `vq_loss` around `0.69` and `vq_bit_accuracy` around `0.5` means VQ bits are
  still near random guessing.
- `kl_per_dim` quickly approaching 0 with `z_std_mean` near 1 can indicate
  posterior collapse.
- For the overfit run with `--beta-max 0`, prioritize reconstruction metrics
  over KL.

## Possible Improvements To Try

- Reconstruction pretraining: train with `--beta-max 0`, then resume with a
  small KL such as `--beta-max 1e-5`.
- KL schedule: increase `--beta-warmup-epochs` or keep `beta` at 0 until
  split/VQ overfit works.
- VQ weighting: increase `--lambda-vq` if split learns but VQ stays near random.
- Smaller debug model: reduce `--dim`, `--encoder-layers`, or `--decoder-layers`
  for faster alignment tests.
- Easier target depth: temporarily reduce `--depth-stop` to make VQ prediction
  easier, then increase it after the pipeline overfits.
- Check target alignment: verify cached `indices.shape[0]` equals
  `octree.nnum[depth_stop]` for the same sample and preprocessing settings.

## Sample From VAE

Generate samples from random latent vectors:

```bash
python scripts/sample_vae.py \
  --vae-ckpt outputs/vae_train/best.pt \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --output-dir outputs/vae_samples \
  --num-samples 8 \
  --temperature-split 1.0 \
  --temperature-vq 1.0 \
  --sample-tokens
```

For deterministic argmax sampling:

```bash
python scripts/sample_vae.py \
  --vae-ckpt outputs/vae_train/best.pt \
  --vqvae-ckpt ckpt/vqvae_large_im5_uncond_bsq32.pth \
  --output-dir outputs/vae_samples_argmax \
  --num-samples 8 \
  --no-sample-tokens
```

Sampling writes:

- `*_sample.pt`: latent `z`, predicted split tokens, predicted VQ indices,
  near-surface points, and SDF values
- `*_surface.ply`: near-surface point cloud sampled from the decoded SDF field

The current sampler exports near-surface point clouds rather than marching-cubes
meshes. It is intended as a fast qualitative check for the generated implicit
field.

## Current Model Flow

Training:

```text
GT octree split tokens + GT VQ bits
        -> OctFormer-style encoder
        -> mu, logvar
        -> z
        -> z-conditioned parent/sibling decoder
        -> split logits + VQ logits
```

Generation:

```text
z ~ N(0, I)
        -> root hidden = z_proj(z)
        -> root 8 child hidden states
        -> recursive split sampling
        -> VQ bit sampling at depth_stop
        -> frozen OctGPT VQ-VAE decode_code
        -> SDF/MPU field
        -> near-surface point cloud
```

The old split-only and VQ-only scripts are obsolete because they depended on the
removed seed-based initialization path. Use:

```text
scripts/train_vae.py
scripts/sample_vae.py
```

## Design Notes

### Project Description

将 Kaiming He 等人提出的分形自回归模型 (Fractal Generative Model)
从二维图像生成拓展到三维形状生成。分形自回归模型通过递归调用
自回归生成模块，将复杂图像分解为多层级、局部化的生成过程。

3D 数据尤其是八叉树结构天然具有层级空间结构，因此可结合八叉树
表示，将三维空间递归划分为多个子区域，并在每一层使用自回归模型
预测子区域的占据状态、分裂情况和局部几何特征。项目参考 OctGPT
中的八叉树建模与多尺度 3D 生成思路，尝试构建一种从粗到细生成
3D 形状的分形自回归框架，并在 ShapeNet airplane 类上验证其可行性。

### Core Idea: Parent / Uncle / Child Modeling

采用分形思想，使用“父-伯-子”的局部生成方式。假设现在已经求出
第 `i` 层节点，要预测第 `i+1` 层子节点的分裂情况，则对每个 parent
构造一个局部 AR 序列：

```text
parent/sibling hidden context
已生成 child hidden
当前 child query
```

经过 Transformer 后得到每个属于 `i+1` 层的 child hidden，再通过
预测 head 判断每个 child 是否 split。最终具体几何内容复用 OctGPT
的 VQ-VAE 模块。

这种父伯子的建模方式，是分形思想下的小块局部预测，和 OctGPT
一次预测第 `i` 层所有节点不同。它避免了当第 `i` 层节点数过多时，
父节点信息在长序列中距离过远、只能通过 dilated window 等方式间接
看到的问题；因为每次建模的都是一个小局部，所以局部 attention 更适用。

### Hidden-State Decoder Design

核心修改思路：

- 使用 parent hidden state 和 uncle hidden state。
- uncle 的找法是当前 parent 所在 sibling group 中的其他 7 个节点。
- 这些同层节点不论最终是否 split，都会有 hidden state。
- child 预测的内容也是 hidden state，然后用 `split_head` 预测 split。
- 删除 split embedding 和 mask token 作为 child AR 输入的用途。
- VQ 也通过 `vq_head(hidden)` 预测。
- `role_emb` 只区分 parent 和 uncle。
- child query 不使用 role embedding，而使用局部 octant position。
- 弃用全局 absolute 3D position embedding，改为局部位置坐标。

局部位置编码使用 child id 的 3-bit octant 坐标：

```text
x = id & 1
y = (id >> 1) & 1
z = (id >> 2) & 1

octant_pos = sx * emb_x + sy * emb_y + sz * emb_z
```

其中 `emb_x / emb_y / emb_z` 是可学习向量。这个位置编码用于 child
query，也用于 encoder token 的局部位置标识。

AR hidden 生成方式：

- 对每个 parent，context 是它所在 sibling group 的 8 个同层节点 hidden。
- 当前 parent 加 parent role embedding。
- 其他 7 个 sibling 加 uncle role embedding。
- 预测 child `k` 时输入：

```text
8 个 parent/sibling context hidden
已生成的 0..k-1 child hidden
当前 child 的 query: octant_pos(k)
```

Transformer 输出当前 query 的 hidden，作为 child `k` 的 hidden state。
8 个 child hidden 全部生成后 scatter 到下一层 hidden buffer；只有真实或
预测存在的 child indices 参与下一层结构。

### Why VAE

一个关键问题是：生成模型生成“具体哪一个实例”的条件从何处来？

固定某个类别时，OctGPT 每层八叉树生成都有随机性；FractalGen 中一个
patch 会递归生成到底层图案。但它们都不是从一开始就显式知道“这个类别
中具体要生成哪一个实例”。

当前父伯 hidden-state 设计隐含一个假设：从最上层 root 开始就应该知道
整体形状是什么。这更符合人们对生成模型的直觉，但如果直接把 root 设计
成一个随机采样量，学习不到结构化 latent；类似结构的 shape，其根部表示
按理应该接近，而无监督随机 root 做不到这一点。

因此采用 VAE 思路：

```text
GT octree / GT VQ tokens
        -> encoder
        -> latent z
        -> decoder 起始条件
        -> split / VQ prediction losses
```

最终生成时，从 prior 中随机采样 `z` 来生成。

### TreeVAE / FractalOctreeVAE

```text
Encoder:
  OctFormer-style encoder:
  GT octree structure + VQ tokens -> mu, logvar

Latent:
  z = mu + exp(0.5 * logvar) * eps

Decoder:
  root hidden = z_proj(z)
  root hidden -> first 8 child hidden states
  parent/sibling hidden AR -> child hidden
  child hidden -> split logits / VQ logits

Loss:
  split CE + VQ CE + beta KL
```

Historical note: an earlier design considered
`full-depth hidden = seed + z_proj(z) + local octant pos`, but the current
implementation removes `seed_token`; the root hidden is fully determined by
`z_proj(z)`.
