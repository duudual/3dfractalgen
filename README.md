## project 描述
将 Kaiming He 等人提出的分形自回归模型 (Fractal Generative Model) 从二维图像生成拓展到三维形
状生成。分形自回归模型通过递归调用自回归生成模块，将复杂图像分解为多层级、局部化的生成过
程。3D数据尤其是八叉树结构天然具有层级空间结构，因此可结合八叉树表示，将三维空间递归划分
为多个子区域，并在每一层使用自回归模型预测子区域的占据状态、分裂情况和局部几何特征。项目
将参考 OctGPT 中的八叉树建模与多尺度3D生成思路，尝试构建一种从粗到细生成3D形状的分形自回
归框架，并在简单3D数据集如ShapeNet的airplane类上验证其可行性。

## 核心设计
采用分形思想，父伯子：现在已经求出第i层节点，要预测第i层节点的分裂情况，使用AR，传入的数据是该i层节点的同属于一个i-1层父节点的兄弟的token和该节点的token，以及要预测8个子节点的token,组成一个seq，经过transformer，使用AR得到每个属于i+1层的子节点的token，然后经过一个预测head预测每个节点的split or no split. 最终生成具体的点的内容复用OctGPT的 VQ-VAE 模块.
这种父伯子的建模方式，是分型思想的小块内可独自预测，与OctGPT的一次预测第i层所有的不同.这种设计本身避免了OctGPT可能出现第i层节点数过多时，由于父节点顺序排放在最前面，导致的没法看到父节点，而只能通过dilated_window等方式间接看到的缺陷,因为每次都是一个小局部，局部attention很适用.
## ShapeNet octree preprocessing

The raw ShapeNet category directory is expected to contain samples like:

```text
data/02691156/<uid>/models/model_normalized.obj
```

Convert OBJ meshes to OctGPT-compatible point clouds:

```bash
cd 3dfractalgen
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --points 200000
```

This writes `pointcloud.npz` into each shape directory. Each file contains:

- `points`: sampled surface points, shape `[N, 3]`, `float16`
- `normals`: corresponding face normals, shape `[N, 3]`, `float16`

The preprocessing follows OctGPT's ShapeNet convention: meshes are normalized
to a centered unit cube, scaled by `--mesh-scale 0.8`, and sampled with
`trimesh.sample.sample_surface` when `trimesh` is installed. The octree
dataset therefore uses `points_scale=1.0` by default.

Then verify OCNN octree construction:

```bash
python scripts/inspect_octree.py --data data/02691156 --depth 8 --full-depth 3
```

For a quick smoke test on only a few meshes:

```bash
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --points 4096 --limit 2 --output-root outputs/airplane_pointcloud_smoke --overwrite
python scripts/inspect_octree.py --data outputs/airplane_pointcloud_smoke --depth 8 --full-depth 3
```

To keep preprocessed data separate from raw ShapeNet meshes, use:

```bash
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --output-root data/02691156_pointcloud
python scripts/inspect_octree.py --data data/02691156_pointcloud --depth 8 --full-depth 3
```

To also generate OctGPT-style SDF supervision for VQ-VAE/SDF training, install
`mesh2sdf` and add `--with-sdf`:

```bash
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --output-root data/02691156_sdf --with-sdf
```

This additionally writes `sdf.npz` into each output shape directory:

- `points`: SDF query points in `[-1, 1]`, shape `[M, 3]`, `float16`
- `grad`: interpolated SDF gradients, shape `[M, 3]`, `float16`
- `sdf`: signed distance values, shape `[M]`, `float16`

The SDF path mirrors `octgpt/tools/sample_sdf.py`: it computes a `2 ** depth`
SDF grid with `mesh2sdf`, builds an OCNN octree from the sampled surface point
cloud, samples random query points inside octree nodes, interpolates SDF values
and gradients, and saves up to `--sdf-max-samples 400000` samples per shape.

To write `pointcloud.npz` back into each raw shape directory, omit `--output-root`:

```bash
python scripts/prepare_shapenet_pointclouds.py --data data/02691156 --points 200000
python scripts/inspect_octree.py --data data/02691156 --depth 8 --full-depth 3
```
