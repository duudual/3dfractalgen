from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

try:
  import trimesh
except ImportError:  # pragma: no cover - exercised only in minimal envs.
  trimesh = None

try:
  import mesh2sdf
except ImportError:  # pragma: no cover - exercised only when SDF is requested.
  mesh2sdf = None

try:
  import ocnn
except ImportError:  # pragma: no cover - exercised only when SDF is requested.
  ocnn = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "02691156"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Sample ShapeNet OBJ meshes into OctGPT-style pointcloud.npz.")
  parser.add_argument(
    "--data",
    default=str(DEFAULT_DATA_DIR),
    help="ShapeNet category directory, e.g. data/02691156.")
  parser.add_argument(
    "--points",
    type=int,
    default=200000,
    help="Number of surface points sampled per mesh.")
  parser.add_argument(
    "--mesh-scale",
    type=float,
    default=0.8,
    help="Scale applied after unit-cube normalization, matching OctGPT.")
  parser.add_argument(
    "--output-root",
    default=None,
    help="Optional output directory. Defaults to writing beside each shape.")
  parser.add_argument(
    "--obj-name",
    default="models/model_normalized.obj",
    help="Relative OBJ path inside each shape directory.")
  parser.add_argument(
    "--seed",
    type=int,
    default=0,
    help="Random seed for deterministic sampling.")
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing pointcloud.npz files.")
  parser.add_argument(
    "--no-normalize",
    action="store_true",
    help="Disable OctGPT-style unit-cube normalization.")
  parser.add_argument(
    "--dtype",
    choices=("float16", "float32"),
    default="float16",
    help="Saved array dtype. OctGPT uses float16.")
  parser.add_argument(
    "--with-sdf",
    action="store_true",
    help="Also generate OctGPT-style sdf.npz with points, grad, and sdf.")
  parser.add_argument(
    "--sdf-depth",
    type=int,
    default=8,
    help="SDF grid depth. The grid size is 2 ** sdf_depth.")
  parser.add_argument(
    "--sdf-full-depth",
    type=int,
    default=4,
    help="Lowest octree depth used for SDF sampling, matching OctGPT.")
  parser.add_argument(
    "--sdf-band",
    type=float,
    default=0.05,
    help="Truncation band used when validating interpolated SDF samples.")
  parser.add_argument(
    "--sdf-samples-per-node",
    type=int,
    default=4,
    help="Number of random SDF samples per octree node.")
  parser.add_argument(
    "--sdf-max-samples",
    type=int,
    default=400000,
    help="Maximum SDF samples saved per shape.")
  parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional maximum number of shapes to process.")
  return parser.parse_args()


def shape_dirs(root: Path, obj_name: str) -> list[Path]:
  dirs = []
  for path in sorted(root.iterdir()):
    if (path / obj_name).exists():
      dirs.append(path)
  return dirs


def _vertex_index(token: str, vertex_count: int) -> int:
  value = int(token.split("/")[0])
  if value < 0:
    return vertex_count + value
  return value - 1


def load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
  vertices: list[list[float]] = []
  faces: list[list[int]] = []

  with path.open("r", encoding="utf-8", errors="ignore") as handle:
    for line in handle:
      if line.startswith("v "):
        _, x, y, z, *_ = line.split()
        vertices.append([float(x), float(y), float(z)])
      elif line.startswith("f "):
        items = line.split()[1:]
        if len(items) < 3:
          continue
        indices = [_vertex_index(item, len(vertices)) for item in items]
        for i in range(1, len(indices) - 1):
          faces.append([indices[0], indices[i], indices[i + 1]])

  if not vertices:
    raise ValueError(f"No vertices found in {path}")
  if not faces:
    raise ValueError(f"No faces found in {path}")
  return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def normalize_vertices(vertices: np.ndarray) -> np.ndarray:
  vmin = vertices.min(axis=0)
  vmax = vertices.max(axis=0)
  center = (vmin + vmax) * 0.5
  scale = float((vmax - vmin).max())
  if scale <= 0:
    raise ValueError("Degenerate mesh bounding box")
  return (vertices - center) / scale


def scale_to_unit_cube(vertices: np.ndarray) -> np.ndarray:
  vmin = vertices.min(axis=0)
  vmax = vertices.max(axis=0)
  center = (vmin + vmax) * 0.5
  scale = float((vmax - vmin).max())
  if scale <= 0:
    raise ValueError("Degenerate mesh bounding box")
  return (vertices - center) * (2.0 / scale)


def sample_surface_points(
  vertices: np.ndarray,
  faces: np.ndarray,
  num_points: int,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  triangles = vertices[faces]
  edge1 = triangles[:, 1] - triangles[:, 0]
  edge2 = triangles[:, 2] - triangles[:, 0]
  cross = np.cross(edge1, edge2)
  double_area = np.linalg.norm(cross, axis=1)
  valid = double_area > 1e-12
  if not np.any(valid):
    raise ValueError("Mesh has no non-degenerate triangles")

  triangles = triangles[valid]
  cross = cross[valid]
  double_area = double_area[valid]
  prob = double_area / double_area.sum()

  tri_idx = rng.choice(len(triangles), size=num_points, p=prob)
  chosen = triangles[tri_idx]

  u = rng.random((num_points, 1), dtype=np.float32)
  v = rng.random((num_points, 1), dtype=np.float32)
  flip = (u + v) > 1.0
  u[flip] = 1.0 - u[flip]
  v[flip] = 1.0 - v[flip]

  points = chosen[:, 0] + u * (chosen[:, 1] - chosen[:, 0])
  points += v * (chosen[:, 2] - chosen[:, 0])

  normals = cross[tri_idx]
  normals = normals / np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
  return points.astype(np.float32), normals.astype(np.float32)


def load_with_trimesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
  if trimesh is None:
    raise ImportError("trimesh is not installed")
  mesh = trimesh.load(path, force="mesh")
  if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump().sum()
  if mesh.vertices.size == 0 or mesh.faces.size == 0:
    raise ValueError(f"Invalid mesh loaded from {path}")
  return (
    np.asarray(mesh.vertices, dtype=np.float32),
    np.asarray(mesh.faces, dtype=np.int64),
  )


def sample_with_trimesh(
  vertices: np.ndarray,
  faces: np.ndarray,
  num_points: int,
) -> tuple[np.ndarray, np.ndarray]:
  if trimesh is None:
    raise ImportError("trimesh is not installed")
  mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
  points, face_idx = trimesh.sample.sample_surface(mesh, num_points)
  normals = mesh.face_normals[face_idx]
  return points.astype(np.float32), normals.astype(np.float32)


def build_trimesh(vertices: np.ndarray, faces: np.ndarray):
  if trimesh is None:
    raise ImportError("trimesh is required for SDF preprocessing")
  return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def compute_mesh_sdf(mesh, size: int, level: float) -> tuple[object, torch.Tensor]:
  if mesh2sdf is None:
    raise ImportError(
      "mesh2sdf is required for --with-sdf. Install it with `pip install mesh2sdf`.")
  voxel_sdf, mesh_new = mesh2sdf.compute(
    mesh.vertices,
    mesh.faces,
    size,
    fix=True,
    level=level,
    return_mesh=True,
  )
  return mesh_new, torch.as_tensor(voxel_sdf)


def sample_sdf_from_octree(
  voxel_sdf: torch.Tensor,
  points: np.ndarray,
  normals: np.ndarray,
  depth: int,
  full_depth: int,
  band: float,
  samples_per_node: int,
  max_samples: int,
) -> dict[str, np.ndarray]:
  if ocnn is None:
    raise ImportError("ocnn is required for --with-sdf")
  if samples_per_node <= 0:
    raise ValueError("--sdf-samples-per-node must be positive")

  device = voxel_sdf.device
  size = int(voxel_sdf.shape[0])
  grid = torch.tensor(
    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
    dtype=torch.long,
    device=device,
  )

  point_cloud = ocnn.octree.Points(
    torch.from_numpy(points.astype(np.float32)),
    torch.from_numpy(normals.astype(np.float32)),
  ).to(device)
  octree = ocnn.octree.Octree(depth=depth, full_depth=full_depth)
  octree.build_octree(point_cloud)

  xyzs, grads, sdfs = [], [], []
  for d in range(full_depth, depth + 1):
    x, y, z, _ = octree.xyzb(d)
    xyz = torch.stack((x, y, z), dim=1).float()
    xyz = xyz[:, None, :] + torch.rand(
      xyz.shape[0], samples_per_node, 3, device=device)
    xyz = xyz.reshape(-1, 3)
    xyz = xyz * (size / (2 ** d))
    xyz = xyz[(xyz < (size - 1)).all(dim=1)]
    if xyz.numel() == 0:
      continue
    xyzs.append(xyz)

    xyzi = torch.floor(xyz)
    corners = xyzi[:, None, :] + grid[None, :, :]
    coordsf = xyz[:, None, :] - corners
    weights = (1 - coordsf.abs()).prod(dim=-1)
    corners = corners.long().reshape(-1, 3)
    cx, cy, cz = corners[:, 0], corners[:, 1], corners[:, 2]
    corner_sdf = voxel_sdf[cx, cy, cz].reshape(-1, 8)
    interp_sdf = torch.sum(corner_sdf * weights, dim=1)

    valid = (corner_sdf.abs() <= band).all(dim=1)
    interp_sdf[~valid] = (interp_sdf[~valid] > 0).float() * band

    gx = corner_sdf[:, 4] - corner_sdf[:, 0] + \
        corner_sdf[:, 5] - corner_sdf[:, 1] + \
        corner_sdf[:, 6] - corner_sdf[:, 2] + \
        corner_sdf[:, 7] - corner_sdf[:, 3]
    gy = corner_sdf[:, 2] - corner_sdf[:, 0] + \
        corner_sdf[:, 3] - corner_sdf[:, 1] + \
        corner_sdf[:, 6] - corner_sdf[:, 4] + \
        corner_sdf[:, 7] - corner_sdf[:, 5]
    gz = corner_sdf[:, 1] - corner_sdf[:, 0] + \
        corner_sdf[:, 3] - corner_sdf[:, 2] + \
        corner_sdf[:, 5] - corner_sdf[:, 4] + \
        corner_sdf[:, 7] - corner_sdf[:, 6]
    grad = torch.stack([gx, gy, gz], dim=-1)
    norm = torch.sqrt(torch.sum(grad ** 2, dim=-1, keepdims=True))
    grad = grad / (norm + 1.0e-8)
    grad[~valid] = 0.0

    sdfs.append(interp_sdf)
    grads.append(grad)

  if not xyzs:
    raise ValueError("No SDF samples were generated")

  xyz = torch.cat(xyzs, dim=0)
  sdf_points = xyz / (size / 2) - 1
  sdf_grad = torch.cat(grads, dim=0)
  sdf_values = torch.cat(sdfs, dim=0)

  count = min(max_samples, sdf_points.shape[0])
  random_idx = torch.randperm(sdf_points.shape[0], device=device)[:count]
  return {
    "points": sdf_points[random_idx].cpu().numpy().astype(np.float16),
    "grad": sdf_grad[random_idx].cpu().numpy().astype(np.float16),
    "sdf": sdf_values[random_idx].cpu().numpy().astype(np.float16),
  }


def process_shape(
  shape_dir: Path,
  output_dir: Path,
  obj_name: str,
  num_points: int,
  rng: np.random.Generator,
  overwrite: bool,
  normalize: bool,
  mesh_scale: float,
  dtype: str,
  with_sdf: bool,
  sdf_depth: int,
  sdf_full_depth: int,
  sdf_band: float,
  sdf_samples_per_node: int,
  sdf_max_samples: int,
) -> bool:
  filename_pointcloud = output_dir / "pointcloud.npz"
  filename_sdf = output_dir / "sdf.npz"
  if filename_pointcloud.exists() and (not with_sdf or filename_sdf.exists()) \
      and not overwrite:
    return False

  obj_path = shape_dir / obj_name
  try:
    vertices, faces = load_with_trimesh(obj_path)
  except ImportError:
    vertices, faces = load_obj_mesh(obj_path)

  if normalize:
    vertices = scale_to_unit_cube(vertices)
    vertices *= mesh_scale

  if trimesh is not None:
    points, normals = sample_with_trimesh(vertices, faces, num_points)
  else:
    points, normals = sample_surface_points(vertices, faces, num_points, rng)

  output_dir.mkdir(parents=True, exist_ok=True)
  array_dtype = np.float16 if dtype == "float16" else np.float32
  np.savez(
    filename_pointcloud,
    points=points.astype(array_dtype),
    normals=normals.astype(array_dtype),
  )

  if with_sdf:
    mesh = build_trimesh(vertices, faces)
    sdf_size = 2 ** sdf_depth
    sdf_level = 1.0 / sdf_size
    sdf_mesh, voxel_sdf = compute_mesh_sdf(mesh, sdf_size, sdf_level)
    if trimesh is not None and getattr(sdf_mesh, "vertices", None) is not None:
      points, normals = sample_with_trimesh(
        np.asarray(sdf_mesh.vertices, dtype=np.float32),
        np.asarray(sdf_mesh.faces, dtype=np.int64),
        num_points,
      )
      np.savez(
        filename_pointcloud,
        points=points.astype(array_dtype),
        normals=normals.astype(array_dtype),
      )
    sdf_data = sample_sdf_from_octree(
      voxel_sdf=voxel_sdf,
      points=points,
      normals=normals,
      depth=sdf_depth,
      full_depth=sdf_full_depth,
      band=sdf_band,
      samples_per_node=sdf_samples_per_node,
      max_samples=sdf_max_samples,
    )
    np.savez(filename_sdf, **sdf_data)
  return True


def main() -> None:
  args = parse_args()
  if args.with_sdf:
    missing = []
    if trimesh is None:
      missing.append("trimesh")
    if mesh2sdf is None:
      missing.append("mesh2sdf")
    if ocnn is None:
      missing.append("ocnn")
    if missing:
      raise RuntimeError(
        "--with-sdf requires missing packages: " + ", ".join(missing))

  root = Path(args.data)
  dirs = shape_dirs(root, args.obj_name)
  if args.limit is not None:
    dirs = dirs[:args.limit]
  if not dirs:
    raise RuntimeError(f"No ShapeNet OBJ meshes found under {root}")

  rng = np.random.default_rng(args.seed)
  output_root = Path(args.output_root) if args.output_root else None
  normalize = not args.no_normalize
  written = 0
  failed: list[tuple[str, str]] = []
  for shape_dir in tqdm(dirs, desc="sampling pointclouds", dynamic_ncols=True):
    output_dir = output_root / shape_dir.name if output_root else shape_dir
    try:
      if process_shape(
        shape_dir=shape_dir,
        output_dir=output_dir,
        obj_name=args.obj_name,
        num_points=args.points,
        rng=rng,
        overwrite=args.overwrite,
        normalize=normalize,
        mesh_scale=args.mesh_scale,
        dtype=args.dtype,
        with_sdf=args.with_sdf,
        sdf_depth=args.sdf_depth,
        sdf_full_depth=args.sdf_full_depth,
        sdf_band=args.sdf_band,
        sdf_samples_per_node=args.sdf_samples_per_node,
        sdf_max_samples=args.sdf_max_samples,
      ):
        written += 1
    except Exception as exc:  # noqa: BLE001
      failed.append((shape_dir.name, str(exc)))

  print(f"shapes scanned: {len(dirs)}")
  print(f"pointcloud.npz written: {written}")
  print(f"skipped existing: {len(dirs) - written - len(failed)}")
  if failed:
    print(f"failed: {len(failed)}")
    for uid, message in failed[:20]:
      print(f"  {uid}: {message}")


if __name__ == "__main__":
  main()
