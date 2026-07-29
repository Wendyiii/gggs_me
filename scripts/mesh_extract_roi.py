#!/usr/bin/env python3
"""Two-process ROI mesh extraction for GGGS.

Run ``render-cache`` and ``fuse`` as separate commands. This keeps GGGS
PyTorch/CUDA rendering and Open3D CPU TSDF integration in different processes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_render_cache() -> None:
    """Render RGB, median depth, intrinsics and extrinsics to per-view NPZ."""
    # Open3D must not be imported in this process.
    import numpy as np
    import torch

    from arguments import ModelParams, PipelineParams, get_combined_args
    from gaussian_renderer import render
    from scene import GaussianModel, Scene
    from utils.general_utils import safe_state

    parser = argparse.ArgumentParser(description=run_render_cache.__doc__)
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree, dataset.sg_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    cache_dir = args.cache_dir or (
        Path(dataset.model_path) / f"mesh_cache_{scene.loaded_iter}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(cache_dir.glob("*.npz"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"{cache_dir} already contains {len(existing)} NPZ files; "
            "use --overwrite or skip directly to the fuse command"
        )

    background_value = 1.0 if args.background == "white" else 0.0
    background = torch.tensor(
        [background_value] * 3, dtype=torch.float32, device="cuda"
    )
    cameras = scene.getTrainCameras()
    cache_names: list[str] = []

    with torch.no_grad():
        for index, camera in enumerate(cameras):
            package = render(
                camera, gaussians, pipe, background, dataset.kernel_size
            )
            color = (
                torch.clamp(package["render"], 0.0, 1.0)
                .cpu()
                .numpy()
                .transpose(1, 2, 0)
                .astype(np.float32)
            )
            depth = package["median_depth"].clone()
            if camera.gt_mask is not None:
                depth[camera.gt_mask < 0.5] = 0

            intrinsic = np.asarray(
                [
                    [camera.Fx, 0.0, camera.Cx],
                    [0.0, camera.Fy, camera.Cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            extrinsic = (
                camera.world_view_transform.T.cpu().numpy().astype(np.float64)
            )
            cache_name = f"{index:03d}.npz"
            np.savez(
                cache_dir / cache_name,
                depth=np.ascontiguousarray(depth[0].cpu().numpy(), dtype=np.float32),
                color=np.ascontiguousarray(color, dtype=np.float32),
                intrinsic=intrinsic,
                extrinsic=extrinsic,
                image_name=np.asarray(camera.image_name),
            )
            cache_names.append(cache_name)
            print(f"cached {index + 1}/{len(cameras)}: {camera.image_name}", flush=True)

    manifest = {
        "model_path": str(Path(dataset.model_path).resolve()),
        "iteration": int(scene.loaded_iter),
        "view_count": len(cameras),
        "background": args.background,
        "files": cache_names,
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"cache saved: {cache_dir}")


def run_fuse() -> None:
    """Crop cached depths in world coordinates and fuse an Open3D CPU TSDF."""
    # PyTorch and GGGS CUDA extensions must not be imported in this process.
    import numpy as np
    import open3d as o3d
    import open3d.core as o3c

    parser = argparse.ArgumentParser(description=run_fuse.__doc__)
    parser.add_argument("-m", "--model-path", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=10000)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--xmin", type=float, default=-1.4)
    parser.add_argument("--xmax", type=float, default=1.4)
    parser.add_argument("--ymin", type=float, default=-1.4)
    parser.add_argument("--ymax", type=float, default=1.4)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=2.3)
    parser.add_argument("--voxel-size", type=float, default=0.004)
    parser.add_argument("--block-resolution", type=int, default=16)
    parser.add_argument("--block-count", type=int, default=50000)
    parser.add_argument("--depth-min", type=float, default=1.0)
    parser.add_argument("--depth-max", type=float, default=8.0)
    args = parser.parse_args()

    if not (args.xmin < args.xmax and args.ymin < args.ymax and args.zmin < args.zmax):
        raise ValueError("Each ROI minimum must be smaller than its maximum")
    if args.voxel_size <= 0:
        raise ValueError("--voxel-size must be positive")
    if args.depth_min <= 0 or args.depth_min >= args.depth_max:
        raise ValueError("Invalid depth range")

    cache_dir = args.cache_dir or (
        args.model_path / f"mesh_cache_{args.iteration}"
    )
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)

    manifest_path = cache_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_files = [cache_dir / name for name in manifest["files"]]
    else:
        # Backward-compatible with the original 70-view cache.
        cache_files = sorted(cache_dir.glob("*.npz"))
    if not cache_files:
        raise FileNotFoundError(f"No NPZ cache files found in {cache_dir}")

    output = args.output or (
        args.model_path / "recon_center_roi_4mm.ply"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    roi_min = np.asarray([args.xmin, args.ymin, args.zmin], dtype=np.float64)
    roi_max = np.asarray([args.xmax, args.ymax, args.zmax], dtype=np.float64)

    device = o3c.Device("CPU:0")
    volume = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(1, 1, 3),
        voxel_size=args.voxel_size,
        block_resolution=args.block_resolution,
        block_count=args.block_count,
        device=device,
    )

    for index, cache_file in enumerate(cache_files):
        with np.load(cache_file) as cached:
            depth_array = cached["depth"].astype(np.float32, copy=True)
            color_array = np.ascontiguousarray(cached["color"], dtype=np.float32)
            intrinsic_array = cached["intrinsic"].astype(np.float64, copy=False)
            extrinsic_array = cached["extrinsic"].astype(np.float64, copy=False)

        height, width = depth_array.shape
        pixel_y, pixel_x = np.mgrid[:height, :width]
        camera_points = np.stack(
            [
                (pixel_x - intrinsic_array[0, 2])
                * depth_array
                / intrinsic_array[0, 0],
                (pixel_y - intrinsic_array[1, 2])
                * depth_array
                / intrinsic_array[1, 1],
                depth_array,
                np.ones_like(depth_array),
            ],
            axis=-1,
        )
        world_points = camera_points @ np.linalg.inv(extrinsic_array).T
        keep = (
            np.isfinite(depth_array)
            & (depth_array > 0)
            & np.all(world_points[..., :3] >= roi_min, axis=-1)
            & np.all(world_points[..., :3] <= roi_max, axis=-1)
        )
        depth_array[~keep] = 0

        depth = o3d.t.geometry.Image(np.ascontiguousarray(depth_array))
        color = o3d.t.geometry.Image(color_array)
        intrinsic = o3c.Tensor(intrinsic_array)
        extrinsic = o3c.Tensor(extrinsic_array)
        blocks = volume.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, args.depth_min, args.depth_max
        )
        volume.integrate(
            blocks, depth, color, intrinsic, extrinsic,
            args.depth_min, args.depth_max
        )
        print(
            f"integrated {index + 1}/{len(cache_files)} "
            f"keep={int(keep.sum())} blocks_view={blocks.shape[0]} "
            f"hashmap={volume.hashmap().size()}",
            flush=True,
        )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    legacy_mesh = mesh.to_legacy()
    if not o3d.io.write_triangle_mesh(str(output), legacy_mesh):
        raise RuntimeError(f"Open3D failed to write {output}")
    print(
        f"saved {output} vertices={len(legacy_mesh.vertices)} "
        f"triangles={len(legacy_mesh.triangles)}"
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"render-cache", "fuse"}:
        print(
            "Usage:\n"
            "  python scripts/mesh_extract_roi.py render-cache [GGGS options]\n"
            "  CUDA_VISIBLE_DEVICES='' python scripts/mesh_extract_roi.py fuse [options]",
            file=sys.stderr,
        )
        raise SystemExit(2)
    command = sys.argv.pop(1)
    if command == "render-cache":
        run_render_cache()
    else:
        run_fuse()


if __name__ == "__main__":
    main()
