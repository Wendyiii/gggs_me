#!/usr/bin/env python3
"""Build and validate a fixed-pose COLMAP model from calibrations.json.

The script deliberately never runs COLMAP mapper/bundle_adjuster: calibrated
intrinsics and world-to-camera poses stay fixed. COLMAP is only used for SIFT
extraction, matching, point triangulation, and image undistortion.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np


def natural_camera_key(name: str) -> tuple[str, int]:
    stem = Path(name).stem
    head, sep, tail = stem.rpartition("_")
    return (head if sep else stem, int(tail) if sep and tail.isdigit() else 0)


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid zero/non-finite quaternion")
    qw, qx, qy, qz = q / norm
    return np.asarray([
        [1 - 2 * (qy*qy + qz*qz), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)],
    ], dtype=np.float64)


def load_calibrations(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("calibrations")
    if not isinstance(items, list):
        raise ValueError(f"{path}: expected a top-level 'calibrations' list")
    result = {}
    for item in items:
        name = str(item["cameraSN"])
        if name in result:
            raise ValueError(f"Duplicate calibration cameraSN: {name}")
        result[name] = item
    return result


def image_files(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(files, key=lambda p: natural_camera_key(p.name))


def run(command: list[str]) -> None:
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def database_images(database: Path) -> list[tuple[int, str, int]]:
    with sqlite3.connect(database) as conn:
        rows = conn.execute(
            "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
        ).fetchall()
    return [(int(i), str(n), int(c)) for i, n, c in rows]


def write_known_model(
    output: Path,
    rows: list[tuple[int, str, int]],
    calibrations: dict[str, dict],
    world_scale: float,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    camera_lines = []
    image_lines = []
    seen_camera_ids = set()

    for image_id, image_name, camera_id in rows:
        camera_name = Path(image_name).stem
        if camera_name not in calibrations:
            raise ValueError(f"No calibration for database image {image_name}")
        c = calibrations[camera_name]
        size = c["imageSize"]
        width, height = int(size["w"]), int(size["h"])
        params = [
            c["fx"], c["fy"], c["cx"], c["cy"],
            c.get("k1", 0), c.get("k2", 0), c.get("p1", 0), c.get("p2", 0),
            c.get("k3", 0), c.get("k4", 0), 0, 0,
        ]
        if camera_id in seen_camera_ids:
            raise ValueError(
                f"camera_id {camera_id} is shared; feature extraction must use "
                "--ImageReader.single_camera_per_image 1"
            )
        seen_camera_ids.add(camera_id)
        camera_lines.append(
            f"{camera_id} FULL_OPENCV {width} {height} "
            + " ".join(f"{float(v):.17g}" for v in params)
        )

        w2c = c["world2Cam"]
        q = np.asarray([w2c[k] for k in ("qw", "qx", "qy", "qz")], dtype=np.float64)
        q /= np.linalg.norm(q)
        t = [float(w2c[k]) * world_scale for k in ("x", "y", "z")]
        image_lines.append(
            f"{image_id} " + " ".join(f"{float(v):.17g}" for v in [*q, *t])
            + f" {camera_id} {image_name}"
        )
        image_lines.append("")

    (output / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    (output / "images.txt").write_text(
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "# POINTS2D[] as (X, Y, POINT3D_ID)\n" + "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    (output / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n",
        encoding="utf-8",
    )


def validate_geometry(calibrations: dict[str, dict], names: list[str], world_scale: float) -> dict:
    centers, directions, errors = [], [], []
    for name in names:
        c = calibrations[name]
        w2c = c["world2Cam"]
        R = quaternion_to_rotation(*(float(w2c[k]) for k in ("qw", "qx", "qy", "qz")))
        t = np.asarray([w2c[k] for k in ("x", "y", "z")], dtype=np.float64) * world_scale
        centers.append(-R.T @ t)
        directions.append(R.T @ np.asarray([0.0, 0.0, 1.0]))
        errors.append(float(c.get("reprojErr", math.nan)))
    centers = np.asarray(centers)
    directions = np.asarray(directions)

    # Least-squares point closest to all optical-axis rays.
    A = np.zeros((3, 3)); b = np.zeros(3); eye = np.eye(3)
    for C, d in zip(centers, directions):
        M = eye - np.outer(d, d)
        A += M; b += M @ C
    target = np.linalg.solve(A, b)
    depths = np.sum((target[None] - centers) * directions, axis=1)
    ray_errors = np.linalg.norm(np.cross(target[None] - centers, directions), axis=1)

    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    upper = distances[np.triu_indices(len(centers), 1)]
    nearest = np.where(distances + np.eye(len(centers)) * 1e100 > 0, distances + np.eye(len(centers)) * 1e100, 1e100).min(axis=1)
    scale = float(np.median(np.linalg.norm(centers - np.median(centers, axis=0), axis=1)))
    duplicate_threshold = max(scale * 1e-6, 1e-9)

    report = {
        "world_scale": world_scale,
        "camera_count": len(names),
        "camera_names": names,
        "center_bbox_min": centers.min(axis=0).tolist(),
        "center_bbox_max": centers.max(axis=0).tolist(),
        "pair_distance_min": float(upper.min()),
        "pair_distance_median": float(np.median(upper)),
        "nearest_center_min": float(nearest.min()),
        "near_duplicate_pairs": int(np.count_nonzero(upper < duplicate_threshold)),
        "estimated_rig_target": target.tolist(),
        "target_positive_depth_cameras": int(np.count_nonzero(depths > 0)),
        "target_ray_error_median": float(np.median(ray_errors)),
        "calibration_reproj_error_mean": float(np.nanmean(errors)),
        "calibration_reproj_error_max": float(np.nanmax(errors)),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="New working directory; must not already exist")
    parser.add_argument("--output", type=Path, required=True,
                        help="New final undistorted GGGS dataset directory")
    parser.add_argument("--colmap", default="colmap")
    parser.add_argument("--gpu-index", default="0")
    parser.add_argument("--max-image-size", type=int, default=3200)
    parser.add_argument("--world-scale", type=float, default=1.0,
                        help="Scale applied to world2Cam translation; use 0.001 for mm to m")
    parser.add_argument("--cpu", action="store_true", help="Use CPU SIFT")
    parser.add_argument("--skip-matching", action="store_true",
                        help="Only write and validate the known model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.world_scale) or args.world_scale <= 0:
        raise ValueError("--world-scale must be a finite positive number")
    for source in (args.calibration, args.image_path):
        if not source.exists():
            raise FileNotFoundError(source)
    if args.workspace.exists() or args.output.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing workspace/output. Choose new paths:\n"
            f"  workspace={args.workspace}\n  output={args.output}"
        )

    images = image_files(args.image_path)
    calibrations = load_calibrations(args.calibration)
    image_names = [p.stem for p in images]
    missing_calib = sorted(set(image_names) - set(calibrations), key=natural_camera_key)
    missing_images = sorted(set(calibrations) - set(image_names), key=natural_camera_key)
    if missing_calib or missing_images:
        raise ValueError(
            f"Image/calibration mismatch. missing_calibration={missing_calib}, "
            f"missing_images={missing_images}"
        )
    for image in images:
        from PIL import Image
        with Image.open(image) as im:
            c = calibrations[image.stem]
            expected = (int(c["imageSize"]["w"]), int(c["imageSize"]["h"]))
            if im.size != expected:
                raise ValueError(f"{image.name}: image size {im.size}, calibration size {expected}")

    args.workspace.mkdir(parents=True)
    database = args.workspace / "database.db"
    known = args.workspace / "sparse_known"
    triangulated = args.workspace / "sparse_triangulated"

    feature_command = [
        args.colmap, "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(args.image_path),
        "--ImageReader.camera_model", "FULL_OPENCV",
        "--ImageReader.single_camera_per_image", "1",
        "--FeatureExtraction.use_gpu", "0" if args.cpu else "1",
        "--FeatureExtraction.gpu_index", str(args.gpu_index),
        "--SiftExtraction.max_image_size", str(args.max_image_size),
    ]
    run(feature_command)
    rows = database_images(database)
    if len(rows) != len(images):
        raise RuntimeError(f"COLMAP database has {len(rows)} images, expected {len(images)}")
    write_known_model(known, rows, calibrations, args.world_scale)

    report = validate_geometry(calibrations, [Path(name).stem for _, name, _ in rows], args.world_scale)
    report["input_image_count"] = len(images)
    report["database_image_count"] = len(rows)
    report["database_unique_camera_count"] = len({camera_id for _, _, camera_id in rows})
    report_path = args.workspace / "calibration_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nCalibration geometry validation:")
    print(json.dumps(report, indent=2))
    run([args.colmap, "model_analyzer", "--path", str(known)])

    if args.skip_matching:
        print(f"\nStopped after fixed-model validation: {known}")
        return

    matcher = [
        args.colmap, "exhaustive_matcher", "--database_path", str(database),
        "--FeatureMatching.use_gpu", "0" if args.cpu else "1",
        "--FeatureMatching.gpu_index", str(args.gpu_index),
    ]
    run(matcher)
    triangulated.mkdir()
    run([
        args.colmap, "point_triangulator",
        "--database_path", str(database),
        "--image_path", str(args.image_path),
        "--input_path", str(known),
        "--output_path", str(triangulated),
        "--refine_intrinsics", "0",
        "--Mapper.fix_existing_frames", "1",
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.ba_refine_sensor_from_rig", "0",
    ])
    run([args.colmap, "model_analyzer", "--path", str(triangulated)])

    args.output.mkdir(parents=True)
    run([
        args.colmap, "image_undistorter",
        "--image_path", str(args.image_path),
        "--input_path", str(triangulated),
        "--output_path", str(args.output),
        "--output_type", "COLMAP",
    ])
    sparse = args.output / "sparse"
    sparse_zero = sparse / "0"
    sparse_zero.mkdir()
    for item in list(sparse.iterdir()):
        if item != sparse_zero:
            shutil.move(str(item), sparse_zero / item.name)
    shutil.copy2(report_path, args.output / report_path.name)
    run([args.colmap, "model_analyzer", "--path", str(sparse_zero)])
    print("\nSUCCESS")
    print(f"Fixed raw model:       {known}")
    print(f"Triangulated model:    {triangulated}")
    print(f"GGGS training dataset: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nCOLMAP command failed with exit code {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode)
