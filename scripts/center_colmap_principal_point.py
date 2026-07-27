#!/usr/bin/env python3
"""Create a centered-principal-point COLMAP dataset without changing geometry."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path

import cv2
import numpy as np

PINHOLE_MODEL_ID = 1


def read_bytes(handle, size: int, fmt: str):
    return struct.unpack("<" + fmt, handle.read(size))


def write_bytes(handle, fmt: str, *values) -> None:
    handle.write(struct.pack("<" + fmt, *values))


def read_cameras(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("rb") as handle:
        count = read_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            camera_id, model_id, width, height = read_bytes(handle, 24, "iiQQ")
            if model_id != PINHOLE_MODEL_ID:
                raise ValueError(
                    f"Camera {camera_id} uses model id {model_id}; expected PINHOLE (1)"
                )
            params = np.asarray(read_bytes(handle, 32, "dddd"), dtype=np.float64)
            cameras[camera_id] = {
                "id": camera_id, "model_id": model_id,
                "width": width, "height": height, "params": params,
            }
    return cameras


def write_cameras(path: Path, cameras: dict[int, dict]) -> None:
    with path.open("wb") as handle:
        write_bytes(handle, "Q", len(cameras))
        for camera_id in sorted(cameras):
            camera = cameras[camera_id]
            write_bytes(
                handle, "iiQQ", camera_id, camera["model_id"],
                camera["width"], camera["height"],
            )
            write_bytes(handle, "dddd", *camera["params"])


def read_images(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("rb") as handle:
        count = read_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            values = read_bytes(handle, 64, "idddddddi")
            image_id = values[0]
            name_bytes = bytearray()
            while True:
                char = handle.read(1)
                if char == b"\x00":
                    break
                if not char:
                    raise EOFError("Unexpected EOF while reading image name")
                name_bytes.extend(char)
            point_count = read_bytes(handle, 8, "Q")[0]
            raw_points = read_bytes(handle, 24 * point_count, "ddq" * point_count)
            xys = np.column_stack((
                np.asarray(raw_points[0::3], dtype=np.float64),
                np.asarray(raw_points[1::3], dtype=np.float64),
            ))
            point_ids = np.asarray(raw_points[2::3], dtype=np.int64)
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(values[1:5], dtype=np.float64),
                "tvec": np.asarray(values[5:8], dtype=np.float64),
                "camera_id": values[8],
                "name": name_bytes.decode("utf-8"),
                "xys": xys,
                "point3D_ids": point_ids,
            }
    return images


def write_images(path: Path, images: dict[int, dict]) -> None:
    with path.open("wb") as handle:
        write_bytes(handle, "Q", len(images))
        for image_id in sorted(images):
            image = images[image_id]
            write_bytes(
                handle, "idddddddi", image_id,
                *image["qvec"], *image["tvec"], image["camera_id"],
            )
            handle.write(image["name"].encode("utf-8") + b"\x00")
            write_bytes(handle, "Q", len(image["xys"]))
            for xy, point_id in zip(image["xys"], image["point3D_ids"]):
                write_bytes(handle, "ddq", float(xy[0]), float(xy[1]), int(point_id))


def read_point_xyz(path: Path) -> dict[int, np.ndarray]:
    points = {}
    with path.open("rb") as handle:
        count = read_bytes(handle, 8, "Q")[0]
        for _ in range(count):
            values = read_bytes(handle, 43, "QdddBBBd")
            point_id = values[0]
            points[point_id] = np.asarray(values[1:4], dtype=np.float64)
            track_length = read_bytes(handle, 8, "Q")[0]
            handle.seek(8 * track_length, 1)
    return points


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    q = qvec / np.linalg.norm(qvec)
    w, x, y, z = q
    return np.asarray([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def reprojection_errors(
    cameras: dict[int, dict],
    images: dict[int, dict],
    points: dict[int, np.ndarray],
) -> np.ndarray:
    errors = []
    for image in images.values():
        camera = cameras[image["camera_id"]]
        fx, fy, cx, cy = camera["params"]
        rotation = qvec_to_rotation(image["qvec"])
        valid = np.asarray([
            point_id >= 0 and int(point_id) in points
            for point_id in image["point3D_ids"]
        ])
        if not valid.any():
            continue
        xyz = np.stack([points[int(i)] for i in image["point3D_ids"][valid]])
        camera_xyz = xyz @ rotation.T + image["tvec"]
        predicted = np.column_stack((
            fx * camera_xyz[:, 0] / camera_xyz[:, 2] + cx,
            fy * camera_xyz[:, 1] / camera_xyz[:, 2] + cy,
        ))
        errors.extend(np.linalg.norm(predicted - image["xys"][valid], axis=1))
    return np.asarray(errors, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_sparse = args.source / "sparse" / "0"
    source_images = args.source / "images"
    for required in (
        source_sparse / "cameras.bin",
        source_sparse / "images.bin",
        source_sparse / "points3D.bin",
        source_images,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")

    cameras = read_cameras(source_sparse / "cameras.bin")
    images = read_images(source_sparse / "images.bin")
    points = read_point_xyz(source_sparse / "points3D.bin")
    before_errors = reprojection_errors(cameras, images, points)

    offsets = {}
    centered_cameras = {}
    for camera_id, camera in cameras.items():
        params = camera["params"].copy()
        target = np.asarray([
            (camera["width"] - 1.0) / 2.0,
            (camera["height"] - 1.0) / 2.0,
        ])
        offset = target - params[2:4]
        offsets[camera_id] = offset
        params[2:4] = target
        centered_cameras[camera_id] = {**camera, "params": params}

    centered_images_data = {}
    for image_id, image in images.items():
        centered_images_data[image_id] = {
            **image,
            "xys": image["xys"] + offsets[image["camera_id"]][None],
        }

    output_sparse = args.output / "sparse" / "0"
    output_images = args.output / "images"
    output_sparse.mkdir(parents=True)
    output_images.mkdir()
    write_cameras(output_sparse / "cameras.bin", centered_cameras)
    write_images(output_sparse / "images.bin", centered_images_data)

    for filename in ("points3D.bin", "points3D.ply", "rigs.bin", "frames.bin"):
        source = source_sparse / filename
        if source.exists():
            shutil.copy2(source, output_sparse / filename)
    for filename in ("calibration_validation.json",):
        source = args.source / filename
        if source.exists():
            shutil.copy2(source, args.output / filename)

    for image in images.values():
        source = source_images / image["name"]
        destination = output_images / image["name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        pixels = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise RuntimeError(f"OpenCV failed to read {source}")
        height, width = pixels.shape[:2]
        dx, dy = offsets[image["camera_id"]]
        matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        shifted = cv2.warpAffine(
            pixels, matrix, (width, height), flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        params = ([cv2.IMWRITE_JPEG_QUALITY, 100]
                  if destination.suffix.lower() in {".jpg", ".jpeg"} else [])
        if not cv2.imwrite(str(destination), shifted, params):
            raise RuntimeError(f"OpenCV failed to write {destination}")

    after_cameras = read_cameras(output_sparse / "cameras.bin")
    after_images = read_images(output_sparse / "images.bin")
    after_errors = reprojection_errors(after_cameras, after_images, points)
    values = np.asarray(list(offsets.values()))
    principal_residuals = np.asarray([
        np.linalg.norm(camera["params"][2:4] - np.asarray([
            (camera["width"] - 1.0) / 2.0,
            (camera["height"] - 1.0) / 2.0,
        ]))
        for camera in after_cameras.values()
    ])
    if len(after_images) != len(images) or len(after_cameras) != len(cameras):
        raise RuntimeError("Camera/image count changed during conversion")
    if before_errors.shape != after_errors.shape:
        raise RuntimeError("Observation count changed during conversion")
    error_delta = np.abs(before_errors - after_errors)
    if principal_residuals.max(initial=0.0) > 1e-12:
        raise RuntimeError("Centered camera model still has a principal-point residual")
    if error_delta.max(initial=0.0) > 1e-8:
        raise RuntimeError("Reprojection errors changed unexpectedly")

    report = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "camera_count": len(cameras),
        "image_count": len(images),
        "point3D_count": len(points),
        "valid_observation_count": len(before_errors),
        "target_definition": "((width - 1) / 2, (height - 1) / 2)",
        "border_fill": "black",
        "interpolation": "OpenCV INTER_LANCZOS4",
        "dx_min": float(values[:, 0].min()),
        "dx_max": float(values[:, 0].max()),
        "dy_min": float(values[:, 1].min()),
        "dy_max": float(values[:, 1].max()),
        "max_shift_pixels": float(np.linalg.norm(values, axis=1).max()),
        "principal_point_residual_max": float(principal_residuals.max(initial=0.0)),
        "reprojection_error_before_mean": float(before_errors.mean()),
        "reprojection_error_after_mean": float(after_errors.mean()),
        "reprojection_error_delta_max": float(error_delta.max(initial=0.0)),
    }
    (args.output / "principal_point_centering.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"\nSUCCESS: {args.output}")


if __name__ == "__main__":
    main()
