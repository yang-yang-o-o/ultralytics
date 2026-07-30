# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO6D-style PnP / ADD / projection helpers (beer.ply + camera from data yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def get_3d_corners(vertices: np.ndarray) -> np.ndarray:
    """AABB center + 8 corners from mesh vertices, shape (4, 9) homogeneous (YOLO6D order)."""
    # vertices: (4, N) homogeneous or (3, N)
    xyz = vertices[:3]
    min_x, max_x = float(xyz[0].min()), float(xyz[0].max())
    min_y, max_y = float(xyz[1].min()), float(xyz[1].max())
    min_z, max_z = float(xyz[2].min()), float(xyz[2].max())
    corners = np.array(
        [
            [(min_x + max_x) / 2.0, (min_y + max_y) / 2.0, (min_z + max_z) / 2.0],
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ],
        dtype=np.float64,
    )
    return np.concatenate((corners.T, np.ones((1, 9), dtype=np.float64)), axis=0)


def camera_matrix(fx: float, fy: float, u0: float, v0: float) -> np.ndarray:
    """3×3 camera intrinsic matrix."""
    return np.array([[fx, 0.0, u0], [0.0, fy, v0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_mesh_corners(mesh_path: str | Path, scale: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Load mesh vertices (scaled) and 9 control points from PLY, following YOLO6D test.py.

    scale=0.001 converts beer-style mm PLY → meters; BOP IC-BIN already in mm → use scale=1.0.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path))
    verts = np.asarray(mesh.vertices, dtype=np.float64) * scale
    vertices = np.c_[verts, np.ones((len(verts), 1))].T  # (4, N)
    corners3d = get_3d_corners(vertices)  # (4, 9)
    return vertices, corners3d


def pnp(points_3d: np.ndarray, points_2d: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve PnP → R (3×3), t (3×1). points_3d (N,3), points_2d (N,2)."""
    dist = np.zeros((8, 1), dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        np.ascontiguousarray(points_3d, dtype=np.float64),
        np.ascontiguousarray(points_2d.reshape(-1, 1, 2), dtype=np.float64),
        k,
        dist,
    )
    if not ok:
        raise RuntimeError("solvePnP failed")
    r, _ = cv2.Rodrigues(rvec)
    return r, tvec


def project_points(points_3d_h: np.ndarray, rt: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project homogeneous 3D points (4,N) with Rt (3,4) and K → (2,N)."""
    cam = (k @ rt) @ points_3d_h
    return np.vstack((cam[0] / cam[2], cam[1] / cam[2]))


def transform_points(points_3d_h: np.ndarray, rt: np.ndarray) -> np.ndarray:
    """Apply Rt (3,4) to homogeneous points (4,N) → (3,N)."""
    return rt @ points_3d_h


def evaluate_pose_pair(
    corners2d_gt: np.ndarray,
    corners2d_pr: np.ndarray,
    corners3d: np.ndarray,
    vertices: np.ndarray,
    k: np.ndarray,
) -> dict[str, float]:
    """Compute corner, projection, ADD errors for one GT/pred 9-point pair (pixel coords)."""
    pts3 = np.asarray(corners3d[:3].T, dtype=np.float64)  # (9,3)
    r_gt, t_gt = pnp(pts3, corners2d_gt, k)
    r_pr, t_pr = pnp(pts3, corners2d_pr, k)
    rt_gt = np.concatenate((r_gt, t_gt), axis=1)
    rt_pr = np.concatenate((r_pr, t_pr), axis=1)

    corner_err = float(np.linalg.norm(corners2d_gt - corners2d_pr, axis=1).mean())
    proj_gt = project_points(vertices, rt_gt, k)
    proj_pr = project_points(vertices, rt_pr, k)
    proj_err = float(np.linalg.norm(proj_gt - proj_pr, axis=0).mean())
    add = float(np.linalg.norm(transform_points(vertices, rt_gt) - transform_points(vertices, rt_pr), axis=0).mean())
    return {"corner2d": corner_err, "proj2d": proj_err, "add": add}


class Pose6DEval:
    """Accumulate YOLO6D-style metrics from data yaml (mesh + intrinsics).

    Supports single-mesh (beer) or per-class meshes via data['meshes'] / data['diams'].
    """

    def __init__(self, data: dict[str, Any]):
        """Initialize from dataset dict (beer-seg6d / icbin-seg6d yaml fields)."""
        self.k = camera_matrix(float(data["fx"]), float(data["fy"]), float(data["u0"]), float(data["v0"]))
        scale = float(data.get("mesh_scale", 0.001))
        self.default_diam = float(data.get("diam", 0.1))
        self.meshes: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

        meshes_cfg = data.get("meshes")
        diams_cfg = data.get("diams") or {}
        if meshes_cfg:
            for k, path in meshes_cfg.items():
                cid = int(k)
                diam = float(diams_cfg.get(k, diams_cfg.get(cid, self.default_diam)))
                verts, corners = load_mesh_corners(path, scale=scale)
                self.meshes[cid] = (verts, corners, diam)
        else:
            mesh = data.get("mesh")
            if mesh is None:
                raise KeyError("data yaml missing 'mesh' (or 'meshes') for PnP evaluation")
            verts, corners = load_mesh_corners(mesh, scale=scale)
            self.meshes[0] = (verts, corners, self.default_diam)

        self.vertices, self.corners3d, self.diam = self.meshes[min(self.meshes)]
        self.errs_corner: list[float] = []
        self.errs_proj: list[float] = []
        self.errs_add: list[float] = []
        self.errs_diam: list[float] = []

    def update(
        self,
        corners2d_gt: np.ndarray,
        corners2d_pr: np.ndarray,
        cls_id: int | None = None,
    ) -> None:
        """Add one instance evaluation (both arrays shape (9,2) in pixels)."""
        cid = 0 if cls_id is None else int(cls_id)
        if cid not in self.meshes:
            cid = min(self.meshes)
        verts, corners, diam = self.meshes[cid]
        try:
            m = evaluate_pose_pair(corners2d_gt, corners2d_pr, corners, verts, self.k)
        except Exception:
            return
        self.errs_corner.append(m["corner2d"])
        self.errs_proj.append(m["proj2d"])
        self.errs_add.append(m["add"])
        self.errs_diam.append(diam)

    def summary(self) -> dict[str, float]:
        """Return mean errors and Acc@5px / ADD@0.1d (per-instance diameter)."""
        eps = 1e-9
        c = np.asarray(self.errs_corner, dtype=np.float64) if self.errs_corner else np.array([np.nan])
        p = np.asarray(self.errs_proj, dtype=np.float64) if self.errs_proj else np.array([np.nan])
        a = np.asarray(self.errs_add, dtype=np.float64) if self.errs_add else np.array([np.nan])
        d = np.asarray(self.errs_diam, dtype=np.float64) if self.errs_diam else np.array([self.default_diam])
        if self.errs_add:
            add_ok = (a <= d * 0.1).sum() * 100.0 / (len(a) + eps)
        else:
            add_ok = 0.0
        return {
            "mean_corner2d": float(np.nanmean(c)),
            "mean_proj2d": float(np.nanmean(p)),
            "mean_add": float(np.nanmean(a)),
            "acc_proj_5px": float((p <= 5.0).sum() * 100.0 / (len(p) + eps)) if self.errs_proj else 0.0,
            "acc_add_0.1d": float(add_ok),
            "n": float(len(self.errs_proj)),
        }
