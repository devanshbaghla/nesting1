"""
Solid shaded renders of a nested pair, in any of seven canonical views.

Implemented with a painter's algorithm rather than an interactive 3D backend,
because a server has no GPU and matplotlib's mplot3d does not depth-sort
reliably across two interpenetrating bodies — parts of the rear copy leak in
front of the near one, which on a nesting result reads as a broken interlock.

The approach: rotate the geometry into a camera frame, drop the depth axis to
project, draw triangles far-to-near with per-face Lambert shading. Correct
occlusion for free, and it works identically for isometric and orthographic
views since only the camera matrix changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection

VIEWS = ("iso", "top", "bottom", "front", "back", "left", "right")
DOWNLOADABLE = ("top", "bottom", "front")

BODY_COLOURS = ((0.106, 0.620, 0.467), (0.851, 0.373, 0.008))   # teal / orange
_LIGHT = np.array([0.35, 0.45, 1.0])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def camera_matrix(view: str) -> np.ndarray:
    """Rows are the camera's right / up / toward-viewer axes in world space."""
    if view == "top":                 # looking down -Z
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    elif view == "bottom":            # looking up +Z
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    elif view == "front":             # looking along +Y
        R = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)
    elif view == "back":              # looking along -Y
        R = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    elif view == "right":             # looking along -X
        R = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)
    elif view == "left":              # looking along +X
        R = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], float)
    elif view == "iso":
        az, el = np.radians(35.0), np.radians(24.0)
        toward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                           np.sin(el)])
        right = np.cross([0, 0, 1.0], toward)
        right /= np.linalg.norm(right)
        up = np.cross(toward, right)
        R = np.array([right, up, toward])
    else:
        raise ValueError(f"unknown view {view!r}; choose from {VIEWS}")
    if np.linalg.det(R) < 0:          # keep it a rotation, not a reflection
        R[0] *= -1
    return R


def _bodies(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    try:
        parts = mesh.split(only_watertight=False)
    except Exception:
        parts = []
    return list(parts) if len(parts) >= 2 else [mesh]


def render(stl_path: str | Path, view: str, out_path: str | Path,
           dpi: int = 110, figsize: float = 6.0, label: str = "",
           show_bbox: bool = True) -> str:
    """Render one view of an STL to PNG. Returns the output path."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    parts = _bodies(mesh)
    R = camera_matrix(view)

    tris, cols, depth = [], [], []
    for i, p in enumerate(parts):
        base = np.array(BODY_COLOURS[i % len(BODY_COLOURS)])
        T = p.triangles @ R.T                       # (F, 3, 3) camera frame
        N = p.face_normals @ R.T
        shade = 0.32 + 0.68 * np.clip(N @ _LIGHT, 0.0, 1.0)
        tris.append(T[:, :, :2])
        cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
        depth.append(T[:, :, 2].mean(axis=1))

    tris = np.concatenate(tris)
    cols = np.concatenate(cols)
    depth = np.concatenate(depth)
    order = np.argsort(depth)                       # far first, near last

    lo = tris.reshape(-1, 2).min(axis=0)
    hi = tris.reshape(-1, 2).max(axis=0)
    span = (hi - lo).max()
    pad = 0.08 * span
    mid = (hi + lo) / 2

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.add_collection(PolyCollection(tris[order], facecolors=cols[order],
                                     edgecolors="none", antialiased=False))
    if show_bbox and view != "iso":
        ax.add_patch(plt.Rectangle(tuple(lo), *(hi - lo), fill=False,
                                   ec="#444", lw=1.0, ls="--", alpha=0.8))
        ax.text(mid[0], lo[1] - pad * 0.55, f"{hi[0]-lo[0]:.1f} mm",
                ha="center", va="top", fontsize=9, color="#444")
        ax.text(lo[0] - pad * 0.55, mid[1], f"{hi[1]-lo[1]:.1f} mm",
                ha="right", va="center", rotation=90, fontsize=9, color="#444")
    ax.set_xlim(mid[0] - span / 2 - pad, mid[0] + span / 2 + pad)
    ax.set_ylim(mid[1] - span / 2 - pad, mid[1] + span / 2 + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(label or view.upper(), fontsize=11, pad=8)
    fig.tight_layout(pad=0.4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=False, facecolor="white")
    plt.close(fig)
    return str(out_path)


def render_sheet(stl_path: str | Path, out_path: str | Path,
                 views=DOWNLOADABLE, dpi: int = 110, label: str = "") -> str:
    """One PNG holding several views side by side."""
    mesh = trimesh.load(str(stl_path), force="mesh")
    parts = _bodies(mesh)
    n = len(views)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6), squeeze=False)
    for ax, view in zip(axes[0], views):
        R = camera_matrix(view)
        tris, cols, depth = [], [], []
        for i, p in enumerate(parts):
            base = np.array(BODY_COLOURS[i % len(BODY_COLOURS)])
            T = p.triangles @ R.T
            N = p.face_normals @ R.T
            shade = 0.32 + 0.68 * np.clip(N @ _LIGHT, 0.0, 1.0)
            tris.append(T[:, :, :2])
            cols.append(np.clip(base[None, :] * shade[:, None], 0, 1))
            depth.append(T[:, :, 2].mean(axis=1))
        tris, cols = np.concatenate(tris), np.concatenate(cols)
        order = np.argsort(np.concatenate(depth))
        ax.add_collection(PolyCollection(tris[order], facecolors=cols[order],
                                         edgecolors="none", antialiased=False))
        lo = tris.reshape(-1, 2).min(axis=0)
        hi = tris.reshape(-1, 2).max(axis=0)
        span = (hi - lo).max(); pad = 0.09 * span; mid = (hi + lo) / 2
        ax.add_patch(plt.Rectangle(tuple(lo), *(hi - lo), fill=False,
                                   ec="#444", lw=1.0, ls="--", alpha=0.8))
        ax.set_xlim(mid[0] - span / 2 - pad, mid[0] + span / 2 + pad)
        ax.set_ylim(mid[1] - span / 2 - pad, mid[1] + span / 2 + pad)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{view.upper()}   {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} mm",
                     fontsize=10)
    if label:
        fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return str(out_path)
