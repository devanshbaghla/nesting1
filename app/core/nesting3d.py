"""
nesting3d.py — 3D nesting of two rigid copies of a mesh under a minimum
surface-to-surface clearance constraint.

This module packages the full algorithm stack:

    MeshAudit           intake checks: watertight, AABB vs OBB, fill ratio, silhouettes
    ScanlineVoxelizer   exact solid voxelisation by z-scanline parity fill
    ClearanceGrid       Euclidean dilation of the fixed part by the clearance
    TranslationOracle   FFT correlation -> globally optimal translation per rotation
    OrientationSet      rotation candidate generators (Z-family, full SO(3))
    SurfacePairDistance dense sampling + KD-tree pruning + exact point-triangle
    Refiner             continuous constrained refinement (profile sweep, descent,
                        bisection, perturbation check)
    Geometry            AABB union, axis leverage, min-area rectangle (calipers)
    Validation          self-tests that gate the result (analytic + cross-checks)
    Preview             orthographic + isometric render of the nested pair
    PairNester          orchestrator: audit -> sweep -> refine -> export -> verify

Which stage decides what
------------------------
The lattice search picks the ROTATION and lands within ~1 mm of the final
translation. The continuous refinement decides the actual delivered pose: the
profile sweep found a sharp feasibility cliff (the interlock either engages or
it does not) that a pure gradient-style descent can walk straight past.

Method
------
For a *fixed* relative rotation, the set of collision-free translations is
obtained by correlating a clearance-dilated occupancy grid of part A with the
occupancy grid of part B. Every lattice translation is evaluated in one FFT, so
the translation optimum is GLOBAL for that rotation. Rotations are then swept.
The lattice solution is finally polished in continuous space against an exact
mesh-to-mesh distance so the clearance is met to ~1e-3 mm.

Complexity per rotation: O(N log N) in the number of grid cells, versus O(N)
per candidate translation for naive collision testing.

Requires: numpy, scipy, trimesh.

Pitfalls this code already handles (each cost a debugging cycle):
  * Surface voxelisation + flood fill inflates thin walls badly (measured +80%
    on a 3 mm wall at 1 mm pitch). Parity scanline on voxel centres is used.
  * Scanline parity breaks when sample rays hit shared edges/vertices exactly,
    which happens constantly on CAD models with integer-coordinate features.
    Fixed by an irrational sub-micron jitter, applied identically everywhere so
    the lattice stays shared between rotations.
  * The jitter must also widen the triangle->column candidate window, or
    boundary columns get dropped and parity breaks again (silently, producing
    +150% volume).
  * Voxel dilation measures centre-to-centre, so a dilation radius of exactly
    `clearance` yields a true gap up to ~1.5*pitch SMALLER. Search with
    `clearance + 1.5*pitch`, then squeeze in continuous space.
  * Radius-based triangle lookup for exact distance explodes on meshes with a
    few huge triangles (this part: max circumradius 63 mm on a 28 mm part).
    Use k-nearest-sample-point -> owning-face lookup instead.
  * A bisection whose upper bracket is itself infeasible returns the bracket
    end, silently. On the first profile sweep this produced ten plausible-
    looking rows that were all wrong. Always assert feasibility at `hi`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

from .mesh_repair import MeshRepair, MeshRepairError, RepairReport

try:                                    # optional: enables the 'bvh' backend
    import fcl
except ImportError:                     # pragma: no cover - absence is normal
    fcl = None

__all__ = [
    "MeshAudit", "ScanlineVoxelizer", "ClearanceGrid", "TranslationOracle",
    "OrientationSet", "SurfaceDistanceField", "TransformedDistanceField",
    "SurfacePairDistance",
    "BVHPairDistance",
    "SurfaceSampleCache", "Refiner", "Geometry", "Validation", "Preview",
    "PairNester", "NestResult", "MeshRepair", "MeshRepairError",
    "RepairReport", "KD_WORKERS", "HAVE_FCL",
]

#: Whether the BVH distance backend can be selected on this install.
HAVE_FCL = fcl is not None

#: Cores used for ``cKDTree.query``. One: the engine runs single-threaded.
#:
#: This is a deliberate trade, not a free win. SciPy will fan a query across
#: every core, and doing so is measurably faster — 24.7 s against 39.1 s on the
#: reference part's ``quick`` profile, so serial costs about 1.6x. Results are
#: identical either way; the fan-out was never a correctness question.
#:
#: It is off because a single-threaded engine is worth more than the 1.6x here.
#: Under a profiler the fan-out charges its time to ``_thread.lock.acquire``
#: rather than to the function that caused it, which put 88% of a profile
#: somewhere unattributable and hid where the work actually was — every
#: optimisation in this module was found only after turning it off. It also
#: means one job now uses one core, so ``NEST_JOB_WORKERS`` scales cleanly
#: instead of having each job fight the others for the same cores.
#:
#: Kept as a named constant rather than inlined so every query site still says
#: out loud that it is serial, and so restoring the fan-out is a one-line
#: change here rather than an edit to eleven call sites.
KD_WORKERS = 1

#: Neighbouring samples whose owning faces get an exact point-triangle test.
#:
#: The claim being hedged is statistical: area-weighted sampling puts a sample
#: within about one spacing of the true closest point, so the face that owns it
#: is very likely among the k nearest. k was 24 as a safety margin, chosen
#: without measurement.
#:
#: Measured instead, over six parts at three poses each -- clear, near contact,
#: and interpenetrating -- k=8 returns a distance identical to k=32 in all
#: eighteen cases, while k=6 and below deviate by up to 7.2e-05 mm. Eight is
#: therefore the smallest value that is not merely close but exact on this
#: geometry, and it costs a third of what 24 did: _pt_tri went 0.0699s -> 0.0234s
#: on the reference part, against 12.05% of all profiled time spent in _pt_tri
#: and point_triangle_distance together.
#:
#: Raise it if a part ever disagrees with `exact_reference`, which is the
#: independent check that would catch it.
PT_TRI_K = 8


# --------------------------------------------------------------------------- #
#  Algorithm switchboard
# --------------------------------------------------------------------------- #
#: Algorithms switched off, with the reason. Nothing is deleted: the code stays
#: in the module, keeps its tests, and is re-enabled by removing an entry here.
#:
#: The shortlist that drove this was ranked on *accuracy* and *cost*. Neither
#: axis says anything about whether an algorithm can be switched off, so the
#: call graph was checked per candidate and five of the ten could not be. They
#: are recorded in ``LOAD_BEARING`` below rather than silently kept, so the
#: disagreement between the shortlist and the code is visible.
DISABLED_ALGORITHMS: dict[str, str] = {
    "slice_profile":
        "planar cross-section profile: unreachable already -- Inspector is "
        "referenced nowhere outside its own module, and it recorded 0 calls "
        "across 23 profiled runs.",
    "pair_nester_verify":
        "connected-component body split check: superseded by "
        "NestingRecommender.verify_one, which is shortlist #7. 0 calls across "
        "23 profiled runs.",
    "so3":
        "ZXZ Euler SO(3) grid sweep: only the 'full' profile requests it, so "
        "disabling it makes 'full' behave as 'standard'. The rotation is then "
        "assumed from the Z-family rather than proven over SO(3).",
    "descend":
        "coordinate descent with step halving: replaced by the profile sweep, "
        "shortlist #9, which is what the accuracy ranking preferred -- descent "
        "can settle on the wrong side of a feasibility discontinuity. The "
        "'quick' profile therefore refines with 'profile' instead, which is "
        "markedly slower: measured 3,059 KD queries against 61 on one part.",
}

#: Candidates from the same shortlist exercise that CANNOT be switched off,
#: because a shortlisted algorithm calls them directly. Kept as a record of
#: why, with the call site that proves it.
LOAD_BEARING: dict[str, str] = {
    "outward feasibility repair":
        "switched off on the evidence of 0 calls across 23 profiled runs, then "
        "switched back on within one change. Choosing a coarser lattice made "
        "the lattice search hand the refiner an infeasible start on "
        "electric_drill.stl, which is precisely the case Refiner.repair exists "
        "for, and the run died with AlgorithmDisabled. 'Never observed' is not "
        "'unreachable' -- it only meant nothing had yet exercised the path.",
    "EDT dilation of free space":
        "TranslationOracle.search reads ClearanceGrid.grid and correlates it "
        "(nesting3d.py, 'A = self.cg.grid' then fftconvolve). Shortlist #3 has "
        "nothing to correlate without it, and the clearance constraint the "
        "service guarantees is exactly that grid.",
    "slice-wise masked argmin":
        "TranslationOracle.search calls _argmin twice, and it is the only "
        "place a translation is extracted from the free-translation map. "
        "Without it shortlist #3 computes the map and no pose is ever chosen.",
    "area-weighted barycentric sampling":
        "SurfacePairDistance.__init__ builds self.pA/self.pB from _sample. "
        "Shortlist #1, #2 and #6 all operate on those point clouds.",
    "KD-tree point-to-point minimum":
        "shortlist #2 is not a separate algorithm from this one -- the "
        "translation-bounded prune is the body of min_distance_to. Disabling "
        "the query removes the prune with it.",
    "concatenate + origin-corner shift":
        "NestingRecommender._assembly is what export_one writes as the STL and "
        "the GLB. Without it there is no final nested object to return, which "
        "is the output being asked for.",
}


class AlgorithmDisabled(RuntimeError):
    """Raised when a switched-off algorithm is invoked."""


def _refuse(key: str):
    """Raise for a disabled algorithm, quoting the recorded reason."""
    raise AlgorithmDisabled(
        f"{key!r} is switched off in DISABLED_ALGORITHMS: "
        f"{DISABLED_ALGORITHMS[key]}")


def algorithm_enabled(key: str) -> bool:
    return key not in DISABLED_ALGORITHMS


# --------------------------------------------------------------------------- #
#  Intake
# --------------------------------------------------------------------------- #
class MeshAudit:
    """Intake checks run before any nesting, because they decide the approach.

    Three questions get answered here:

    1. Is the mesh watertight with consistent winding? Parity voxelisation is
       only defined for a closed surface.
    2. Is the supplied orientation already the tight one? If the axis-aligned
       box beats the fitted oriented box, the part arrives aligned and the
       reference copy can stay at identity, which halves the search space.
    3. How much air is in the bounding box? A low fill ratio is what makes
       nesting worth attempting at all — a solid brick cannot interlock.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh

    def report(self) -> dict:
        m = self.mesh
        aabb = float(np.prod(m.extents))
        obb = m.bounding_box_oriented
        return {
            "faces": int(len(m.faces)),
            "watertight": bool(m.is_watertight),
            "winding_consistent": bool(m.is_winding_consistent),
            "bodies": int(m.body_count),
            "extents": m.extents.tolist(),
            "volume": float(m.volume),
            "area": float(m.area),
            "aabb_volume": aabb,
            "obb_extents": obb.extents.tolist(),
            "obb_volume": float(obb.volume),
            "already_axis_aligned": bool(aabb <= obb.volume),
            "fill_ratio": float(m.volume / aabb),
            "max_triangle_circumradius": float(np.linalg.norm(
                m.triangles - m.triangles.mean(1)[:, None], axis=2).max()),
        }

    def silhouette_areas(self) -> dict:
        """Convex-hull area of the projected outline on each plane.

        Caveat worth stating plainly: a convex hull FILLS every pocket, so this
        number alone cannot reveal a cavity and did not find the one on the
        reference part. Use it only as a coarse aspect-ratio read, and use
        :meth:`Preview.part_glb` to actually see concavity.
        """
        from scipy.spatial import ConvexHull
        out = {}
        for a, b, name in ((1, 2, "yz"), (0, 2, "xz"), (0, 1, "xy")):
            pts = self.mesh.vertices[:, [a, b]]
            out[name] = float(ConvexHull(pts).volume)   # 2D hull "volume" = area
        return out

    def assert_usable(self, repair: bool = False):
        """Gate the intake, optionally closing an open mesh first.

        ``repair=True`` runs :meth:`MeshRepair.ensure_solid` and rebinds
        ``self.mesh`` to the repaired copy, so the report describes what will
        actually be nested rather than what arrived.
        """
        if repair:
            self.mesh, _ = MeshRepair.ensure_solid(self.mesh)
        r = self.report()
        if not (r["watertight"] and r["winding_consistent"]):
            raise MeshRepairError(MeshRepair.describe_defect(self.mesh))
        return r


# --------------------------------------------------------------------------- #
#  Voxelisation
# --------------------------------------------------------------------------- #
class ScanlineVoxelizer:
    """Exact solid voxelisation of a watertight mesh by z-scanline parity fill.

    Every mesh is rasterised on the SAME global lattice: voxel centres sit at
    ``(i + 0.5) * pitch`` in world coordinates. Consequently a relative
    translation between two rasterised meshes is an exact integer multiple of
    ``pitch``, which is what makes the FFT translation search exact.

    A voxel is occupied iff its centre lies inside the mesh. Accuracy on a
    real part: +0.18% volume error at 0.5 mm pitch, max 0.88% over 40 random
    orientations.

    :meth:`rasterize` assumes a closed surface and does not check — on an open
    mesh the parity fill runs past the boundary and returns a wrong solid with
    no error. Use :meth:`solid` at any entry point where the mesh has not
    already been through :class:`MeshRepair`.
    """

    #: irrational jitter (mm-scale) keeping sample rays off edges and vertices
    JX = 1.4142135e-4
    JY = 1.7320508e-4

    #: candidate columns expanded at once. Bounds peak memory on meshes
    #: with a few oversized triangles, where one window can be most of
    #: the lattice; 4M columns is about 100 MB of working arrays.
    RASTER_CHUNK = 4_000_000

    @classmethod
    def rasterize(cls, mesh: trimesh.Trimesh, pitch: float):
        """Return ``(occ, i0)``.

        ``occ[i, j, k]`` is True when the voxel centre at
        ``(i0 + [i, j, k] + 0.5) * pitch`` is inside the mesh.
        """
        tris = np.asarray(mesh.triangles, dtype=np.float64)
        lo, hi = mesh.bounds
        i0 = np.floor(lo / pitch).astype(np.int64) - 1
        i1 = np.ceil(hi / pitch).astype(np.int64) + 1
        nx, ny, nz = (i1 - i0).astype(int)

        xs = (np.arange(i0[0], i1[0]) + 0.5) * pitch + cls.JX
        ys = (np.arange(i0[1], i1[1]) + 0.5) * pitch + cls.JY

        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        e1, e2 = v1 - v0, v2 - v0
        denom = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        keep = np.abs(denom) > 1e-12            # skip edge-on triangles
        v0, e1, e2, denom = v0[keep], e1[keep], e2[keep], denom[keep]

        tlo = np.minimum(np.minimum(v0, v0 + e1), v0 + e2)
        thi = np.maximum(np.maximum(v0, v0 + e1), v0 + e2)
        # widen by one cell: the barycentric test is authoritative, but a
        # dropped candidate column silently destroys the parity fill
        ixlo = np.clip(np.floor((tlo[:, 0] - cls.JX) / pitch - 0.5).astype(np.int64) - 1 - i0[0], 0, nx - 1)
        ixhi = np.clip(np.ceil((thi[:, 0] - cls.JX) / pitch - 0.5).astype(np.int64) + 1 - i0[0], 0, nx - 1)
        iylo = np.clip(np.floor((tlo[:, 1] - cls.JY) / pitch - 0.5).astype(np.int64) - 1 - i0[1], 0, ny - 1)
        iyhi = np.clip(np.ceil((thi[:, 1] - cls.JY) / pitch - 0.5).astype(np.int64) + 1 - i0[1], 0, ny - 1)

        # Every triangle's candidate window, expanded and tested in one pass
        # rather than one numpy call per triangle. The loop this replaces was
        # 90% of rasterize (2.490 s of 2.768 s measured), and on a 1.5M-face
        # part it ran 1.5 million times per call.
        #
        # The barycentric test below is unchanged and still authoritative; all
        # that changes is that the (triangle, column) pairs are enumerated as
        # one flat array. Chunked over triangles so peak memory stays bounded
        # on meshes with a few huge triangles, where one window can cover a
        # large slice of the lattice on its own.
        wx = (ixhi - ixlo + 1)
        wy = (iyhi - iylo + 1)
        live = np.flatnonzero((wx > 0) & (wy > 0))
        cols, zhit = [], []
        if len(live):
            counts_all = (wx[live] * wy[live]).astype(np.int64)
            # walk triangles in chunks of roughly CHUNK candidate columns
            edges, run = [0], 0
            for i, c in enumerate(counts_all):
                run += int(c)
                if run >= cls.RASTER_CHUNK:
                    edges.append(i + 1); run = 0
            if edges[-1] != len(live):
                edges.append(len(live))
            for lo_i, hi_i in zip(edges[:-1], edges[1:]):
                sel = live[lo_i:hi_i]
                cnt = (wx[sel] * wy[sel]).astype(np.int64)
                tri = np.repeat(sel, cnt)
                # offset of each candidate within its own triangle's window
                starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
                off = np.arange(int(cnt.sum())) - np.repeat(starts, cnt)
                wyr = np.repeat(wy[sel], cnt)
                ix = ixlo[tri] + off // wyr
                iy = iylo[tri] + off % wyr
                px = xs[ix] - v0[tri, 0]
                py = ys[iy] - v0[tri, 1]
                dn = denom[tri]
                u = (px * e2[tri, 1] - py * e2[tri, 0]) / dn
                v = (py * e1[tri, 0] - px * e1[tri, 1]) / dn
                inside = (u >= 0) & (v >= 0) & (u + v <= 1)
                if not inside.any():
                    continue
                tri, u, v = tri[inside], u[inside], v[inside]
                cols.append(ix[inside] * ny + iy[inside])
                zhit.append(v0[tri, 2] + u * e1[tri, 2] + v * e2[tri, 2])

        # int16, not int32: the array holds a difference count, and its
        # magnitude is bounded by how many spans start or end in one cell --
        # single digits on real geometry. It halves the largest allocation the
        # voxeliser makes, which is what decides whether a big part fits in
        # memory at all. Measured bit-identical, and 15% off the call.
        acc = np.zeros((nx * ny, nz + 1), dtype=np.int16)
        if cols:
            col = np.concatenate(cols)
            z = np.concatenate(zhit)
            order = np.lexsort((z, col))
            col, z = col[order], z[order]

            # drop tangential double-hits in pairs (preserves parity)
            if len(col) > 1:
                same = (col[1:] == col[:-1]) & (np.abs(z[1:] - z[:-1]) < 1e-9)
                dup = np.zeros(len(col), bool)
                dup[:-1] |= same
                dup[1:] |= same
                col, z = col[~dup], z[~dup]

            uniq = np.unique(col)
            starts = np.repeat(np.searchsorted(col, uniq),
                               np.bincount(col, minlength=nx * ny)[uniq])
            rank = np.arange(len(col)) - starts
            ent = (rank % 2) == 0
            c_in, z_in, z_out = col[ent], z[ent], z[~ent]
            m = min(len(z_in), len(z_out))
            c_in, z_in, z_out = c_in[:m], z_in[:m], z_out[:m]

            k0 = np.clip(np.ceil(z_in / pitch - 0.5).astype(np.int64) - i0[2], 0, nz)
            k1 = np.clip(np.floor(z_out / pitch - 0.5).astype(np.int64) - i0[2], -1, nz - 1)
            ok = k1 >= k0
            np.add.at(acc, (c_in[ok], k0[ok]), 1)          # difference array
            np.add.at(acc, (c_in[ok], k1[ok] + 1), -1)

        # accumulate in place: np.cumsum(acc) would allocate a second array
        # the size of the lattice before `> 0` collapses it to a bool
        np.cumsum(acc, axis=1, out=acc)
        occ = acc[:, :nz] > 0
        return occ.reshape(nx, ny, nz), i0

    @classmethod
    def solid(cls, mesh: trimesh.Trimesh, pitch: float, repair: bool = True,
              log=None) -> tuple[np.ndarray, np.ndarray, trimesh.Trimesh]:
        """Rasterise, closing the mesh first if it is open.

        Returns ``(occ, i0, mesh)``. The third element is the mesh the grid was
        actually built from — the repaired copy when a repair happened — and
        callers **must** adopt it. Bounds shift when holes are filled, and a
        grid built from one mesh with translations measured against another is
        wrong in a way nothing downstream would notice.

        :raises MeshRepairError: the mesh is open and cannot be closed safely.
        """
        mesh, _ = MeshRepair.ensure_solid(mesh, allow_repair=repair, log=log)
        occ, i0 = cls.rasterize(mesh, pitch)
        return occ, i0, mesh

    @staticmethod
    def volume_error(mesh: trimesh.Trimesh, pitch: float) -> float:
        """Relative volume error of the voxelisation — a cheap self-test."""
        occ, _ = ScanlineVoxelizer.rasterize(mesh, pitch)
        return occ.sum() * pitch ** 3 / mesh.volume - 1.0

    @staticmethod
    def rotation_robustness(mesh: trimesh.Trimesh, pitch: float,
                            trials: int = 40, seed: int = 0) -> dict:
        """Volume error over random orientations — the test that matters.

        A single axis-aligned check passes even when parity is broken, because
        the failure is triggered by rays grazing edges, and the sweep rotates
        the part into exactly those configurations. This is the gate that
        caught the jitter and candidate-window bugs.
        """
        rng = np.random.default_rng(seed)
        errs = []
        for _ in range(trials):
            T = trimesh.transformations.random_rotation_matrix(rng.random(3))
            m = mesh.copy(); m.apply_transform(T)
            occ, _ = ScanlineVoxelizer.rasterize(m, pitch)
            errs.append(occ.sum() * pitch ** 3 / mesh.volume - 1.0)
        errs = np.array(errs)
        return {"mean": float(errs.mean()), "max_abs": float(np.abs(errs).max()),
                "trials": trials}


# --------------------------------------------------------------------------- #
#  Distance field
# --------------------------------------------------------------------------- #
class SurfaceDistanceField:
    """Distance from any point to a fixed part's surface, read off a lattice.

    :class:`ClearanceGrid` already runs a Euclidean distance transform of part
    A and then throws the distances away, keeping only ``dt <= radius`` as a
    boolean. This holds on to them. The array is the expensive thing to
    produce and it is produced anyway, so retaining it costs memory and no
    time: 2.7 MB at 1 mm pitch on the reference part, 21 MB at 0.5 mm.

    What it buys is a lower bound on the surface distance for a whole point
    cloud in one gather, where :class:`scipy.spatial.cKDTree` needs a tree
    descent per point. That is not accurate enough to *answer* a clearance
    query, but it is accurate enough to rule most points out of one — see
    :meth:`SurfacePairDistance.min_distance_to`, which uses it to shrink the
    KD query rather than to replace it, so the answer is unchanged.

    Two properties make the bound safe to rely on:

    * The value is conservative. ``tolerance`` is the whole voxel diagonal,
      twice the half-diagonal a lattice can actually be wrong by, so
      ``lower_bound`` sits below the true surface distance with margin.
    * Reading outside the array clamps to the edge (``mode="nearest"``) rather
      than failing. An edge voxel is nearer the part than anything beyond it,
      so a clamped read still under-states the distance. Queries far outside
      the grid stay correct; they just stop being selective.

    Samples of the part lie *on* its surface, so the distance to the sample
    cloud that :class:`SurfacePairDistance` measures is never smaller than the
    distance to the surface. The bound therefore holds for that metric too,
    which is the one the refiner gates on.
    """

    def __init__(self, distance: np.ndarray, origin: np.ndarray, pitch: float):
        self.distance = np.asarray(distance, dtype=np.float32)
        self.origin = np.asarray(origin, dtype=np.int64)
        self.pitch = float(pitch)
        #: how far below the true distance :meth:`lower_bound` may sit
        self.tolerance = float(np.sqrt(3.0) * pitch)

    @property
    def nbytes(self) -> int:
        return int(self.distance.nbytes)

    @classmethod
    def build(cls, occupancy: np.ndarray, i0: np.ndarray, pitch: float,
              pad: int) -> "SurfaceDistanceField":
        """Distance transform of a padded occupancy grid, kept as floats."""
        return cls.build_with_mask(occupancy, i0, pitch, pad, None)[0]

    @classmethod
    def build_with_mask(cls, occupancy: np.ndarray, i0: np.ndarray,
                        pitch: float, pad: int, radius: float | None):
        """Return ``(field, distance <= radius)``, or ``(field, None)``.

        The threshold is taken at the transform's own precision and only then
        is the array narrowed to float32 for storage. Doing it the other way
        round would let a value within a rounding step of ``radius`` land on
        the wrong side, which would silently change the feasible set the sweep
        searches — a different answer, to save four bytes a voxel.
        """
        padded = np.zeros(np.asarray(occupancy.shape) + 2 * pad, bool)
        padded[pad:-pad, pad:-pad, pad:-pad] = occupancy
        dt = distance_transform_edt(~padded, sampling=pitch)
        mask = None if radius is None else (dt <= radius)
        return cls(dt, np.asarray(i0) - pad, pitch), mask

    def transformed(self, matrix: np.ndarray) -> "TransformedDistanceField":
        """A view of this field describing the part moved by ``matrix``.

        A rigid motion preserves distance, so the field of a moved copy is the
        original field read at the pre-image of the query point. That makes one
        transform enough to serve every placement of the same part — the array
        is shared, not copied, so a view costs nothing but the 4x4.
        """
        return TransformedDistanceField(self, matrix)

    def values(self, points: np.ndarray) -> np.ndarray:
        """Interpolated surface distance at each point, in file units."""
        idx = (np.asarray(points, float) / self.pitch - 0.5) - self.origin[None, :]
        return map_coordinates(self.distance, idx.T, order=1, mode="nearest")

    def lower_bound(self, points: np.ndarray) -> np.ndarray:
        """Values floored by the lattice tolerance; never above the truth."""
        return np.maximum(self.values(points) - self.tolerance, 0.0)


class TransformedDistanceField:
    """One part's distance field, read as though the part had been moved.

    Nesting places the same part twice, and verification re-reads both copies
    from disk. Rasterising each placement separately would mean a distance
    transform per copy per recommendation; a rigid motion preserves distance,
    so mapping the query back through the inverse gives the same answer from
    the array that already exists.

    Only rigid motions are admissible. A scale or a shear would stretch
    distance and quietly break the bound every caller relies on, so the matrix
    is checked once at construction rather than trusted.
    """

    def __init__(self, field: SurfaceDistanceField, matrix: np.ndarray):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4, 4):
            raise ValueError(f"expected a 4x4 transform, got {matrix.shape}")
        R = matrix[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-8):
            raise ValueError("transform is not rigid; distance would not be "
                             "preserved and the bound would be invalid")
        self.field = field
        self.matrix = matrix
        self.inverse = np.linalg.inv(matrix)
        self.pitch = field.pitch
        self.tolerance = field.tolerance

    @property
    def nbytes(self) -> int:
        return 0                      # a view; the array belongs to the field

    def values(self, points: np.ndarray) -> np.ndarray:
        return self.field.values(
            trimesh.transform_points(np.asarray(points, float), self.inverse))

    def lower_bound(self, points: np.ndarray) -> np.ndarray:
        return np.maximum(self.values(points) - self.tolerance, 0.0)


# --------------------------------------------------------------------------- #
#  Clearance dilation
# --------------------------------------------------------------------------- #
class ClearanceGrid:
    """Occupancy of the fixed part A, Euclidean-dilated by the clearance.

    B collides with the dilated A exactly when some occupied voxel centre of B
    lies within ``radius`` of an occupied voxel centre of A, i.e. when the
    centre-to-centre gap is below ``radius``.

    Because the criterion is centre-based while the real constraint is
    surface-based, the true surface gap can be up to ~1.5*pitch smaller than
    ``radius``. Use :meth:`safe_radius` for a guaranteed-conservative value and
    recover the slack afterwards with :class:`Refiner`.
    """

    def __init__(self, mesh: trimesh.Trimesh, pitch: float, radius: float,
                 repair: bool = True):
        # `solid` may hand back a repaired copy; bind that one, because
        # TranslationOracle measures translations against self.mesh.bounds and
        # those must describe the geometry the grid was built from
        occ, i0, mesh = ScanlineVoxelizer.solid(mesh, pitch, repair=repair)
        self.mesh, self.pitch, self.radius = mesh, pitch, radius
        pad = int(np.ceil(radius / pitch)) + 2
        # the distance transform is kept rather than thrown away after the
        # threshold: the dilation below is one comparison against it, and the
        # refiner reuses the same array to prune its KD queries. `grid` and
        # `origin` are what they always were, so the sweep is unaffected.
        self.field, self.grid = SurfaceDistanceField.build_with_mask(
            occ, i0, pitch, pad, radius)
        self.origin = self.field.origin
        pool = SurfaceSampleCache.current()
        if pool is not None:
            pool.publish_field(mesh, pitch, self.field)

    @staticmethod
    def safe_radius(clearance: float, pitch: float) -> float:
        """Dilation radius that guarantees a true surface gap >= ``clearance``."""
        return clearance + 1.5 * pitch


# --------------------------------------------------------------------------- #
#  Global translation search
# --------------------------------------------------------------------------- #
@dataclass
class TranslationResult:
    volume: float
    t_volume: np.ndarray
    area: float
    t_area: np.ndarray


class TranslationOracle:
    """Exhaustive, exact translation search for one fixed relative rotation.

    ``free = (A_dilated correlate B) < 0.5`` marks every lattice translation at
    which the parts do not violate the clearance. The objective (AABB of the
    union) is separable per axis, so it is assembled from three 1-D length
    arrays rather than evaluated per candidate.
    """

    def __init__(self, clearance_grid: ClearanceGrid):
        self.cg = clearance_grid
        self.meshA = clearance_grid.mesh
        self.pitch = clearance_grid.pitch

    def search(self, meshB: trimesh.Trimesh, window: float | None = None,
               centre: Sequence[float] | None = None) -> TranslationResult | None:
        """Best translation of ``meshB`` (already rotated) against the fixed A.

        ``window`` optionally restricts the search to translations within
        +/-window mm of ``centre`` (used for fine local refinement).
        """
        pitch = self.pitch
        B, iB = ScanlineVoxelizer.rasterize(meshB, pitch)
        if not B.any():
            return None
        A = self.cg.grid
        corr = fftconvolve(A.astype(np.float32),
                           B[::-1, ::-1, ::-1].astype(np.float32), mode="full")
        free = corr < 0.5
        del corr

        nb = np.array(B.shape)
        bA, bB = self.meshA.bounds, meshB.bounds
        lengths, offsets = [], []
        for ax in range(3):
            k = np.arange(free.shape[ax]) - (nb[ax] - 1)
            t = (self.cg.origin[ax] + k - iB[ax]) * pitch
            lo = np.minimum(bA[0][ax], bB[0][ax] + t)
            hi = np.maximum(bA[1][ax], bB[1][ax] + t)
            lengths.append((hi - lo).astype(np.float32))
            offsets.append(t)
            if window is not None:
                shape = [-1 if i == ax else 1 for i in range(3)]
                free &= (np.abs(t - centre[ax]) <= window).reshape(shape)

        if not free.any():
            return None
        Lx, Ly, Lz = lengths
        iv = self._argmin(free, lambda i: Lx[i] * Ly[:, None] * Lz[None, :])
        ia = self._argmin(free, lambda i: np.broadcast_to(
            (Lx[i] * Ly)[:, None], (len(Ly), len(Lz))))
        tv = np.array([offsets[a][iv[a]] for a in range(3)])
        ta = np.array([offsets[a][ia[a]] for a in range(3)])
        vol = float(Lx[iv[0]] * Ly[iv[1]] * Lz[iv[2]])
        area = float(Lx[ia[0]] * Ly[ia[1]])
        return TranslationResult(vol, tv, area, ta)

    @staticmethod
    def _argmin(free: np.ndarray, slice_obj: Callable[[int], np.ndarray]):
        """Masked argmin, evaluated slice-by-slice to bound peak memory."""
        best, best_idx = np.inf, (0, 0, 0)
        for i in range(free.shape[0]):
            fi = free[i]
            if not fi.any():
                continue
            vals = np.where(fi, slice_obj(i), np.inf)
            j, k = np.unravel_index(np.argmin(vals), vals.shape)
            if vals[j, k] < best:
                best, best_idx = float(vals[j, k]), (i, int(j), int(k))
        return best_idx


# --------------------------------------------------------------------------- #
#  Rotation candidates
# --------------------------------------------------------------------------- #
class OrientationSet:
    """Generators of candidate relative rotations, as ``(label, 4x4)`` pairs."""

    @staticmethod
    def Rz(deg: float) -> np.ndarray:
        return trimesh.transformations.rotation_matrix(np.radians(deg), [0, 0, 1])

    @staticmethod
    def Rx(deg: float) -> np.ndarray:
        return trimesh.transformations.rotation_matrix(np.radians(deg), [1, 0, 0])

    @classmethod
    def z_family(cls, step: float = 5.0):
        """Spin about Z, with and without a 180 deg flip about X.

        Appropriate when the part is slender: any tilt off the long axis
        inflates the bounding box far more than an interlock can recover.
        """
        out = []
        for flip in (False, True):
            for d in np.arange(0.0, 360.0, step):
                T = cls.Rx(180) @ cls.Rz(d) if flip else cls.Rz(d)
                out.append((f"{'flipX.' if flip else ''}Rz{d:+.1f}", T))
        return out

    @classmethod
    def so3(cls, step: float = 30.0):
        """ZXZ Euler grid covering all of SO(3) — the unbiased sweep."""
        if not algorithm_enabled('so3'):
            _refuse('so3')
        out = []
        for al in np.arange(0.0, 180.01, step):
            z1s = [0.0] if al in (0.0, 180.0) else np.arange(0.0, 360.0, step)
            for z1 in z1s:
                for z2 in np.arange(0.0, 360.0, step):
                    out.append((f"ZXZ{z1:.0f},{al:.0f},{z2:.0f}",
                                cls.Rz(z2) @ cls.Rx(al) @ cls.Rz(z1)))
        return out

    @classmethod
    def around(cls, base: np.ndarray, span: float = 6.0, step: float = 1.5):
        """Local Z-refinement around an already-good rotation."""
        return [(f"d{d:+.1f}", base @ cls.Rz(d))
                for d in np.arange(-span, span + 1e-9, step)]


# --------------------------------------------------------------------------- #
#  Exact surface distance
# --------------------------------------------------------------------------- #
class SurfaceSampleCache:
    """Reuse of the fixed part's surface samples and KD-tree across one job.

    Every candidate arrangement re-measures the same part A against a freshly
    rotated part B, so :class:`SurfacePairDistance` re-samples A and rebuilds
    its KD-tree once per candidate from geometry that never moved. On a
    ten-candidate run that is ten identical samplings and ten identical tree
    builds. This holds the first result and hands it back.

    Activated as a context manager, and read through :meth:`current` rather
    than passed down, so ``SurfacePairDistance`` keeps its signature::

        with SurfaceSampleCache() as pool:
            ...                       # every construction inside reuses part A
        pool.stats()                  # {'builds': 11, 'reuses': 10, ...}

    Scope and staleness
    -------------------
    The active cache lives in thread-local state, so jobs running concurrently
    in the worker pool never see each other's entries, and leaving the ``with``
    block drops every entry — a cache cannot outlive the job that opened it.
    Entries are keyed on ``hash(mesh)``, which trimesh derives from the vertex
    and face data, so editing a mesh in place invalidates its entry and two
    meshes with identical geometry correctly share one.

    Arrays handed out are shared by every borrower and must be treated as
    read-only; nothing in this module writes to them.
    """

    _local = threading.local()

    def __init__(self):
        self._entries: dict = {}
        self.builds = 0
        self.reuses = 0
        self._outer = None

    # -- activation -------------------------------------------------------- #
    @classmethod
    def current(cls) -> "SurfaceSampleCache | None":
        """The cache open on this thread, or None when running uncached."""
        return getattr(cls._local, "active", None)

    def __enter__(self) -> "SurfaceSampleCache":
        self._outer = self.current()          # nesting restores, never clobbers
        type(self)._local.active = self
        return self

    def __exit__(self, *exc) -> bool:
        type(self)._local.active = self._outer
        self._outer = None
        self._entries.clear()                 # no entry outlives its job
        return False

    # -- lookup ------------------------------------------------------------ #
    def samples_for(self, mesh, n_samples: int, seed: int, rng, sampler):
        """Return ``(points, face_ids, tree)`` for ``mesh``, sampling only once.

        ``rng`` is advanced exactly as the skipped sampling would have advanced
        it. That is the whole reason this is safe: the caller draws part B from
        the *same* generator immediately afterwards, so a cache hit that left
        the stream untouched would silently re-sample B from part A's draws and
        change the numbers. Storing the post-sampling state and restoring it
        keeps the two paths bit-identical.

        The stored state is only meaningful because the caller seeds a fresh
        generator from ``seed`` immediately before calling, and ``seed`` is part
        of the key — so the stream position on entry is always the same one the
        stored state was recorded from.
        """
        key = (hash(mesh), int(n_samples), int(seed))
        entry = self._entries.get(key)
        if entry is not None:
            points, face_ids, tree, state_after = entry
            rng.bit_generator.state = state_after
            self.reuses += 1
            return points, face_ids, tree

        points, face_ids = sampler(mesh, n_samples, rng)
        tree = cKDTree(points)
        self._entries[key] = (points, face_ids, tree, rng.bit_generator.state)
        self.builds += 1
        return points, face_ids, tree

    def publish_field(self, mesh, pitch: float, field) -> None:
        """Offer a distance field built elsewhere for the rest of the job.

        :class:`ClearanceGrid` runs the distance transform during the sweep and
        the refiner wants the same array afterwards. Rather than have the
        refiner rebuild it, the grid leaves it here on its way past. First one
        in wins, so the coarse sweep does not evict the finer field.
        """
        self._entries.setdefault(("field", hash(mesh), float(pitch)), field)

    def field_for(self, mesh, pitch: float, builder=None):
        """The distance field for ``mesh`` at ``pitch``, or None if unbuilt.

        With no ``builder`` this never constructs one: the caller is asking
        whether the sweep already paid for it, and a miss just means the KD
        path runs unpruned, which is what it did before.
        """
        key = ("field", hash(mesh), float(pitch))
        field = self._entries.get(key)
        if field is not None:
            self.reuses += 1
            return field
        if builder is None:
            return None
        field = builder()
        self._entries[key] = field
        self.builds += 1
        return field

    def bvh_for(self, mesh, builder):
        """Return the BVH of ``mesh``, building it at most once per job.

        The same reuse as :meth:`samples_for` and for the same reason — part A
        is fixed across every candidate — but with no RNG to keep in step,
        because building a hierarchy draws no random numbers.
        """
        key = ("bvh", hash(mesh))
        model = self._entries.get(key)
        if model is not None:
            self.reuses += 1
            return model
        model = builder(mesh)
        self._entries[key] = model
        self.builds += 1
        return model

    def stats(self) -> dict:
        return {"builds": self.builds, "reuses": self.reuses,
                "live_entries": len(self._entries)}


class SurfacePairDistance:
    """Minimum surface-to-surface distance between A and a translated B.

    B's rotation is baked in at construction; only its translation varies, so
    both KD-trees are built once and a translation is applied to the *query
    points* instead of rebuilding.

    Two metrics:
      ``sampled(t)``  point-to-point over dense samples. Always >= the true
                      gap. Fast. Measured bias on a real part: 0.013 mm at
                      0.25 mm sample spacing.
      ``exact(t)``    localises the near-contact band with the sampled metric,
                      then computes true point-to-triangle distances there.

    Point pruning is rigorous: a sample at distance ``d0 > dmin0 + 2*move`` at
    the reference pose cannot become the closest pair under any translation of
    magnitude <= ``move``, since its distance drops by at most ``move`` while
    the running minimum rises by at most ``move``.

    A :class:`SurfaceDistanceField` for part A (``field``) and one for part B
    (``field_b``), when supplied, are used to shrink the KD queries against the
    corresponding tree — never to answer them. See :meth:`min_distance_to` and
    :meth:`mask_within`. Every number this class returns is identical with and
    without them; only the number of tree descents changes. Either may be
    omitted, and omitting one costs nothing but the pruning on that side.
    """

    def __init__(self, meshA, meshB, t_ref, n_samples: int = 220_000,
                 move: float = 6.0, seed: int = 0, field=None, field_b=None):
        self.field, self.field_b = field, field_b
        rng = np.random.default_rng(seed)
        pool = SurfaceSampleCache.current()
        if pool is None:
            self.pA, self.fA = self._sample(meshA, n_samples, rng)
            self.treeA = cKDTree(self.pA)
        else:
            # A is the part that never moves; the pool advances rng exactly as
            # the skipped sampling would, so B below is drawn identically
            self.pA, self.fA, self.treeA = pool.samples_for(
                meshA, n_samples, seed, rng, self._sample)
        self.pB, self.fB = self._sample(meshB, n_samples, rng)
        self.triA, self.triB = meshA.triangles, meshB.triangles
        self.treeB = cKDTree(self.pB)
        self.spacing = max(np.sqrt(meshA.area / n_samples),
                           np.sqrt(meshB.area / n_samples))

        t_ref = np.asarray(t_ref, float)
        # Both prunes share one band, taken from B's side. A's side is a plain
        # threshold against that band, so it needs `field_b` — a field for the
        # geometry `treeB` was built from — and gets one only where the caller
        # can supply it without paying for a second distance transform.
        keepB, band = self._band_mask(self.treeA, self.pB + t_ref, move,
                                      self.field)
        keepA = self.mask_within(self.treeB, self.pA - t_ref, band, self.field_b)
        self.qB, self.qfB = self.pB[keepB], self.fB[keepB]
        self.qA, self.qfA = self.pA[keepA], self.fA[keepA]
        self.subB = self.qB[::4]                # cheap metric for inner loops

    # -- field-pruned tree queries ------------------------------------------ #
    #: seeds used to bracket the minimum before filtering. One is enough to be
    #: correct; a handful is enough to be tight, and they cost one batched
    #: query. Measured on the reference part: 1 seed keeps 15% of the cloud,
    #: 32 seeds keep 12% at contact and under 2% well separated.
    _SEEDS = 32

    def min_distance_to(self, tree, points, field, lb=None) -> float:
        """``tree.query(points).min()``, with most of the points ruled out first.

        Exact, not approximate. The field gives a lower bound ``lb`` on every
        point's distance in a single gather. Query the few points with the
        smallest bound to get a distance actually achieved, ``u``. Any point
        with ``lb > u`` has a true distance above ``u`` too, so it cannot hold
        the minimum and is dropped. Whatever survives still contains the
        argmin, so querying only those returns the same number the full query
        would have — the tree, the metric and the answer are unchanged, and
        only the size of the descent differs.

        Falls back to the plain query with no field, on a tiny cloud where the
        gather would not pay, or when the filter rules nothing out.

        ``lb`` lets a caller that already gathered the bound pass it in rather
        than pay for it twice; :meth:`exact` needs the same bound for its
        minimum and for its selection.
        """
        points = np.asarray(points, float)
        if field is None or len(points) <= self._SEEDS:
            return float(tree.query(points, workers=KD_WORKERS)[0].min())

        lb = field.lower_bound(points) if lb is None else lb
        seeds = np.argpartition(lb, self._SEEDS)[:self._SEEDS]
        upper = float(tree.query(points[seeds], workers=KD_WORKERS)[0].min())
        keep = lb <= upper
        if keep.all():
            return float(tree.query(points, workers=KD_WORKERS)[0].min())
        if not keep.any():
            return upper
        d, _ = tree.query(points[keep], workers=KD_WORKERS)
        return float(min(upper, d.min()))

    def _band_mask(self, tree, points, move: float, field):
        """``(dB <= band, band)`` for the constructor's pruning step.

        The band needs the true minimum, which :meth:`min_distance_to` returns
        exactly, and then a per-point test against it. Only points whose lower
        bound is inside the band can pass the real test, so distances are
        computed for those alone and everything else is False by construction.
        The resulting mask equals the one a full query produces.
        """
        points = np.asarray(points, float)
        if field is None:
            d, _ = tree.query(points, workers=KD_WORKERS)
            band = float(d.min()) + 2.0 * move + 1.0
            return d <= band, band

        band = self.min_distance_to(tree, points, field) + 2.0 * move + 1.0
        return self.mask_within(tree, points, band, field), band

    def mask_within(self, tree, points, band: float, field, lb=None):
        """``tree.query(points) <= band``, computing distances only where it can hold.

        A point whose lower bound already exceeds the band cannot have a true
        distance inside it, so it is False without being queried. The mask
        equals the one a full query produces; only the count of descents falls,
        and it falls furthest when the band is tight — the constructor's
        ``move=0`` case narrows it to a millimetre, and :meth:`exact` narrows
        it to 0.8 mm.
        """
        points = np.asarray(points, float)
        if field is None:
            d, _ = tree.query(points, workers=KD_WORKERS)
            return d <= band
        candidate = (field.lower_bound(points) if lb is None else lb) <= band
        mask = np.zeros(len(points), bool)
        if candidate.any():
            d, _ = tree.query(points[candidate], workers=KD_WORKERS)
            mask[candidate] = d <= band
        return mask

    # -- metrics ----------------------------------------------------------- #
    def sampled(self, t, coarse: bool = False) -> float:
        pts = self.subB if coarse else self.qB
        return self.min_distance_to(self.treeA, pts + np.asarray(t, float),
                                    self.field)

    def exact(self, t, k: int = PT_TRI_K, band: float = 0.8) -> float:
        """Minimum surface distance with B translated by ``t``.

        Unchanged in what it computes. The sampled minimum still localises the
        near-contact band and the answer still comes from true point-to-triangle
        distances there; the fields only decide which points are worth a tree
        descent, and they can only exclude a point whose lower bound already
        exceeds the threshold it would have been tested against. The selection
        and the returned distance are the ones the unfiltered code produced.

        This is the tightest band in the pipeline — 0.8 mm against the
        constructor's 13 mm — so it is also where the filter discards most:
        measured, it keeps about a sixth of each cloud.
        """
        t = np.asarray(t, float)

        ptsB = self.qB + t
        # one gather serves both the minimum and the selection below it
        lbB = None if self.field is None else self.field.lower_bound(ptsB)
        best = self.min_distance_to(self.treeA, ptsB, self.field, lb=lbB)
        sel = ptsB[self.mask_within(self.treeA, ptsB, best + band,
                                    self.field, lb=lbB)]
        best = min(best, self._pt_tri(sel, self.treeA, self.fA, self.triA, k))

        # the B side is thresholded against the running best, which the A-side
        # triangle pass may just have lowered — so it needs no minimum of its own
        ptsA = self.qA - t
        sel = ptsA[self.mask_within(self.treeB, ptsA, best + band, self.field_b)]
        best = min(best, self._pt_tri(sel, self.treeB, self.fB, self.triB, k))
        return best

    @staticmethod
    def _pt_tri(pts, tree, face_of_point, tris, k) -> float:
        """Exact distance from ``pts`` to the faces owning their k nearest samples.

        Radius queries are avoided deliberately: one oversized triangle makes a
        radius query return most of the mesh. Area-weighted sampling guarantees
        the truly closest face owns a sample within ~one spacing of the closest
        point, so it appears among the k nearest.
        """
        if len(pts) == 0:
            return np.inf
        _, idx = tree.query(pts, k=k, workers=KD_WORKERS)
        faces = face_of_point[np.asarray(idx).ravel()]
        return float(Geometry.point_triangle_distance(
            np.repeat(pts, k, axis=0), tris[faces]).min())

    def exact_reference(self, t, band: float = None) -> float:
        """Reference implementation: radius query over triangle centroids.

        Kept because it validates :meth:`exact` by an independent route — it
        makes no assumption about sampling density finding the right face, it
        simply tests every triangle that could possibly be in range.

        Slow, and pathologically so on meshes with a few oversized triangles:
        the query radius must cover the largest circumradius, so one 63 mm
        triangle drags most of the mesh into every query. 26 s versus 2 s on
        the reference part. Use for spot checks, not in a loop.
        """
        from scipy.spatial import cKDTree as _KD
        t = np.asarray(t, float)
        band = 2.5 * self.spacing if band is None else band
        ctA, ctB = _KD(self.triA.mean(1)), _KD(self.triB.mean(1) + t)
        rA = np.linalg.norm(self.triA - self.triA.mean(1)[:, None], axis=2).max()
        rB = np.linalg.norm(self.triB - self.triB.mean(1)[:, None], axis=2).max()
        d, _ = self.treeA.query(self.qB + t, workers=KD_WORKERS)
        best = float(d.min())
        best = min(best, self._radius_probe(self.qB[d <= best + band] + t,
                                            ctA, self.triA, best + band + rA))
        d, _ = self.treeB.query(self.qA - t, workers=KD_WORKERS)
        best = min(best, self._radius_probe(self.qA[d <= best + band],
                                            ctB, self.triB + t, best + band + rB))
        return best

    @staticmethod
    def _radius_probe(pts, ctree, tris, radius) -> float:
        if len(pts) == 0:
            return np.inf
        best = np.inf
        for chunk in np.array_split(pts, max(1, len(pts) // 2000)):
            idx = ctree.query_ball_point(chunk, radius)
            keep = [(i, v) for i, v in enumerate(idx) if v]
            if not keep:
                continue
            pi = np.concatenate([np.full(len(v), i) for i, v in keep])
            ti = np.concatenate([v for _, v in keep]).astype(int)
            best = min(best, float(Geometry.point_triangle_distance(
                chunk[pi], tris[ti]).min()))
        return best

    @staticmethod
    def _sample(mesh, n, rng):
        """Area-weighted surface samples plus vertices, with owning-face ids."""
        area = mesh.area_faces
        fid = rng.choice(len(area), size=n, p=area / area.sum())
        tri = mesh.triangles[fid]
        u, v = rng.random((n, 1)), rng.random((n, 1))
        over = (u + v) > 1
        u[over], v[over] = 1 - u[over], 1 - v[over]
        pts = tri[:, 0] + (tri[:, 1] - tri[:, 0]) * u + (tri[:, 2] - tri[:, 0]) * v
        vface = np.full(len(mesh.vertices), -1, np.int64)
        for c in range(3):
            vface[mesh.faces[:, c]] = np.arange(len(mesh.faces))
        ok = vface >= 0
        return (np.vstack([pts, mesh.vertices[ok]]),
                np.concatenate([fid, vface[ok]]))


class BVHPairDistance:
    """Exact surface distance by bounding-volume hierarchy traversal (FCL).

    Interface-compatible with :class:`SurfacePairDistance` — same constructor,
    same ``exact`` / ``sampled`` — so the two are interchangeable through
    ``NesterFactory.distance``. What changes is the method, not the meaning.

    Why this is not just faster
    ---------------------------
    The sampled metric answers "how close do these two *point clouds* get",
    then repairs the answer near the minimum with point-to-triangle distances.
    Its cost is set by the sampling density, which has to be high because the
    density is also what bounds its bias. A BVH answers the mesh question
    directly: descend both trees, prune any pair of boxes further apart than
    the best distance found so far, and only test triangle pairs that survive.
    Cost tracks the geometry near contact rather than the sample count, and
    there is no bias to bound — the answer is exact to floating point.

    Measured against :meth:`SurfacePairDistance.exact` on the reference part at
    its own refined poses: agreement to 8.9e-16 mm, 164 ms per evaluation down
    to 2.4 ms. Construction is 21.6 ms of BVH building against 1,029 ms of
    sampling, tree building and band pruning.

    Overlap
    -------
    FCL reports 0.0 for interpenetrating meshes rather than a penetration
    depth, so this class cannot say *how far* inside the parts are. Nothing
    needs that: every caller asks ``gap >= clearance``, and 0.0 answers it
    correctly. :meth:`Refiner.repair` still walks out of an infeasible start,
    it just cannot see the depth it is climbing out of.

    ``n_samples``, ``move`` and ``seed`` are accepted and ignored. They
    parameterise sampling, and there is none.
    """

    def __init__(self, meshA, meshB, t_ref, n_samples: int = 220_000,
                 move: float = 6.0, seed: int = 0):
        if fcl is None:
            raise RuntimeError(
                "the 'bvh' distance backend needs python-fcl, which is not "
                "installed. Run `pip install python-fcl`, or select the "
                "'sampled' backend (the default).")
        pool = SurfaceSampleCache.current()
        # A is the fixed part; its hierarchy is identical for every candidate
        geomA = (self._bvh(meshA) if pool is None
                 else pool.bvh_for(meshA, self._bvh))
        self._objA = fcl.CollisionObject(geomA, fcl.Transform())
        self._objB = fcl.CollisionObject(self._bvh(meshB), fcl.Transform())
        self._request = fcl.DistanceRequest()

        # interface parity with the sampled metric; there is no point cloud,
        # so the diagnostics that describe one report nothing
        self.spacing = 0.0
        self.qA = self.qB = np.empty((0, 3))

    @staticmethod
    def _bvh(mesh):
        model = fcl.BVHModel()
        model.beginModel(len(mesh.vertices), len(mesh.faces))
        model.addSubModel(np.asarray(mesh.vertices, dtype=np.float64),
                          np.asarray(mesh.faces, dtype=np.int64))
        model.endModel()
        return model

    # -- metrics ----------------------------------------------------------- #
    def exact(self, t, k: int = PT_TRI_K, band: float = 0.8) -> float:
        """Minimum surface distance with B translated by ``t``.

        ``k`` and ``band`` tune the sampled metric's face lookup and have no
        meaning here; they are accepted so the two classes stay swappable.
        """
        self._objB.setTranslation(np.asarray(t, dtype=np.float64))
        result = fcl.DistanceResult()
        fcl.distance(self._objA, self._objB, self._request, result)
        d = float(result.min_distance)
        return d if d > 0.0 else 0.0        # overlap reports 0.0, never a depth

    def sampled(self, t, coarse: bool = False) -> float:
        """The same exact distance.

        On the sampled metric this is the cheap over-estimate used for inner
        loops, and ``coarse`` thins the point set further. Here there is one
        cost and one answer, so both arguments collapse. Callers get a stricter
        number than they asked for, never a looser one.
        """
        return self.exact(t)

    def exact_reference(self, t, band: float = None) -> float:
        """No independent second route exists for this backend; returns
        :meth:`exact`. Cross-checking a BVH result means running the sampled
        backend beside it."""
        return self.exact(t)


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
class Geometry:

    @staticmethod
    def union_extents(boundsA, boundsB, t):
        lo = np.minimum(boundsA[0], boundsB[0] + t)
        hi = np.maximum(boundsA[1], boundsB[1] + t)
        return hi - lo

    @staticmethod
    def axis_leverage(single_extents, union_extents) -> list:
        """Rank axes by how much bounding volume is recoverable on each.

        Shrinking axis *a* by 1 mm saves ``volume / L_a`` mm^3, and at most
        ``L_a - single_a`` mm is available before the parts would have to
        overlap. The product is the real prize on that axis, and it decides
        which axis to bisect against the clearance constraint and which merely
        to scan.

        On the reference part: y offered 20.0 mm of slack at 3.7k mm^3/mm,
        z offered 10.4 mm at 1.5k mm^3/mm, and x offered nothing — so y is the
        binding axis, z the scan axis, x a perturbation check only.

        Returns ``[(axis, slack, marginal, prize), ...]``, best first.
        """
        L = np.asarray(union_extents, float)
        s = np.asarray(single_extents, float)
        vol = float(L.prod())
        rows = []
        for a in range(3):
            slack = float(max(L[a] - s[a], 0.0))
            marginal = vol / L[a]
            rows.append((a, slack, float(marginal), slack * float(marginal)))
        rows.sort(key=lambda r: -r[3])
        return rows

    @staticmethod
    def point_triangle_distance(P: np.ndarray, T: np.ndarray) -> np.ndarray:
        """Exact point-to-triangle distance, vectorised (Ericson, RTCD 5.1.5)."""
        a, b, c = T[:, 0], T[:, 1], T[:, 2]
        ab, ac = b - a, c - a
        ap, bp, cp = P - a, P - b, P - c
        d1, d2 = (ab * ap).sum(1), (ac * ap).sum(1)
        d3, d4 = (ab * bp).sum(1), (ac * bp).sum(1)
        d5, d6 = (ab * cp).sum(1), (ac * cp).sum(1)
        va = d3 * d6 - d5 * d4
        vb = d5 * d2 - d1 * d6
        vc = d1 * d4 - d3 * d2
        denom = 1.0 / np.maximum(va + vb + vc, 1e-300)

        Q = a + ab * (vb * denom)[:, None] + ac * (vc * denom)[:, None]
        m = (d1 <= 0) & (d2 <= 0);                          Q[m] = a[m]
        m = (d3 >= 0) & (d4 <= d3);                         Q[m] = b[m]
        m = (d6 >= 0) & (d5 <= d6);                         Q[m] = c[m]
        m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        Q[m] = a[m] + ab[m] * (d1[m] / np.maximum(d1[m] - d3[m], 1e-300))[:, None]
        m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        Q[m] = a[m] + ac[m] * (d2[m] / np.maximum(d2[m] - d6[m], 1e-300))[:, None]
        m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        num, den = d4[m] - d3[m], (d4[m] - d3[m]) + (d5[m] - d6[m])
        Q[m] = b[m] + (c[m] - b[m]) * (num / np.maximum(den, 1e-300))[:, None]
        return np.linalg.norm(P - Q, axis=1)

    @staticmethod
    def min_area_rect(points_xy: np.ndarray):
        """Rotating-calipers minimum-area enclosing rectangle -> (angle, w, h, area).

        Used to check whether globally re-orienting the finished pair shrinks
        its footprint. It cannot beat the axis-aligned box when the union hull
        is already rectangular, which is the common case for boxy parts.
        """
        from scipy.spatial import ConvexHull
        hull = points_xy[ConvexHull(points_xy).vertices]
        best = None
        for i in range(len(hull)):
            e = hull[(i + 1) % len(hull)] - hull[i]
            L = float(np.hypot(*e))
            if L < 1e-12:
                continue
            u = e / L
            proj = hull @ np.array([[u[0], u[1]], [-u[1], u[0]]]).T
            w = float(np.ptp(proj[:, 0]))
            h = float(np.ptp(proj[:, 1]))
            if best is None or w * h < best[3]:
                best = (float(np.arctan2(u[1], u[0])), float(w), float(h), float(w * h))
        return best


# --------------------------------------------------------------------------- #
#  Continuous refinement
# --------------------------------------------------------------------------- #
class Refiner:
    """Constrained continuous polish of a lattice solution.

    The lattice search is deliberately conservative, so the starting pose is
    feasible but loose. These routines recover the slack.
    """

    def __init__(self, objective: Callable[[np.ndarray], float],
                 feasible: Callable[[np.ndarray], bool]):
        self.objective, self.feasible = objective, feasible

    def bisect_axis(self, t, axis: int, lo: float, hi: float, iters: int = 13,
                    expand: int = 6):
        """Smallest value on ``axis`` in [lo, hi] that stays feasible.

        Feasibility must be monotone in that direction, which holds once the
        parts are separated along it.

        The upper bracket is verified and grown if needed. Skipping that check
        makes the routine return ``hi`` unchanged whenever ``hi`` is itself
        infeasible — a wrong answer that looks entirely reasonable in a table
        of results. Returns ``None`` if no feasible bracket exists.
        """
        t = np.asarray(t, float).copy()
        span = hi - lo
        for _ in range(expand):
            probe = t.copy(); probe[axis] = hi
            if self.feasible(probe):
                break
            lo, hi = hi, hi + span
        else:
            return None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            probe = t.copy(); probe[axis] = mid
            if self.feasible(probe):
                hi = mid
            else:
                lo = mid
        t[axis] = hi
        return t

    def profile_sweep(self, t0, scan_axis: int, bisect_axis: int,
                      scan_values, lo: float, hi: float, iters: int = 13,
                      report: Callable[[str], None] | None = None):
        """Scan one axis; for each value, bisect the binding axis to the limit.

        This is the routine that fixes the delivered pose. The feasible set of
        an interlocking pair is not smooth: the mating feature either engages
        or it does not, and at the engagement boundary the required separation
        jumps discontinuously. On the reference part the binding offset held
        flat at 55.08 mm for six millimetres of scan and then jumped to 60.4 mm
        within one 0.1 mm step. The optimum sits immediately before that cliff.

        Coordinate descent can cross such a cliff and settle on the far side,
        and any smooth local method has nothing to follow. Scanning one axis
        exhaustively while solving the other exactly is what locates the edge.

        Returns rows ``(objective, t)`` sorted best first.
        """
        rows = []
        for val in scan_values:
            t = np.asarray(t0, float).copy()
            t[scan_axis] = val
            t = self.bisect_axis(t, bisect_axis, lo, hi, iters)
            if t is None:
                continue
            obj = self.objective(t)
            rows.append((obj, t.copy()))
            if report:
                report(f"      scan {val:7.2f} -> bind {t[bisect_axis]:7.3f} "
                       f"objective {obj:12,.0f}")
        rows.sort(key=lambda r: r[0])
        return rows

    def perturb_check(self, t, axis: int, deltas, bisect_axis: int,
                      lo: float, hi: float, report=None) -> dict:
        """Re-solve at perturbed values of ``axis`` to confirm a local optimum.

        The lattice search already ranked this axis, but it did so under a
        conservative clearance. Re-solving in continuous space confirms the
        choice still holds once the slack is recovered — on the reference part
        every non-zero x shift cost more in width than it recovered in depth.
        """
        base = self.objective(np.asarray(t, float))
        out = {0.0: base}
        for d in deltas:
            probe = np.asarray(t, float).copy()
            probe[axis] += d
            solved = self.bisect_axis(probe, bisect_axis, lo, hi)
            out[float(d)] = self.objective(solved) if solved is not None else np.inf
            if report:
                report(f"      delta {d:+.1f} -> objective {out[float(d)]:12,.0f}")
        out["is_optimum"] = bool(base <= min(v for k, v in out.items()
                                             if isinstance(k, float)))
        return out

    def descend(self, t, steps: Iterable[float] = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)):
        """Axis-wise coordinate descent on the objective, subject to feasibility."""
        if not algorithm_enabled('descend'):
            _refuse('descend')
        cur = np.asarray(t, float).copy()
        best = self.objective(cur)
        for step in steps:
            moved = True
            while moved:
                moved = False
                for ax in range(3):
                    for s in (-step, step):
                        cand = cur.copy(); cand[ax] += s
                        val = self.objective(cand)
                        if val < best - 1e-9 and self.feasible(cand):
                            cur, best, moved = cand, val, True
        return cur, best

    @staticmethod
    def repair(t, axis: int, feasible, step: float = 0.25, limit: int = 400):
        """Push apart along ``axis`` until feasible (for an infeasible start)."""
        if not algorithm_enabled('refiner_repair'):
            _refuse('refiner_repair')
        t = np.asarray(t, float).copy()
        for _ in range(limit):
            if feasible(t):
                return t
            t[axis] += step
        raise RuntimeError("could not repair to a feasible pose")


# --------------------------------------------------------------------------- #
#  Self-tests
# --------------------------------------------------------------------------- #
class Validation:
    """Checks with known answers, run before trusting any nesting output.

    Every number this pipeline produces rests on two primitives: the voxeliser
    and the distance metric. Both fail quietly when they fail — a broken parity
    fill returns a plausible solid, and a broken distance returns a plausible
    float. These tests have analytic answers, so they cannot pass by accident.
    """

    @staticmethod
    def sphere_pair(radius: float = 10.0, centre_distance: float = 25.0,
                    subdivisions: int = 4, n: int = 50_000, backend=None) -> dict:
        """Two spheres: the surface gap is analytically ``d - 2r``.

        Faceting makes the discrete surface sit marginally inside the true one,
        so a correct implementation lands on or a hair above the analytic value.

        ``backend`` is the distance class to test, so the gate exercises the
        metric the run will actually use rather than a fixed one.
        """
        backend = backend or SurfacePairDistance
        s = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        t = np.array([centre_distance, 0.0, 0.0])
        d = backend(s, s.copy(), t, n_samples=n, move=0.0)
        got = d.exact(t)
        want = centre_distance - 2 * radius
        return {"expected": want, "got": float(got),
                "error": float(got - want), "pass": bool(abs(got - want) < 0.05)}

    @staticmethod
    def voxeliser(mesh: trimesh.Trimesh, pitch: float = 0.5,
                  tol: float = 0.02, trials: int = 20) -> dict:
        axis = ScanlineVoxelizer.volume_error(mesh, pitch)
        rot = ScanlineVoxelizer.rotation_robustness(mesh, pitch, trials)
        return {"axis_aligned_error": float(axis), "rotated": rot,
                "pass": bool(abs(axis) < tol and rot["max_abs"] < tol)}

    @staticmethod
    def distance_agreement(dist: "SurfacePairDistance", t, tol: float = 1e-3) -> dict:
        """Cross-check the fast metric against the independent slow one."""
        fast, ref = dist.exact(t), dist.exact_reference(t)
        return {"fast": float(fast), "reference": float(ref),
                "delta": float(fast - ref), "pass": bool(abs(fast - ref) < tol)}

    @staticmethod
    def clearance_met(gap: float, clearance: float, tol: float = 1e-3) -> dict:
        return {"gap": float(gap), "required": float(clearance),
                "pass": bool(gap >= clearance - tol)}


# --------------------------------------------------------------------------- #
#  Interactive model export
# --------------------------------------------------------------------------- #
class Preview:
    """glTF-binary export of a nested pair, for an interactive viewer.

    This used to render PNGs with matplotlib: three orthographic projections
    plus a shaded isometric subplot. It was replaced because it was the single
    most expensive stage in the pipeline and none of it fed the result.
    Measured on ``electric_drill.stl`` (quick profile), the three images cost
    50.6 s of a 118.1 s run -- 42.8% -- and on ``Spanner-stl.stl`` one
    ``Poly3DCollection(shade=True)`` alone cost 232.7 s of a 238.9 s render,
    because mplot3d builds a Path object and a masked array per triangle.

    A GLB carries the same geometry in one binary file that a browser can
    orbit, so the viewer replaces every fixed viewpoint at once -- the side
    elevation that confirms an interlock is now something you rotate to rather
    than something the server has to guess in advance. Writing it is
    serialisation only: no rasterising, no depth sort, no figure.

    Each copy becomes its own named node with its own colour, so the two parts
    stay tellable apart and the viewer can address them individually.
    """

    #: teal / orange, the colours the PNG renderer used, kept for continuity
    COLOURS = ((27, 158, 119), (217, 95, 2))

    @staticmethod
    def _material(rgb, name):
        return trimesh.visual.material.PBRMaterial(
            name=name,
            baseColorFactor=[rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0],
            metallicFactor=0.05, roughnessFactor=0.65)

    @classmethod
    def pair_glb(cls, parts: Sequence[trimesh.Trimesh], path: str,
                 names: Sequence[str] | None = None) -> str:
        """Write ``parts`` to ``path`` as a single GLB, one node per part.

        Materials rather than per-face colour arrays: a face-colour array costs
        four bytes a triangle in the file and buys nothing here, because each
        copy is one flat colour.
        """
        scene = trimesh.Scene()
        for i, part in enumerate(parts):
            m = part.copy()
            m.visual = trimesh.visual.TextureVisuals(
                material=cls._material(cls.COLOURS[i % len(cls.COLOURS)],
                                       f"copy_{i}"))
            label = (names[i] if names and i < len(names)
                     else f"copy_{chr(ord('A') + i)}")
            scene.add_geometry(m, node_name=label, geom_name=label)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(scene.export(file_type="glb"))
        return str(path)

    @classmethod
    def part_glb(cls, mesh: trimesh.Trimesh, path: str) -> str:
        """Write a single part, for inspecting the input before nesting."""
        return cls.pair_glb([mesh], path, names=["part"])


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class NestResult:
    transform: np.ndarray            # 4x4 placing copy B relative to copy A
    label: str
    extents: np.ndarray
    volume: float
    area: float
    gap: float
    log: list = field(default_factory=list)

    def summary(self) -> str:
        e = self.extents
        return (f"{self.label}: bbox {e[0]:.2f} x {e[1]:.2f} x {e[2]:.2f} mm | "
                f"volume {self.volume:,.0f} mm^3 | footprint {self.area:,.0f} mm^2 | "
                f"gap {self.gap:.4f} mm")


class PairNester:
    """Nest two copies of ``mesh`` with a guaranteed minimum surface clearance.

    >>> nester = PairNester(trimesh.load("part.stl"), clearance=5.0)
    >>> result = nester.run()
    >>> nester.export("nested.stl")
    """

    def __init__(self, mesh: trimesh.Trimesh, clearance: float = 5.0,
                 coarse_pitch: float = 1.0, fine_pitch: float = 0.5,
                 n_samples: int = 220_000, verbose: bool = True,
                 repair: bool = True):
        self.verbose = verbose
        # an open mesh is closed here, once, so every stage below sees the same
        # geometry; MeshRepairError carries wording fit to show a user
        self.mesh, self.repair_report = MeshRepair.ensure_solid(
            mesh, allow_repair=repair, log=self._log)
        self.clearance = clearance
        self.coarse_pitch, self.fine_pitch = coarse_pitch, fine_pitch
        self.n_samples = n_samples
        self.result: NestResult | None = None

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # -- stages ------------------------------------------------------------ #
    def sweep(self, orientations, pitch: float, top: int = 5):
        """Rank candidate rotations by the union bounding-box volume."""
        radius = ClearanceGrid.safe_radius(self.clearance, pitch)
        cg = ClearanceGrid(self.mesh, pitch, radius)
        oracle = TranslationOracle(cg)
        rows = []
        for label, T in orientations:
            mB = self.mesh.copy(); mB.apply_transform(T)
            r = oracle.search(mB)
            if r is None:
                continue
            rows.append((r.volume, label, T, r.t_volume, r.area, r.t_area))
        rows.sort(key=lambda r: r[0])
        for v, label, *_ in rows[:top]:
            self._log(f"    {label:22s} volume {v:12,.0f}")
        return rows

    def refine(self, T: np.ndarray, t0: np.ndarray, guard_extra: float = 0.08,
               strategy: str = "profile", scan_span: float = 1.6,
               scan_step: float = 0.1):
        """Recover the slack the conservative lattice search left behind.

        ``strategy='profile'`` scans the secondary axis and bisects the binding
        axis at each step. It is the default because it is the only one of the
        two that reliably finds the feasibility cliff. ``strategy='descend'``
        runs plain coordinate descent — cheaper, smooth-only, and it can settle
        on the wrong side of a discontinuity.
        """
        mB = self.mesh.copy(); mB.apply_transform(T)
        dist = SurfacePairDistance(self.mesh, mB, t0, self.n_samples)
        self._log(f"    sample spacing {dist.spacing:.3f} mm, "
                  f"{len(dist.qB):,} active points")

        def objective(t):
            return float(Geometry.union_extents(self.mesh.bounds, mB.bounds, t).prod())

        guard = self.clearance + guard_extra
        coarse_ok = lambda t: dist.sampled(t, coarse=True) >= guard
        t = np.asarray(t0, float).copy()
        if not coarse_ok(t):
            t = Refiner.repair(t, int(np.argmax(np.abs(t))), coarse_ok)

        lev = Geometry.axis_leverage(
            self.mesh.extents, Geometry.union_extents(self.mesh.bounds, mB.bounds, t))
        bind_ax, scan_ax, free_ax = lev[0][0], lev[1][0], lev[2][0]
        self._log(f"    axis leverage: bind={'xyz'[bind_ax]} "
                  f"(slack {lev[0][1]:.1f} mm, {lev[0][2]:,.0f} mm^3/mm)  "
                  f"scan={'xyz'[scan_ax]}  check={'xyz'[free_ax]}")

        ref = Refiner(objective, coarse_ok)
        if strategy == "profile":
            span = np.arange(t[scan_ax] - scan_span, t[scan_ax] + scan_span + 1e-9,
                             scan_step)
            lo = t[bind_ax] - 12.0
            rows = ref.profile_sweep(t, scan_ax, bind_ax, span, lo, t[bind_ax] + 6.0,
                                     report=self._log if self.verbose else None)
            if not rows:
                raise RuntimeError("profile sweep found no feasible column")
            t = rows[0][1]
            chk = ref.perturb_check(t, free_ax, (-2.0, -1.0, 1.0, 2.0), bind_ax,
                                    lo, t[bind_ax] + 8.0,
                                    report=self._log if self.verbose else None)
            self._log(f"    perturbation check on {'xyz'[free_ax]}: "
                      f"{'optimum holds' if chk['is_optimum'] else 'BETTER POSE OFF-AXIS'}")
        elif strategy == "descend":
            t, _ = ref.descend(t)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        # final tightening against the exact metric, on the binding axis only
        exact_ok = lambda t: dist.exact(t) >= self.clearance
        tight = Refiner(objective, exact_ok).bisect_axis(
            t, bind_ax, t[bind_ax] - 1.5, t[bind_ax], iters=9)
        if tight is not None:
            t = tight
        return t, dist.exact(t), objective(t), dist

    def run(self, coarse_step: float = 15.0, so3_step: float | None = 30.0,
            fine_span: float = 6.0, fine_step: float = 1.5,
            strategy: str = "profile", validate: bool = True) -> NestResult:
        self._log("stage 0: intake audit")
        audit = MeshAudit(self.mesh).assert_usable()
        self._log(f"    {audit['faces']} faces, fill ratio {audit['fill_ratio']:.1%}, "
                  f"{'already axis-aligned' if audit['already_axis_aligned'] else 'OBB is tighter than AABB'}")
        if validate:
            v = Validation.voxeliser(self.mesh, self.fine_pitch, trials=12)
            self._log(f"    voxeliser: axis {100*v['axis_aligned_error']:+.2f}%, "
                      f"rotated max {100*v['rotated']['max_abs']:.2f}%  "
                      f"[{'pass' if v['pass'] else 'FAIL'}]")
            if not v["pass"]:
                raise RuntimeError("voxeliser validation failed; refusing to nest")

        self._log("stage 1: coarse orientation sweep")
        cands = OrientationSet.z_family(coarse_step)
        if so3_step:
            cands += OrientationSet.so3(so3_step)
        rows = self.sweep(cands, self.coarse_pitch)
        if not rows:
            raise RuntimeError("no feasible arrangement found")
        _, label, T_best, _, _, _ = rows[0]

        self._log("stage 2: fine local sweep")
        rows = self.sweep(OrientationSet.around(T_best, fine_span, fine_step),
                          self.fine_pitch)
        vol, sub, T, t0, *_ = rows[0]

        self._log("stage 3: continuous refinement")
        t, gap, vol, dist = self.refine(T, t0, strategy=strategy)

        if validate:
            chk = Validation.clearance_met(gap, self.clearance)
            self._log(f"    clearance {gap:.4f} >= {self.clearance} "
                      f"[{'pass' if chk['pass'] else 'FAIL'}]")
            if not chk["pass"]:
                raise RuntimeError("refined pose violates the clearance")

        M = np.eye(4); M[:3, :3] = T[:3, :3]; M[:3, 3] = t
        mB = self.mesh.copy(); mB.apply_transform(M)
        e = Geometry.union_extents(self.mesh.bounds, mB.bounds, np.zeros(3))
        self.result = NestResult(M, f"{label}{sub}", e, float(e.prod()),
                                 float(e[0] * e[1]), float(gap))
        self._log(self.result.summary())
        return self.result

    # -- output ------------------------------------------------------------ #
    def assembly(self, origin_corner: bool = True) -> trimesh.Trimesh:
        if self.result is None:
            raise RuntimeError("call run() first")
        A = self.mesh.copy()
        B = self.mesh.copy(); B.apply_transform(self.result.transform)
        if origin_corner:
            shift = -np.minimum(A.bounds[0], B.bounds[0])
            A.apply_translation(shift); B.apply_translation(shift)
        return trimesh.util.concatenate([A, B])

    def export(self, path: str, origin_corner: bool = True) -> str:
        self.assembly(origin_corner).export(path)
        return path

    def preview(self, path: str, origin_corner: bool = True) -> str:
        """Write the nested pair as a GLB; see :class:`Preview` for why."""
        parts = self.assembly(origin_corner).split(only_watertight=False)
        return Preview.pair_glb(parts, path)

    def verify(self, path: str, n_samples: int = 250_000, seed: int = 12345) -> dict:
        """Re-measure the written file from scratch: bodies, volumes, true gap."""
        if not algorithm_enabled('pair_nester_verify'):
            _refuse('pair_nester_verify')
        chk = trimesh.load(path)
        parts = chk.split(only_watertight=False)
        out = {
            "bodies": len(parts),
            "extents": (chk.bounds[1] - chk.bounds[0]).tolist(),
            "volumes": [float(p.volume) for p in parts],
            "watertight": [bool(p.is_watertight) for p in parts],
        }
        if len(parts) == 2:
            d = SurfacePairDistance(parts[0], parts[1], np.zeros(3),
                                    n_samples, move=0.0, seed=seed)
            out["gap"] = float(d.exact(np.zeros(3)))
        return out

    # -- reference baselines ----------------------------------------------- #
    def baselines(self) -> dict:
        """Bounding volumes of the trivial non-nested arrangements, for context."""
        e, c = self.mesh.extents, self.clearance
        out = {}
        for ax, name in enumerate("xyz"):
            f = e.copy(); f[ax] = 2 * e[ax] + c
            out[f"side_by_side_{name}"] = {"extents": f.tolist(),
                                           "volume": float(f.prod()),
                                           "area": float(f[0] * f[1])}
        return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Nest 2 copies of an STL with clearance.")
    ap.add_argument("stl")
    ap.add_argument("-o", "--out", default="nested.stl")
    ap.add_argument("-c", "--clearance", type=float, default=5.0)
    ap.add_argument("--coarse-pitch", type=float, default=1.0)
    ap.add_argument("--fine-pitch", type=float, default=0.5)
    ap.add_argument("--coarse-step", type=float, default=15.0)
    ap.add_argument("--so3-step", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=220_000)
    ap.add_argument("--no-so3", action="store_true", help="skip the full SO(3) sweep")
    args = ap.parse_args()

    part = trimesh.load(args.stl)
    nester = PairNester(part, args.clearance, args.coarse_pitch,
                        args.fine_pitch, args.samples)
    print(f"voxeliser check: {100 * ScanlineVoxelizer.volume_error(part, args.fine_pitch):+.2f}% volume error")
    nester.run(args.coarse_step, None if args.no_so3 else args.so3_step)
    nester.export(args.out)
    print("baselines:", json.dumps(nester.baselines(), indent=2))
    print("verify:", json.dumps(nester.verify(args.out), indent=2))
    print("transform:\n", np.round(nester.result.transform, 4))


# =========================================================================== #
#  APPROACHES THAT WERE TRIED AND REJECTED
#
#  Kept as working, benchmarked code. Each is correct; each lost on cost or
#  accuracy for this class of part. On a different mesh the ranking can flip,
#  so the alternatives are here to be re-measured rather than re-discovered.
# =========================================================================== #
class AlternativeVoxelizers:
    """Occupancy methods that lost to :class:`ScanlineVoxelizer`."""

    @staticmethod
    def trimesh_surface_fill(mesh, pitch: float):
        """Surface voxelisation + flood fill (``mesh.voxelized().fill()``).

        REJECTED. Marks every voxel a triangle passes through, so a wall thinner
        than the pitch becomes two voxels wide. Measured on the reference part:
        19,878 mm^3 against a true 11,068 mm^3 at 1.0 mm pitch — +79.6%. The
        inflation is conservative for clearance but silently destroys the nest
        quality, since it pads every surface by ~half a voxel.
        """
        vg = mesh.voxelized(pitch=pitch).fill()
        return np.asarray(vg.matrix), np.round(vg.transform[:3, 3] / pitch - 0.5).astype(int)

    @staticmethod
    def ray_containment(mesh, pitch: float):
        """Point-in-mesh test on every voxel centre (``mesh.contains``).

        REJECTED on speed. Exact and trivially correct, but trimesh's pure
        Python ray/triangle intersector needs ``rtree`` and still costs minutes
        for the ~10^5-10^6 centres a real part needs. The scanline method gets
        the same answer by amortising one sorted crossing list per column.
        """
        lo, hi = mesh.bounds[0] - pitch, mesh.bounds[1] + pitch
        axes = [np.arange(np.floor(lo[i] / pitch) + 0.5,
                          np.ceil(hi[i] / pitch) + 0.5) * pitch for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
        shape = pts.shape[:3]
        occ = mesh.contains(pts.reshape(-1, 3)).reshape(shape)
        return occ, np.array([np.floor(lo[i] / pitch) for i in range(3)], int)

    @classmethod
    def benchmark(cls, mesh, pitch: float = 1.0, include_rays: bool = False):
        """Volume error and wall time for every occupancy method."""
        import time
        out = {}
        for name, fn in [("scanline", ScanlineVoxelizer.rasterize),
                         ("trimesh_fill", cls.trimesh_surface_fill)] + \
                        ([("ray_contains", cls.ray_containment)] if include_rays else []):
            t0 = time.time()
            occ, _ = fn(mesh, pitch)
            out[name] = {"volume_error_pct": 100 * (occ.sum() * pitch ** 3 / mesh.volume - 1),
                         "seconds": time.time() - t0}
        return out


class AlternativeDistance:
    """Distance methods that lost to the k-NN-to-face lookup."""

    @staticmethod
    def radius_triangle_lookup(pts, centroid_tree, tris, radius) -> float:
        """Gather candidate faces by a radius query on triangle centroids.

        REJECTED. The radius must be ``d + max_circumradius`` to stay correct,
        and one oversized triangle poisons it: the reference part is 28 mm wide
        yet carries a triangle of circumradius 63 mm, so every query returned
        most of the mesh. Cost 26 s per evaluation against 2 s for the k-NN
        route. Still the right choice on uniformly tessellated meshes.
        """
        if len(pts) == 0:
            return np.inf
        best = np.inf
        for chunk in np.array_split(pts, max(1, len(pts) // 2000)):
            idx = centroid_tree.query_ball_point(chunk, radius)
            keep = [(i, v) for i, v in enumerate(idx) if v]
            if not keep:
                continue
            pi = np.concatenate([np.full(len(v), i) for i, v in keep])
            ti = np.concatenate([v for _, v in keep]).astype(int)
            best = min(best, float(Geometry.point_triangle_distance(
                chunk[pi], tris[ti]).min()))
        return best

    @staticmethod
    def trimesh_proximity(mesh, points) -> float:
        """``trimesh.proximity.ProximityQuery.on_surface`` (rtree BVH).

        REJECTED on speed: 11.8 s for 20,000 query points on the reference
        part, where 200,000+ are needed for sub-0.05 mm sampling bias.
        """
        return float(trimesh.proximity.ProximityQuery(mesh).on_surface(points)[1].min())


class DistanceField:
    """Precomputed unsigned distance field around a mesh, with trilinear lookup.

    DESIGNED AS A FASTER SQUEEZE ORACLE, then set aside: covering the reference
    part at 0.25 mm needs ~29 M cells (~230 MB for the EDT output), and the
    field inherits a half-voxel bias from the occupancy it is built on. At
    0.5 mm it is cheap and fine for coarse work, but the final gap still has to
    be certified against :meth:`SurfacePairDistance.exact`.

    Worth it only when a search needs hundreds of evaluations: lookup is O(P)
    with a tiny constant, versus a KD-tree query per evaluation.
    """

    def __init__(self, mesh, pitch: float = 0.5, margin: float = 12.0):
        occ, i0 = ScanlineVoxelizer.rasterize(mesh, pitch)
        pad = int(np.ceil(margin / pitch))
        grid = np.zeros(np.array(occ.shape) + 2 * pad, bool)
        grid[pad:-pad, pad:-pad, pad:-pad] = occ
        self.field = distance_transform_edt(~grid, sampling=pitch).astype(np.float32)
        self.origin = (i0 - pad + 0.5) * pitch          # world coord of cell [0,0,0]
        self.pitch = pitch
        self.cells = self.field.size

    def query(self, pts) -> np.ndarray:
        """Trilinear-interpolated distance at arbitrary world points."""
        g = (np.asarray(pts, float) - self.origin) / self.pitch
        g = np.clip(g, 0, np.array(self.field.shape) - 1.001)
        i = np.floor(g).astype(int)
        f = g - i
        out = np.zeros(len(g))
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((f[:, 0] if dx else 1 - f[:, 0]) *
                         (f[:, 1] if dy else 1 - f[:, 1]) *
                         (f[:, 2] if dz else 1 - f[:, 2]))
                    out += w * self.field[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return out

    def min_gap(self, pts, t) -> float:
        return float(self.query(np.asarray(pts, float) + np.asarray(t, float)).min())


# =========================================================================== #
#  Shape reconnaissance — run before nesting to know what you are packing
# =========================================================================== #
class Inspector:
    """Cheap diagnostics that decide how to configure the search."""

    @staticmethod
    def report(mesh) -> dict:
        """Fill ratio, AABB vs OBB, watertightness.

        A low fill ratio means concavities worth interlocking into. An OBB no
        smaller than the AABB means the part is already in its natural frame,
        which justifies restricting the sweep to the Z-family.
        """
        aabb = float(np.prod(mesh.extents))
        obb = float(mesh.bounding_box_oriented.volume)
        return {"extents": mesh.extents.tolist(),
                "volume": float(mesh.volume),
                "aabb_volume": aabb,
                "obb_volume": obb,
                "obb_beats_aabb": obb < aabb * 0.98,
                "fill_ratio": float(mesh.volume / aabb),
                "watertight": bool(mesh.is_watertight),
                "faces": len(mesh.faces)}

    @staticmethod
    def slice_profile(mesh, axis: int = 2, n: int = 13):
        """Cross-section area along an axis. Reveals pockets and thin sections.

        Uses ``mesh.section``, which needs ``shapely``; falls back to counting
        occupied voxels per layer, which needs nothing extra.
        """
        if not algorithm_enabled('slice_profile'):
            _refuse('slice_profile')
        lo, hi = mesh.bounds[0][axis], mesh.bounds[1][axis]
        try:
            rows = []
            for v in np.linspace(lo, hi, n + 2)[1:-1]:
                normal = np.eye(3)[axis]
                sec = mesh.section(plane_origin=normal * v, plane_normal=normal)
                rows.append((float(v), 0.0 if sec is None else float(sec.to_2D()[0].area)))
            return rows
        except Exception:
            occ, i0 = ScanlineVoxelizer.rasterize(mesh, (hi - lo) / 120)
            pitch = (hi - lo) / 120
            counts = occ.sum(axis=tuple(a for a in range(3) if a != axis))
            idx = np.linspace(0, len(counts) - 1, n).astype(int)
            return [(float((i0[axis] + k + 0.5) * pitch), float(counts[k] * pitch ** 2))
                    for k in idx]



# =========================================================================== #
#  Inventory
# =========================================================================== #
#: Every distinct algorithm considered, with its verdict. "used" ones run in
#: PairNester.run(); "rejected" ones are implemented above for re-measurement;
#: "analysed" ones were reasoned about and never needed code.
ALGORITHMS = [
    # stage,            name,                                    status,     where
    ("recon",  "AABB vs OBB principal-frame check",              "used",     "Inspector.report"),
    ("recon",  "planar cross-section profile",                   "used",     "Inspector.slice_profile"),
    ("voxel",  "z-scanline parity solid voxelisation",           "used",     "ScanlineVoxelizer"),
    ("voxel",  "irrational jitter + widened column window",      "used",     "ScanlineVoxelizer"),
    ("voxel",  "difference-array interval fill",                 "used",     "ScanlineVoxelizer"),
    ("voxel",  "surface voxelisation + flood fill",              "rejected", "AlternativeVoxelizers"),
    ("voxel",  "ray-cast point-in-mesh containment",             "rejected", "AlternativeVoxelizers"),
    ("clear",  "EDT dilation of free space",                     "used",     "ClearanceGrid"),
    ("search", "FFT correlation no-fit translation map",         "used",     "TranslationOracle"),
    ("search", "separable per-axis AABB objective",              "used",     "TranslationOracle"),
    ("search", "slice-wise masked argmin",                       "used",     "TranslationOracle"),
    ("search", "Z-spin + flip orientation family",               "used",     "OrientationSet.z_family"),
    ("search", "ZXZ Euler SO(3) grid sweep",                     "used",     "OrientationSet.so3"),
    ("search", "local orientation refinement",                   "used",     "OrientationSet.around"),
    ("dist",   "area-weighted barycentric surface sampling",     "used",     "SurfacePairDistance"),
    ("dist",   "KD-tree point-to-point minimum",                 "used",     "SurfacePairDistance.sampled"),
    ("dist",   "translation-bounded pruning band",               "used",     "SurfacePairDistance"),
    ("dist",   "Ericson exact point-triangle distance",          "used",     "Geometry"),
    ("dist",   "k-NN sample to owning-face candidate lookup",    "used",     "SurfacePairDistance"),
    ("dist",   "radius query on triangle centroids",             "rejected", "AlternativeDistance"),
    ("dist",   "trimesh ProximityQuery BVH",                     "rejected", "AlternativeDistance"),
    ("dist",   "precomputed EDT field + trilinear lookup",       "optional", "DistanceField"),
    ("refine", "coordinate descent with step halving",           "used",     "Refiner.descend"),
    ("refine", "monotone bisection on the binding axis",         "used",     "Refiner.bisect_axis"),
    ("refine", "outward feasibility repair",                     "used",     "Refiner.repair"),
    ("refine", "profile sweep (outer grid, inner bisection)",    "used",     "Refiner.profile_sweep"),
    ("global", "rotating-calipers min-area rectangle",           "optional", "Geometry.min_area_rect"),
    ("global", "global re-orientation of the finished pair",     "analysed", "-- hull already rectangular"),
    ("out",    "concatenate + origin-corner shift",              "used",     "PairNester.assembly"),
    ("out",    "connected-component body split check",           "used",     "PairNester.verify"),
    ("out",    "fresh-seed independent gap re-measurement",      "used",     "PairNester.verify"),
    ("out",    "naive packing baselines",                        "used",     "PairNester.baselines"),
]


def algorithm_table() -> str:
    """Printable inventory: 33 algorithms, grouped by stage."""
    from collections import Counter
    n = Counter(s for _, _, s, _ in ALGORITHMS)
    lines = [f"{len(ALGORITHMS)} algorithms  ("
             + ", ".join(f"{v} {k}" for k, v in sorted(n.items())) + ")", ""]
    stage = None
    for st, name, status, where in ALGORITHMS:
        if st != stage:
            lines.append(f"[{st}]")
            stage = st
        lines.append(f"   {status:9s} {name:44s} {where}")
    return "\n".join(lines)
