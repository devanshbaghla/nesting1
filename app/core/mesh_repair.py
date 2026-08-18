"""
mesh_repair.py — make an imperfect STL usable for solid voxelisation.

Why this is a gate and not a nicety
-----------------------------------
Parity scanline voxelisation (:class:`~.nesting3d.ScanlineVoxelizer`) is only
defined for a closed surface: a ray entering the solid must eventually leave
it. On a column that sees a hole the crossing count is odd, the entry/exit
pairing shifts by one, and the fill runs on past the real boundary. Nothing
raises — the pipeline returns a plausible-looking solid that is simply wrong,
and every number downstream (volume, clearance, the delivered pose) inherits
the error. So an open mesh has to be closed before voxelisation, or refused.

Most STLs that fail the gate fail for dull, repairable reasons:

  * vertices never welded, so adjacent triangles do not share an edge and the
    whole surface reads as open (by far the most common — plenty of exporters
    write each facet independently, which is what the STL format invites);
  * a handful of zero-area slivers or exactly duplicated faces;
  * winding flipped on some faces by an old exporter, which leaves the surface
    closed but makes ``volume`` and the outward normal meaningless;
  * a few small genuine holes where a fillet or boolean did not knit.

All of those are fixed here. Genuinely open geometry — a surface patch, a
zero-thickness shell, a part exported with its back faces missing — is not
repairable in any meaningful sense and is refused with an explanation.

Repairs are verified, not trusted
---------------------------------
``trimesh.repair.fill_holes`` caps a boundary loop with a triangle fan. On a
small knit failure that is exactly right. On a large opening it invents a flat
lid across the mouth of the part, producing a mesh that is watertight and
plausible and has nothing to do with the object. The checks in
:meth:`MeshRepair._validate` exist to catch that: a repaired solid must have
positive finite volume and must fit inside its own convex hull. A repair that
fails them is discarded and reported as a failure, which is a better outcome
than nesting a lie.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import trimesh

__all__ = ["MeshRepairError", "RepairReport", "MeshRepair",
           "DenoiseReport", "MeshDenoise",
           "SolidifyReport", "MeshSolidify"]


@dataclass
class DenoiseReport:
    """Which disconnected fragments were dropped, and what that bought."""

    components_before: int = 1
    components_after: int = 1
    dropped: list = field(default_factory=list)
    faces_before: int = 0
    faces_after: int = 0
    extents_before: list = field(default_factory=list)
    extents_after: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.dropped)

    def bbox_shrink(self) -> float:
        """Fraction of the bounding box the strays were responsible for."""
        before = float(np.prod(self.extents_before)) if self.extents_before else 0.0
        after = float(np.prod(self.extents_after)) if self.extents_after else 0.0
        return 0.0 if before <= 0 else max(0.0, 1.0 - after / before)

    def summary(self) -> str:
        if not self.changed:
            return "no stray fragments found"
        shrink = self.bbox_shrink()
        note = (f", bounding box down {100 * shrink:.1f}%"
                if shrink > 1e-6 else ", bounding box unchanged")
        return (f"dropped {len(self.dropped)} stray "
                f"{'fragment' if len(self.dropped) == 1 else 'fragments'} "
                f"({self.faces_before - self.faces_after:,} faces){note}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["changed"] = self.changed
        d["bbox_shrink"] = self.bbox_shrink()
        d["summary"] = self.summary()
        return d


class MeshDenoise:
    """Remove disconnected fragments that are not part of the object.

    Scanned and converted STLs routinely carry debris: a few triangles left
    behind by a boolean, a speck of scanner noise, a duplicate shell offset
    from the part. None of it is visible at a glance and none of it stops the
    mesh being watertight, so it survives every check in :class:`MeshRepair`.

    It still ruins the answer. Every number this pipeline reports —  bounding
    volume, footprint, the axis leverage that decides which axis to bisect —
    is measured against the axis-aligned box around the geometry. One stray
    triangle sitting 200 mm off the part stretches that box by 200 mm, and the
    nesting is then optimised for a shape that is mostly empty air. The failure
    is silent: the arrangement is still feasible, still meets the clearance,
    and is simply far worse than it should be.

    A fragment is treated as debris when its bounding-box diagonal is under
    ``ratio`` of the largest fragment's — physical size, which is what the
    bounding box actually responds to, rather than volume, which is undefined
    for the open shells debris often is. The default 5% is deliberately timid:
    a genuine two-piece assembly exported as one file keeps both pieces.
    """

    #: fragment is debris below this share of the main fragment's bbox diagonal
    DIAGONAL_RATIO = 0.05

    @classmethod
    def strip_stray_shells(cls, mesh: trimesh.Trimesh, *, ratio: float = None,
                           log=None) -> tuple[trimesh.Trimesh, DenoiseReport]:
        """Return ``(mesh, report)``, dropping debris fragments.

        A single-body mesh is returned untouched without splitting anything,
        which is the common case and costs one cached lookup.
        """
        ratio = cls.DIAGONAL_RATIO if ratio is None else ratio
        report = DenoiseReport(
            faces_before=int(len(mesh.faces)), faces_after=int(len(mesh.faces)),
            extents_before=[float(v) for v in mesh.extents],
            extents_after=[float(v) for v in mesh.extents])

        if mesh.body_count <= 1:
            return mesh, report

        parts = mesh.split(only_watertight=False)
        report.components_before = report.components_after = len(parts)
        if len(parts) <= 1:
            return mesh, report

        diagonals = [float(np.linalg.norm(p.extents)) for p in parts]
        largest = max(diagonals)
        if largest <= 0:
            return mesh, report

        keep, dropped = [], []
        for part, diag in zip(parts, diagonals):
            if diag >= ratio * largest:
                keep.append(part)
            else:
                dropped.append({"faces": int(len(part.faces)),
                                "diagonal": round(diag, 4),
                                "share_of_part": round(diag / largest, 6)})
        if not dropped or not keep:
            return mesh, report

        out = keep[0] if len(keep) == 1 else trimesh.util.concatenate(keep)
        report.dropped = dropped
        report.components_after = len(keep)
        report.faces_after = int(len(out.faces))
        report.extents_after = [float(v) for v in out.extents]
        if log:
            log(f"    {report.summary()}")
        return out, report


class MeshRepairError(ValueError):
    """An open mesh that automatic repair could not close.

    ``str(exc)`` is written for the person who uploaded the file; ``report``
    carries the numbers for the log.
    """

    def __init__(self, message: str, report: "RepairReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass
class RepairReport:
    """What was wrong, what was done about it, and whether it worked."""

    before: dict
    after: dict = field(default_factory=dict)
    attempted: bool = False
    repaired: bool = False
    steps: list = field(default_factory=list)
    rejected_because: str = ""
    repair_allowed: bool = True
    #: set when topological repair failed and the voxel fallback ran
    solidify: "SolidifyReport | None" = None

    @property
    def approximated(self) -> bool:
        """True when the surface was rebuilt rather than repaired.

        The distinction matters to anyone reading a downstream number: a
        repaired mesh is the part, a solidified one is a rasterisation of it
        to within ``solidify.error_mm()``.
        """
        return bool(self.solidify and self.solidify.ok)

    @property
    def ok(self) -> bool:
        """True when the mesh is now a closed, outward-wound, positive solid."""
        state = self.after or self.before
        return (bool(state.get("watertight"))
                and bool(state.get("winding_consistent"))
                and float(state.get("volume", 0.0)) > 0.0
                and not self.rejected_because)

    def summary(self) -> str:
        b, a = self.before, (self.after or self.before)
        if not self.attempted:
            return "mesh is a closed solid; no repair needed"
        if not self.ok:
            if self.solidify and not self.solidify.ok:
                return f"repair and {self.solidify.summary()}"
            return (f"repair failed: {self.rejected_because}"
                    if self.rejected_because else
                    f"repair failed: {a.get('open_edges', 0):,} open edges remain")
        if self.approximated:
            return (f"closed by {self.solidify.summary()} "
                    f"(the surface is an approximation, not the original)")
        return (f"repaired: {b['open_edges']:,} -> {a['open_edges']:,} open edges, "
                f"{b['faces']:,} -> {a['faces']:,} faces "
                f"({', '.join(self.steps) or 'no change'})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        d["approximated"] = self.approximated
        d["solidify"] = self.solidify.to_dict() if self.solidify else None
        d["summary"] = self.summary()
        return d


class MeshRepair:
    """Detect and repair the mesh defects that break solid voxelisation."""

    #: filling one ring of holes can expose another; keep going, but not forever
    MAX_FILL_PASSES = 3
    #: a solid may not exceed its own convex hull (slack for voxel-free rounding)
    HULL_TOLERANCE = 1.02

    # -- diagnosis --------------------------------------------------------- #
    @staticmethod
    def open_edge_count(mesh: trimesh.Trimesh) -> int:
        """Edges used by exactly one face — the boundary of the surface.

        Zero on a closed manifold. Counted directly rather than inferred from
        ``is_watertight`` because the number is what makes the difference
        between "two triangles did not knit" and "this is half a part" legible
        in a log or an error message.
        """
        e = mesh.edges_sorted
        if len(e) == 0:
            return 0
        # pack each sorted (i, j) pair into one integer so np.unique stays 1-D
        key = e[:, 0].astype(np.int64) * (len(mesh.vertices) + 1) + e[:, 1]
        _, counts = np.unique(key, return_counts=True)
        return int((counts == 1).sum())

    @classmethod
    def diagnose(cls, mesh: trimesh.Trimesh) -> dict:
        return {
            "watertight": bool(mesh.is_watertight),
            "winding_consistent": bool(mesh.is_winding_consistent),
            "open_edges": cls.open_edge_count(mesh),
            "faces": int(len(mesh.faces)),
            "vertices": int(len(mesh.vertices)),
            "bodies": int(mesh.body_count),
            "volume": float(mesh.volume),
        }

    # -- repair ------------------------------------------------------------ #
    @classmethod
    def repair(cls, mesh: trimesh.Trimesh, *, log=None) -> tuple[trimesh.Trimesh, RepairReport]:
        """Return ``(mesh, report)``, working on a copy.

        Steps run in the order that makes each one cheap for the next: weld
        first, because welding alone closes the majority of "open" STLs and
        every later step then has fewer faces and real adjacency to work with;
        drop the junk faces; agree the winding; only then invent geometry by
        filling what is left.

        The mesh is returned whether or not the repair succeeded — check
        ``report.ok``. :meth:`ensure_solid` is the calling convention that
        turns a failure into an error.
        """
        report = RepairReport(before=cls.diagnose(mesh))
        report.attempted = True
        work = mesh.copy()

        for name, action in cls._steps():
            before = cls._fingerprint(work)
            try:
                action(work)
            except Exception as exc:               # a step that cannot run is
                if log:                            # not fatal; the next may
                    log(f"      repair step {name!r} skipped: {exc}")
                continue
            if cls._fingerprint(work) != before:
                report.steps.append(name)
                if log:
                    log(f"      {name}")

        # filling can expose a second ring of boundary loops behind the first
        for _ in range(cls.MAX_FILL_PASSES - 1):
            if work.is_watertight:
                break
            before = cls._fingerprint(work)
            try:
                work.merge_vertices()
                trimesh.repair.fill_holes(work)
            except Exception:
                break
            if cls._fingerprint(work) == before:
                break
            if "fill holes" not in report.steps:
                report.steps.append("fill holes")

        # normals last: they are only meaningful once the surface is closed,
        # and fix_inversion needs a signed volume to know which way is out
        for name, action in cls._final_steps():
            before = cls._fingerprint(work)
            try:
                action(work)
            except Exception:
                continue
            if cls._fingerprint(work) != before and name not in report.steps:
                report.steps.append(name)
                if log:
                    log(f"      {name}")

        report.after = cls.diagnose(work)
        report.rejected_because = cls._validate(work, report)
        report.repaired = report.ok
        return (work if report.ok else mesh), report

    @staticmethod
    def _steps():
        """(label, action) pairs applied in order. Each mutates in place."""
        return (
            ("drop non-finite vertices", lambda m: m.remove_infinite_values()),
            ("weld duplicate vertices", lambda m: m.merge_vertices()),
            ("drop degenerate faces",
             lambda m: m.update_faces(m.nondegenerate_faces())),
            ("drop duplicate faces", lambda m: m.update_faces(m.unique_faces())),
            ("drop unreferenced vertices", lambda m: m.remove_unreferenced_vertices()),
            ("fix winding", lambda m: trimesh.repair.fix_winding(m)),
            ("fill holes", lambda m: trimesh.repair.fill_holes(m)),
        )

    @staticmethod
    def _final_steps():
        return (
            ("fix normals", lambda m: trimesh.repair.fix_normals(m)),
            ("flip inside-out solid", lambda m: trimesh.repair.fix_inversion(m)),
        )

    @staticmethod
    def _fingerprint(mesh: trimesh.Trimesh) -> tuple:
        """Cheap "did that step change anything" probe.

        The volume of an open mesh is meaningless and can be NaN, which never
        compares equal to itself — so it is normalised, or every step would
        report a change it did not make.
        """
        vol = float(mesh.volume)
        return (len(mesh.faces), len(mesh.vertices),
                bool(mesh.is_winding_consistent),
                round(vol, 9) if np.isfinite(vol) else 0.0)

    @classmethod
    def _validate(cls, mesh: trimesh.Trimesh, report: RepairReport) -> str:
        """Return "" if the repaired mesh is a plausible solid, else the reason.

        ``fill_holes`` is happy to cap a large opening with a flat lid, which
        yields a watertight mesh that is not the part. These are the cheap
        invariants a real solid cannot violate.
        """
        after = report.after
        if not after["watertight"]:
            return ""            # plain failure to close; reported as such
        if not np.isfinite(after["volume"]) or after["volume"] <= 0:
            return "the closed mesh has no positive volume"
        if after["faces"] == 0:
            return "no faces survived the repair"
        try:
            hull = float(mesh.convex_hull.volume)
        except Exception:
            return ""            # cannot check; the volume test already passed
        if hull > 0 and after["volume"] > hull * cls.HULL_TOLERANCE:
            return ("the filled solid is larger than its own convex hull, so "
                    "hole filling closed the surface in the wrong place")
        return ""

    # -- the calling convention -------------------------------------------- #
    @classmethod
    def ensure_solid(cls, mesh: trimesh.Trimesh, *, allow_repair: bool = True,
                     allow_solidify: bool = True, solidify_pitch: float | None = None,
                     log=None) -> tuple[trimesh.Trimesh, RepairReport]:
        """Return a closed, consistently wound mesh, repairing it if needed.

        A mesh that is already a valid solid is returned untouched and nothing
        is logged — the whole path is one cached ``is_watertight`` lookup, so
        valid input behaves exactly as it did before this module existed.

        Topological repair is tried first and preferred, because it preserves
        the exact surface. Only when that fails — a surface patch, an assembly
        of open shells, anything with a real opening rather than an unknit seam
        — does :class:`MeshSolidify` rasterise the geometry into a closed
        approximation. The fallback is a last resort, not a shortcut: it is
        reached only where this method used to raise, and the approximation it
        made is recorded in ``report.solidify`` for the caller to report.

        :raises MeshRepairError: the mesh is open and cannot be closed at all.
        """
        state = cls.diagnose(mesh)
        if (state["watertight"] and state["winding_consistent"]
                and state["volume"] > 0):
            return mesh, RepairReport(before=state, after=state)

        if not allow_repair:
            report = RepairReport(before=state, repair_allowed=False)
            raise MeshRepairError(cls._explain(report), report)

        if log:
            log(f"    {cls._defect_line(state)}; attempting repair")
        fixed, report = cls.repair(mesh, log=log)
        if report.ok:
            if log:
                log(f"    {report.summary()}")
            return fixed, report

        if not allow_solidify:
            raise MeshRepairError(cls._explain(report), report)

        if log:
            log(f"    {report.summary()}; falling back to voxel remesh")
        solid, solidify = MeshSolidify.solidify(mesh, pitch=solidify_pitch, log=log)
        report.solidify = solidify
        if not solidify.ok:
            raise MeshRepairError(cls._explain(report), report)
        report.after = cls.diagnose(solid)
        report.steps.append("voxel remesh")
        report.repaired = True
        return solid, report

    # -- user-facing wording ------------------------------------------------ #
    @staticmethod
    def _defect_line(state: dict) -> str:
        """The one-clause diagnosis, for a log line or the head of an error."""
        if state["open_edges"]:
            return (f"mesh is not a closed solid "
                    f"({state['open_edges']:,} open edges)")
        if not state["winding_consistent"]:
            return "mesh is closed but its face windings disagree"
        if state["volume"] <= 0:
            return "mesh is closed but wound inside-out (negative volume)"
        return "mesh needs no repair"

    @classmethod
    def describe_defect(cls, mesh: trimesh.Trimesh) -> str:
        """Why this mesh cannot be voxelised, phrased for the person who sent it."""
        return cls._explain(RepairReport(before=cls.diagnose(mesh)))

    @classmethod
    def _explain(cls, report: RepairReport) -> str:
        """One paragraph a person can act on, not a stack trace."""
        b = report.before
        a = report.after or b
        advice = ("Repair it as a closed solid in your CAD tool — Meshmixer's "
                  "\"Make Solid\", netfabb's repair, or Blender's 3D-Print "
                  "toolbox all do this — then try it again.")

        if report.solidify is not None and not report.solidify.ok:
            return (f"This STL is not a closed solid ({a['open_edges']:,} open "
                    f"edges across {a['bodies']:,} "
                    f"{'body' if a['bodies'] == 1 else 'bodies'}), automatic "
                    f"repair could not close it, and rebuilding it as a solid "
                    f"also failed: {report.solidify.rejected_because}. {advice}")

        if report.rejected_because:
            return (f"This STL could not be repaired safely: {report.rejected_because}. "
                    f"The automatic fix was discarded rather than used, because "
                    f"nesting the wrong solid would give you wrong numbers. {advice}")

        if not report.attempted:
            disabled = ("" if report.repair_allowed
                        else " Automatic repair is switched off for this run.")
            return (f"This STL cannot be voxelised as a solid: "
                    f"{cls._defect_line(b)}.{disabled} {advice}")

        if a["open_edges"] >= b["open_edges"] > 0:
            return (f"This STL is not a closed solid and automatic repair could not "
                    f"close it: {a['open_edges']:,} open edges remain across "
                    f"{a['bodies']:,} separate {'body' if a['bodies'] == 1 else 'bodies'}. "
                    f"That usually means the file is a surface patch or a "
                    f"zero-thickness shell rather than a solid part. {advice}")

        if a["open_edges"]:
            return (f"This STL is not a closed solid. Automatic repair closed "
                    f"{b['open_edges'] - a['open_edges']:,} of {b['open_edges']:,} open "
                    f"edges but {a['open_edges']:,} remain, so the shape cannot be "
                    f"voxelised reliably. {advice}")

        return (f"This STL could not be made usable — {cls._defect_line(a)} — "
                f"even after automatic repair. {advice}")


@dataclass
class SolidifyReport:
    """What the voxel remesh replaced, and how faithful the replacement is."""

    attempted: bool = False
    solidified: bool = False
    pitch: float = 0.0
    grid: list = field(default_factory=list)
    filled_voxels: int = 0
    surface_voxels: int = 0
    interior_voxels: int = 0
    faces_before: int = 0
    faces_after: int = 0
    volume_before: float = 0.0
    volume_after: float = 0.0
    extents_before: list = field(default_factory=list)
    extents_after: list = field(default_factory=list)
    bodies_before: int = 0
    bodies_after: int = 0
    manifold_fixes: int = 0
    rejected_because: str = ""

    @property
    def ok(self) -> bool:
        return self.solidified and not self.rejected_because

    def error_mm(self) -> float:
        """Worst-case surface deviation the remesh can introduce.

        A voxel boundary sits within half a pitch of the true surface on each
        axis, so the body diagonal of half a voxel bounds the error.
        """
        return float(self.pitch * np.sqrt(3.0) / 2.0)

    def summary(self) -> str:
        if not self.attempted:
            return "no voxel remesh needed"
        if not self.ok:
            return f"voxel remesh rejected: {self.rejected_because}"
        merged = ""
        if self.bodies_before != self.bodies_after:
            merged = (f", {self.bodies_before} shells -> "
                      f"{self.bodies_after} closed "
                      f"{'body' if self.bodies_after == 1 else 'bodies'}")
        return (f"voxel remesh at pitch {self.pitch:.3f}: "
                f"{self.faces_before:,} -> {self.faces_after:,} faces, "
                f"{self.filled_voxels:,} voxels{merged}, surface accurate to "
                f"+/-{self.error_mm():.3f} (file units)")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        d["error_mm"] = self.error_mm()
        d["summary"] = self.summary()
        return d


class MeshSolidify:
    """Rebuild an unclosable mesh as a closed solid via voxel occupancy.

    :class:`MeshRepair` fixes meshes that are *nearly* solid — unwelded
    vertices, slivers, a few unknit holes. It deliberately gives up on the
    rest, because ``fill_holes`` on a large opening invents a flat lid and the
    result is a confident lie.

    This is the other way to get a closed surface, and it is honest about what
    it does: rasterise the geometry into an occupancy grid, flood-fill the
    interior, and emit the boundary between filled and empty. The output is
    closed and 2-manifold *by construction* rather than by repair, so it works
    on input :class:`MeshRepair` cannot touch at all — an assembly of open
    shells, a part whose seams were never knit, scanner output.

    The trade is fidelity: the surface is quantised to the grid, so it is
    blocky at the pitch and every dimension is rounded outward by up to one
    voxel. That is an approximation of the part, not a repair of it, and
    :class:`SolidifyReport` states the bound (``error_mm``) so a caller can
    decide whether it is tolerable. Voxelisation is also the operation the
    nesting pipeline performs anyway, which is why quantising to a grid finer
    than the nesting pitch costs little downstream.

    It is a last resort, not a shortcut. It refuses the same lies the repair
    path does: a flat sheet is not thickened into a slab, and an opening too
    wide to bridge produces a hollow husk, which is detected and rejected
    rather than nested.

    Only trimesh and scipy, both already required: the marching-cubes path in
    ``trimesh.voxel`` needs scikit-image, so the boundary is extracted from the
    occupancy array directly instead.
    """

    #: default pitch is the bounding-box diagonal over this, so parts of very
    #: different sizes get comparable relative fidelity
    DIAGONAL_STEPS = 160.0
    #: never rasterise a grid larger than this; pitch is coarsened to fit
    MAX_VOXELS = 12_000_000
    #: slack on the hull bound, over and above the quantisation margin
    HULL_TOLERANCE = 1.05
    #: cap on the diagonal-contact closing loop (it is monotone, so it ends)
    MAX_MANIFOLD_PASSES = 12
    #: bridge openings up to this many voxels wide before flood-filling
    CLOSE_VOXELS = 1
    #: erosion depth used to tell a filled solid from a hollow shell
    LEAK_EROSION = 2

    # -- pitch -------------------------------------------------------------- #
    @classmethod
    def default_pitch(cls, mesh: trimesh.Trimesh) -> float:
        """Pitch that resolves the part well without an unbounded grid."""
        diagonal = float(np.linalg.norm(mesh.extents))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("mesh has no finite extent to rasterise")
        pitch = diagonal / cls.DIAGONAL_STEPS
        while np.prod(np.maximum(mesh.extents / pitch, 1.0)) > cls.MAX_VOXELS:
            pitch *= 1.25
        return float(pitch)

    # -- occupancy ----------------------------------------------------------- #
    @classmethod
    def occupancy(cls, mesh: trimesh.Trimesh, pitch: float,
                  close_voxels: int | None = None) -> tuple:
        """Rasterise to ``(solid, surface, transform)`` on one voxel lattice.

        ``surface`` is the voxels the triangles pass through; ``solid`` adds
        everything the surface encloses. The interior is found by flood-filling
        the *outside* from the grid border and taking the complement, which is
        the only definition that survives a mesh with several shells — parity
        along a scanline does not.

        A seam a fraction of a voxel wide would let that flood escape and
        leave nothing but a hollow shell, so the surface is dilated by
        ``close_voxels`` to bridge narrow openings, filled, then eroded back.
        Unioning the original surface afterwards keeps thin walls the erosion
        would otherwise eat. Openings genuinely wider than that are not
        bridged, by design — see :meth:`_fill_leaked`.
        """
        from scipy import ndimage

        r = cls.CLOSE_VOXELS if close_voxels is None else int(close_voxels)
        grid = mesh.voxelized(pitch=pitch)
        surface = np.asarray(grid.matrix, dtype=bool)

        pad = r + 1                            # room for the border flood
        work = np.pad(surface, pad)
        sealed = ndimage.binary_dilation(work, iterations=r) if r else work
        filled = ndimage.binary_fill_holes(sealed)
        if r:
            filled = ndimage.binary_erosion(filled, iterations=r) | work
        solid = filled[pad:-pad, pad:-pad, pad:-pad].copy()
        return solid, surface, np.asarray(grid.transform, float)

    @classmethod
    def _fill_leaked(cls, solid: np.ndarray) -> bool:
        """True when nothing survives erosion, i.e. the result is a shell.

        An opening wider than the bridging radius lets the flood fill into the
        part, and what comes back is the surface alone: closed, watertight, and
        hollow. Nesting that is not obviously wrong, which is what makes it
        dangerous — the outer envelope looks right while the volume and fill
        ratio describe a different object. A genuine solid keeps a core a few
        voxels in from its own surface; a one-voxel shell does not.
        """
        from scipy import ndimage

        return not ndimage.binary_erosion(
            solid, iterations=cls.LEAK_EROSION).any()

    # -- non-manifold repair on the grid ------------------------------------ #
    @staticmethod
    def _window(b: int, c: int, hi_b: bool, hi_c: bool) -> tuple:
        idx = [slice(None)] * 3
        idx[b] = slice(1, None) if hi_b else slice(0, -1)
        idx[c] = slice(1, None) if hi_c else slice(0, -1)
        return tuple(idx)

    @classmethod
    def close_diagonal_contacts(cls, occ: np.ndarray) -> int:
        """Fill the checkerboard cells that would make the surface non-manifold.

        Two voxels touching only along an edge — filled on one diagonal of a
        2x2 window, empty on the other — put four boundary faces on that one
        edge. The surface is then closed but not 2-manifold, ``is_watertight``
        is False, and the winding cannot be made consistent across the pinch.
        Filling one of the two empty cells widens the contact to a full face
        and removes the pinch. This only ever adds material, so it terminates.

        Returns the number of cells filled.
        """
        filled = 0
        for _ in range(cls.MAX_MANIFOLD_PASSES):
            changed = 0
            for b, c in ((0, 1), (0, 2), (1, 2)):
                lo_lo = cls._window(b, c, False, False)
                lo_hi = cls._window(b, c, False, True)
                hi_lo = cls._window(b, c, True, False)
                hi_hi = cls._window(b, c, True, True)
                a, d = occ[lo_lo], occ[hi_hi]
                p, q = occ[lo_hi], occ[hi_lo]
                pinch_ad = a & d & ~p & ~q
                pinch_pq = p & q & ~a & ~d
                n = int(pinch_ad.sum()) + int(pinch_pq.sum())
                if n:
                    occ[lo_hi] |= pinch_ad
                    occ[lo_lo] |= pinch_pq
                    changed += n
            filled += changed
            if not changed:
                break
        return filled

    # -- boundary extraction ------------------------------------------------ #
    #: corners of the face on the +axis side of a voxel, wound counter-clockwise
    #: seen from outside; the -axis table is the same loop reversed
    _POS_FACE = {
        0: ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)),
        1: ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)),
        2: ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)),
    }
    _NEG_FACE = {
        0: ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)),
        1: ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)),
        2: ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)),
    }

    @classmethod
    def boundary_mesh(cls, occ: np.ndarray, transform: np.ndarray) -> trimesh.Trimesh:
        """Surface between filled and empty voxels, as a welded triangle mesh."""
        pad = np.zeros(np.asarray(occ.shape) + 2, dtype=bool)
        pad[1:-1, 1:-1, 1:-1] = occ

        quads = []
        for axis in (0, 1, 2):
            for shift, table in ((-1, cls._POS_FACE), (+1, cls._NEG_FACE)):
                # a face exists where this voxel is filled and its neighbour is not
                exposed = pad & ~np.roll(pad, shift, axis=axis)
                idx = np.argwhere(exposed[1:-1, 1:-1, 1:-1])
                if len(idx):
                    corners = np.asarray(table[axis], dtype=np.int64)
                    quads.append(idx[:, None, :] + corners[None, :, :])
        if not quads:
            raise ValueError("occupancy grid is empty; nothing to rasterise")

        quad_corners = np.concatenate(quads, axis=0)
        unique, inverse = np.unique(quad_corners.reshape(-1, 3), axis=0,
                                    return_inverse=True)
        quad = np.asarray(inverse).reshape(-1, 4)
        faces = np.vstack([quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]])
        # a grid index addresses a voxel centre, so its low corner is half a
        # pitch back along every axis
        verts = trimesh.transform_points(unique.astype(float) - 0.5, transform)

        out = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        out.merge_vertices()
        out.update_faces(out.nondegenerate_faces())
        out.update_faces(out.unique_faces())
        out.remove_unreferenced_vertices()
        return out

    # -- the entry point ----------------------------------------------------- #
    @classmethod
    def solidify(cls, mesh: trimesh.Trimesh, *, pitch: float | None = None,
                 close_voxels: int | None = None,
                 log=None) -> tuple[trimesh.Trimesh, SolidifyReport]:
        """Return ``(mesh, report)``; the mesh is unchanged if it did not work.

        Check ``report.ok``. The original is handed back on failure so the
        caller can fall through to its own error path.
        """
        report = SolidifyReport(
            attempted=True,
            faces_before=int(len(mesh.faces)),
            volume_before=(float(mesh.volume)
                           if np.isfinite(mesh.volume) else 0.0),
            extents_before=[float(v) for v in mesh.extents],
            bodies_before=int(mesh.body_count))
        try:
            report.pitch = float(pitch) if pitch else cls.default_pitch(mesh)
        except ValueError as exc:
            report.rejected_because = str(exc)
            return mesh, report

        report.rejected_because = cls._check_has_interior(mesh, report.pitch)
        if report.rejected_because:
            return mesh, report

        if log:
            log(f"      voxel remesh at pitch {report.pitch:.3f}")
        try:
            occ, surface, transform = cls.occupancy(mesh, report.pitch,
                                                    close_voxels)
        except Exception as exc:
            report.rejected_because = f"voxelisation failed ({exc})"
            return mesh, report

        report.grid = [int(n) for n in occ.shape]
        report.surface_voxels = int(surface.sum())
        if not occ.any():
            report.rejected_because = "the part did not intersect any voxel"
            return mesh, report
        if cls._fill_leaked(occ):
            report.rejected_because = (
                "the surface has an opening wider than the voxel grid, so the "
                "fill escaped through it and left a hollow shell rather than a "
                "solid; a finer pitch may close it if the gap is narrow")
            return mesh, report

        report.manifold_fixes = cls.close_diagonal_contacts(occ)
        report.filled_voxels = int(occ.sum())
        report.interior_voxels = report.filled_voxels - report.surface_voxels
        if log and report.manifold_fixes:
            log(f"      closed {report.manifold_fixes:,} non-manifold "
                f"voxel contacts")

        try:
            solid = cls.boundary_mesh(occ, transform)
            trimesh.repair.fix_winding(solid)
            trimesh.repair.fix_normals(solid)
            trimesh.repair.fix_inversion(solid)
        except Exception as exc:
            report.rejected_because = f"surface extraction failed ({exc})"
            return mesh, report

        report.faces_after = int(len(solid.faces))
        report.volume_after = float(solid.volume)
        report.extents_after = [float(v) for v in solid.extents]
        report.bodies_after = int(solid.body_count)
        report.rejected_because = cls._validate(solid, mesh, report)
        if report.rejected_because:
            return mesh, report

        report.solidified = True
        if log:
            log(f"    {report.summary()}")
        return solid, report

    #: the part must be at least this many voxels thick on its thinnest axis
    MIN_THICKNESS_VOXELS = 2.0

    @classmethod
    def _check_has_interior(cls, mesh: trimesh.Trimesh, pitch: float) -> str:
        """Return "" if there is a volume here to recover, else why there is not.

        Rasterising is only a *repair* when the input encloses something. Give
        it a flat sheet and it returns a one-voxel slab: watertight, plausible,
        and pure invention — the same failure mode as capping a large opening
        with a lid, which is exactly what this module refuses to do elsewhere.
        A surface with no thickness has to be refused, not thickened.
        """
        extents = np.asarray(mesh.extents, float)
        if not np.all(np.isfinite(extents)):
            return "the mesh has no finite bounding box"
        thinnest = float(extents.min())
        if thinnest < cls.MIN_THICKNESS_VOXELS * pitch:
            return (f"the input is flat ({thinnest:.4g} across its thinnest "
                    f"axis), so closing it would invent thickness rather than "
                    f"recover it")
        try:
            hull = float(mesh.convex_hull.volume)
        except Exception:
            return "the mesh is degenerate; it has no convex hull"
        if not np.isfinite(hull) or hull <= 0:
            return "the mesh encloses no volume to recover"
        return ""

    @classmethod
    def _validate(cls, solid: trimesh.Trimesh, source: trimesh.Trimesh,
                  report: SolidifyReport) -> str:
        """Return "" if the remesh is a usable solid, else why it is not.

        The same standard :meth:`MeshRepair._validate` applies — a solid that
        escapes its own convex hull is not the part — plus the closure the
        whole exercise is for.
        """
        if not solid.is_watertight:
            return "the extracted surface is still not closed"
        if not solid.is_winding_consistent:
            return "the extracted surface could not be wound consistently"
        if not np.isfinite(solid.volume) or solid.volume <= 0:
            return "the extracted solid has no positive volume"
        # a multi-body result is not an error: parity voxelisation is defined
        # per closed shell, so an assembly rasterises and nests correctly as
        # one rigid group. It is recorded, not refused.
        try:
            hull = source.convex_hull
            hull_volume, hull_area = float(hull.volume), float(hull.area)
        except Exception:
            return ""
        if hull_volume <= 0:
            return ""
        # The rasterised surface sits up to half a voxel diagonal outside the
        # real one, so it must fit not in the hull but in the hull grown by
        # that much: the first Steiner term V + A*r is the bound, and it is
        # what keeps this check meaningful at a coarse pitch instead of
        # rejecting every legitimate rebuild. A fill that escaped the surface
        # overshoots it by far more than a quantisation margin.
        allowed = (hull_volume + hull_area * report.error_mm()) * cls.HULL_TOLERANCE
        if report.volume_after > allowed:
            return (f"the rasterised solid ({report.volume_after:,.0f}) is "
                    f"larger than the input's convex hull allows "
                    f"({allowed:,.0f}), so the fill escaped the surface")
        return ""
