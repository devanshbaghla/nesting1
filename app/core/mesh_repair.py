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

__all__ = ["MeshRepairError", "RepairReport", "MeshRepair"]


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
            return (f"repair failed: {self.rejected_because}"
                    if self.rejected_because else
                    f"repair failed: {a.get('open_edges', 0):,} open edges remain")
        return (f"repaired: {b['open_edges']:,} -> {a['open_edges']:,} open edges, "
                f"{b['faces']:,} -> {a['faces']:,} faces "
                f"({', '.join(self.steps) or 'no change'})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
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
                     log=None) -> tuple[trimesh.Trimesh, RepairReport]:
        """Return a closed, consistently wound mesh, repairing it if needed.

        A mesh that is already a valid solid is returned untouched and nothing
        is logged — the whole path is one cached ``is_watertight`` lookup, so
        valid input behaves exactly as it did before this module existed.

        :raises MeshRepairError: the mesh is open and cannot be closed safely.
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
        if not report.ok:
            raise MeshRepairError(cls._explain(report), report)
        if log:
            log(f"    {report.summary()}")
        return fixed, report

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
