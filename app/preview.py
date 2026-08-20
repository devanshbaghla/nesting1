"""Pre-nesting inspection: split a part from its debris and show both.

Denoising used to happen silently on upload. That is the wrong default for
something irreversible: the rule cannot know whether the small thing beside
your part is scanner noise or the pin that belongs to it, and once dropped it
is gone from every number the run reports. So the file is classified, drawn,
and left alone until someone looks at it.

The rule
--------
A body is noise when it is small in **two** independent senses:

1. its **largest** dimension is under ``ratio`` of the **smallest** dimension of
   the object (2% by default); and
2. it carries under ``area_share`` of the part's total surface area (0.01%).

Distance does not enter into it. An earlier version required a fragment to stand
clear of the object, on the reasoning that anything touching the part is part of
the part. Real files argued the other way: a headlamp housing carries hundreds
of shells a quarter of a millimetre across, flush against the body, and they are
debris whatever they touch.

Why two tests and not one
-------------------------
A bounding box is a poor judge of importance. The dimension ratio alone, at 25%
of the object's smallest dimension, works out at a 56 mm limit on a 400 mm
housing -- and deleted 306 of its 351 bodies, a quarter of the mesh, including a
socket carrying 6,512 mm2 of surface. Nothing about its box said it mattered.

Surface area does say so, and decisively: on that part the specks measure
0.003 mm2 against the socket's 6,512 mm2, a factor of two million. So the area
test is what protects structure, and it holds even when the ratio is set badly --
which is the point, because one ratio cannot suit a 20 mm bracket and a 400 mm
housing alike.

What is deliberately *not* used: whether a body is enclosed by the part. It
sounds like the right question and cannot be answered here -- the housing is not
watertight, so a flood fill of the outside leaks through the open surface into
every cavity and reports every body as external.

The object is the largest fragment rather than the whole file -- measuring
against the whole would let a speck 200 mm off the part inflate the very box it
is being judged against. Comparing a fragment's longest axis against the
object's shortest is the strict direction, so a long thin thing that might be a
real feature survives.

``ratio`` is adjustable per request and the viewer exposes it; every fragment
reports its size, area and distance so a borderline call is visible; and nothing
is deleted until someone presses the button.

This is a different question from the one :class:`~.core.mesh_repair.
MeshDenoise` answers during a run, which compares bounding-box diagonals at a
5% threshold. That one still guards the engine. This one is what a person is
shown before they agree to it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh

__all__ = ["NOISE_RATIO", "AREA_SHARE", "PREVIEW_FACE_BUDGET",
           "SWEEP_MEMORY_BUDGET", "Fragment", "PreviewReport", "classify",
           "components", "geometry_payload", "drop_noise", "sweep_cost",
           "suggested_pitch"]

#: a fragment is noise below this share of the object's smallest dimension.
#: 2%, not 25%: the ratio is measured against the object's smallest dimension,
#: which on a 400 mm housing is 225 mm — so 25% of it is a 56 mm limit, and a
#: 56 mm limit deletes brackets and sockets along with the specks. At 2% the
#: limit on that part is 4.5 mm and the largest thing it touches carries 13 mm²
#: of surface, which is 0.003% of the part.
NOISE_RATIO = 0.02

#: ...and it must also carry less than this share of the part's total surface
#: area. This is the guard that protects structure from the ratio being wrong.
#: Debris is negligible in every sense at once: on the reference headlamp the
#: specks measure 0.003 mm² while the smallest real feature the dimension rule
#: caught measures 6,512 mm² — a factor of two million, so the two populations
#: separate cleanly however the ratio is set.
AREA_SHARE = 1e-4

#: how much the FFT translation search may allocate for one correlation,
#: in bytes. The search evaluates every lattice offset at once, so the array is
#: `A.shape + B.shape - 1` per axis — about double each dimension, eight times
#: the part's own voxel count. At the default 0.5 mm fine pitch a 294x340x423 mm
#: headlamp needs 10.8 GB of it and the run dies on an allocation failure with
#: nothing useful to say. 2 GB is a budget a workstation can actually meet.
SWEEP_MEMORY_BUDGET = 2 * 1024 ** 3

#: triangles sent to the browser before the part is decimated for drawing.
#: A preview is a picture, not the geometry the engine uses; 300k triangles is
#: already more than a screen can resolve and about 11 MB on the wire.
PREVIEW_FACE_BUDGET = 300_000


@dataclass
class Fragment:
    """One connected shell, and why it was or was not called noise."""

    index: int
    faces: int
    extents: list
    largest: float
    volume: float
    centre: list
    is_noise: bool
    #: largest dimension / object's smallest; None when there is nothing to
    #: compare against, which JSON can carry and infinity cannot
    share: float | None
    #: surface distance to the object; 0.0 for the object itself, None when it
    #: was not measured because the fragment was too big to be noise anyway
    gap: float | None = None
    #: why it was kept or dropped, in the words the viewer shows
    reason: str = ""
    #: surface area, and what share of the whole part that is. A box can be
    #: small while the body inside it is real structure; area catches that.
    area: float = 0.0
    area_share: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreviewReport:
    """Everything the viewer needs to describe the file in words."""

    filename: str = ""
    faces: int = 0
    bodies: int = 0
    extents: list = field(default_factory=list)
    object_extents: list = field(default_factory=list)
    object_smallest: float = 0.0
    threshold: float = 0.0           # absolute size below which a body is noise
    ratio: float = NOISE_RATIO
    area_ratio: float = AREA_SHARE   # ...and below this share of the surface
    area_limit: float = 0.0          # that share, in file units squared
    total_area: float = 0.0
    kept_for_area: int = 0           # small boxes spared for carrying surface
    fragments: list = field(default_factory=list)
    noise_bodies: int = 0
    noise_faces: int = 0
    watertight: bool = False
    decimated_to: int = 0            # 0 when the preview is the real geometry
    suggested_pitch: float = 0.0     # finest lattice this part can afford
    pitch_options: list = field(default_factory=list)

    @property
    def has_noise(self) -> bool:
        return self.noise_bodies > 0

    def summary(self) -> str:
        if self.bodies <= 1:
            return "single body; no loose fragments to remove"
        spared = ("" if not self.kept_for_area else
                  f" {self.kept_for_area} small "
                  f"{'body carries' if self.kept_for_area == 1 else 'bodies carry'}"
                  f" too much surface to be debris and stays.")
        if not self.has_noise:
            return (f"{self.bodies} bodies, none of them noise — nothing is both "
                    f"under {self.threshold:.3f} across and under "
                    f"{self.area_limit:.3g} of surface." + spared)
        return (f"{self.noise_bodies} of {self.bodies} bodies look like noise: "
                f"under {self.threshold:.3f} across "
                f"({100 * self.ratio:.1f}% of the object's smallest dimension, "
                f"{self.object_smallest:.2f}) and under {self.area_limit:.3g} of "
                f"surface. {self.noise_faces:,} of {self.faces:,} triangles "
                f"({100 * self.noise_faces / max(self.faces, 1):.1f}%)." + spared)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["has_noise"] = self.has_noise
        d["summary"] = self.summary()
        return d


# --------------------------------------------------------------------------- #
def load_mesh(path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def components(mesh: trimesh.Trimesh) -> list:
    """Face indices per connected shell, grouped by shared vertices.

    Connectivity is by shared *vertex*, not shared edge, and the difference is
    not academic. ``mesh.face_adjacency`` only links faces across an edge used
    by exactly two faces, so a boundary edge or a non-manifold junction breaks
    the chain. On a headlamp housing with 266,562 boundary edges that reported
    7,392 bodies with a median size of one triangle, and the largest "body" was
    0.5% of the mesh — which makes any rule measured against the largest body
    meaningless. By shared vertices the same file is 351 bodies, agreeing with
    ``mesh.body_count``.

    Also deliberately not ``mesh.split()``: splitting rebuilds each body as its
    own mesh and hands a bridging face to both, turning the drill's 10,865
    faces into 10,877. Face indices partition the original exactly, so what is
    kept is bit-for-bit what was uploaded.
    """
    if len(mesh.faces) == 0:
        return []
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components as _cc

    n_faces, n_verts = len(mesh.faces), len(mesh.vertices)
    rows = np.repeat(np.arange(n_faces), 3)
    incidence = sp.coo_matrix(
        (np.ones(len(rows), bool), (rows, mesh.faces.reshape(-1))),
        shape=(n_faces, n_verts)).tocsr()
    # vertices sharing a face are adjacent; components of that graph are shells
    _, vertex_label = _cc((incidence.T @ incidence), directed=False)
    # all three corners of a face are in one component, so any of them will do
    face_label = vertex_label[mesh.faces[:, 0]]
    order = np.argsort(face_label, kind="stable")
    bounds = np.flatnonzero(np.diff(face_label[order])) + 1
    return [np.sort(g) for g in np.split(order, bounds) if len(g)]


def _box(mesh: trimesh.Trimesh, faces: np.ndarray):
    """(extents, centre) of a face subset, without building a mesh for it."""
    pts = mesh.triangles[faces].reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return hi - lo, (lo + hi) / 2.0


def _within(outer, inner) -> bool:
    """Is ``inner``'s box wholly inside ``outer``'s? A cheap enclosure test."""
    (oe, oc), (ie, ic) = outer, inner
    return bool((np.abs(np.asarray(ic) - np.asarray(oc))
                 + np.asarray(ie) / 2.0 <= np.asarray(oe) / 2.0 + 1e-9).all())


def classify(mesh: trimesh.Trimesh, *, ratio: float = NOISE_RATIO,
             area_share: float = AREA_SHARE,
             filename: str = "") -> tuple[list, PreviewReport]:
    """Return ``(components, report)`` — face groups, tagged noise or not."""
    report = PreviewReport(filename=filename, faces=int(len(mesh.faces)),
                           ratio=float(ratio),
                           area_ratio=float(area_share),
                           extents=[float(v) for v in mesh.extents],
                           watertight=bool(mesh.is_watertight))

    comps = components(mesh)
    report.bodies = len(comps)
    if len(comps) <= 1:
        report.object_extents = [float(v) for v in mesh.extents]
        report.object_smallest = float(np.min(mesh.extents))
        report.threshold = report.object_smallest * ratio
        report.fragments = [Fragment(
            index=0, faces=int(len(mesh.faces)),
            extents=[float(v) for v in mesh.extents],
            largest=float(np.max(mesh.extents)), volume=float(mesh.volume),
            centre=[float(v) for v in mesh.bounds.mean(axis=0)],
            is_noise=False, share=None, gap=0.0,
            reason="the object").to_dict()]
        return comps or [np.arange(len(mesh.faces))], report

    boxes = [_box(mesh, c) for c in comps]
    # the object is the largest fragment: judging a speck against a box its own
    # distance inflated would let the worst debris escape the rule
    obj_i = int(np.argmax([float(np.max(e)) for e, _ in boxes]))
    smallest = float(np.min(boxes[obj_i][0]))
    report.object_extents = [float(v) for v in boxes[obj_i][0]]
    report.object_smallest = smallest
    report.threshold = smallest * ratio

    # surface area per body, from one pass over the whole mesh
    face_area = mesh.area_faces
    total_area = float(face_area.sum()) or 1.0
    areas = [float(face_area[c].sum()) for c in comps]
    report.total_area = total_area
    report.area_limit = total_area * area_share

    small = {i for i, (e, _) in enumerate(boxes)
             if i != obj_i and float(np.max(e)) < report.threshold}
    # distance is reported, never used to decide: it is the context a person
    # needs to spot a call the ratio got wrong, not part of the test
    gaps = _gaps_to(boxes, obj_i, sorted(small))

    for i, (comp, (extents, centre)) in enumerate(zip(comps, boxes)):
        largest = float(np.max(extents))
        area = areas[i]
        share_of_area = area / total_area
        is_small = i in small
        # a body carrying real surface is structure, whatever its box measures
        negligible = share_of_area < area_share
        noise = bool(is_small and negligible)
        gap = 0.0 if i == obj_i else gaps.get(i)

        if i == obj_i:
            reason = "the object"
        elif not is_small:
            reason = (f"kept: {largest:.3g} across, over the "
                      f"{report.threshold:.3g} limit")
        elif not negligible:
            reason = (f"kept: {largest:.3g} across but carries {area:.3g} of "
                      f"surface ({100 * share_of_area:.3f}% of the part)")
        else:
            where = ("inside the object"
                     if _within(boxes[obj_i], (extents, centre))
                     else f"{gap:.3g} from the main shell"
                     if gap else "against the main shell")
            reason = (f"only {largest:.3g} across and {area:.3g} of surface "
                      f"({where})")

        report.fragments.append(Fragment(
            index=i, faces=int(len(comp)),
            extents=[float(v) for v in extents],
            largest=largest, volume=0.0,
            centre=[float(v) for v in centre],
            is_noise=noise,
            share=(largest / smallest) if smallest > 0 else None,
            gap=gap, reason=reason, area=area, area_share=share_of_area,
        ).to_dict())
        if noise:
            report.noise_bodies += 1
            report.noise_faces += int(len(comp))
        elif is_small:
            report.kept_for_area += 1
    return comps, report


def _gaps_to(boxes, obj_i: int, wanted) -> dict:
    """Clear space between each wanted fragment's box and the object's.

    Box separation, not mesh distance, and that is the point: it is a *lower*
    bound on the surface gap, so a fragment is only called detached when it
    provably is. Comparing meshes point-to-point does the opposite — a 2 mm nub
    pushed through the face of a coarsely tessellated box has no vertex near
    any of the box's eight corners and measures 34 mm away while physically
    interpenetrating it. Deleting that would be deleting part of the part.

    Overlapping boxes give zero, so anything sharing space with the object is
    kept whatever the rule would otherwise say. The cost is that debris sealed
    inside a cavity is kept too; that is the safe direction, and the fragment
    list says where each one sits.
    """
    obj_extent, obj_centre = boxes[obj_i]
    out = {}
    for i in wanted:
        extent, centre = boxes[i]
        # per-axis clear space; negative means the boxes overlap on that axis
        clear = (np.abs(np.asarray(centre) - np.asarray(obj_centre))
                 - (np.asarray(extent) + np.asarray(obj_extent)) / 2.0)
        out[i] = float(np.linalg.norm(np.maximum(clear, 0.0)))
    return out


# --------------------------------------------------------------------------- #
def _for_drawing(mesh: trimesh.Trimesh, budget: int):
    """Decimate only if the browser would choke, and say by how much.

    The mesh is welded first, and that is not optional. Quadric decimation
    collapses shared edges; handed a triangle soup where every face owns its
    own three vertices there are no shared edges to collapse, so it deletes
    triangles instead of merging them. Measured on a headlamp housing: the
    unwelded route kept 27% of the surface area and missed the face target,
    while welding first kept 83% and hit it exactly, five times faster. The
    unwelded output looks like the part has been shattered.
    """
    if len(mesh.faces) <= budget:
        return mesh, 0
    try:
        welded = mesh.copy()
        welded.merge_vertices()
        small = welded.simplify_quadric_decimation(face_count=budget)
        if len(small.faces) == 0:
            return mesh, 0
        return small, len(small.faces)
    except Exception:
        return mesh, 0                # drawing slowly beats not drawing


def geometry_payload(mesh: trimesh.Trimesh, comps, report: PreviewReport,
                     budget: int = PREVIEW_FACE_BUDGET) -> dict:
    """Flat triangle soup for the viewer, split into keep and noise.

    Two buffers rather than one mesh with per-vertex colour: the viewer needs
    to hide, isolate and recolour the noise as a group, and STL has no vertex
    sharing worth preserving anyway. Positions only — normals are computed on
    the GPU side from the geometry, which halves what crosses the wire.

    Only the kept body is ever decimated. Noise fragments are small by
    definition and are the thing being looked at, so they go over whole.
    """
    noisy = sorted(f["index"] for f in report.fragments if f["is_noise"])
    noisy_set = set(noisy)
    keep = np.concatenate([c for i, c in enumerate(comps)
                           if i not in noisy_set]) if len(comps) else np.array([], int)

    part = np.asarray(mesh.triangles[keep], dtype=np.float32) if len(keep) else None
    if part is not None and len(keep) > budget:
        # submesh, not a rebuilt triangle soup: it carries the original vertex
        # sharing through, which is what decimation needs to work at all
        small, decimated = _for_drawing(mesh.submesh([keep], append=True), budget)
        if decimated:
            part = np.asarray(small.triangles, dtype=np.float32)
            report.decimated_to = decimated

    lo, hi = mesh.bounds
    return {
        "part": part.reshape(-1).tolist() if part is not None else [],
        "noise": [{"index": i,
                   "positions": np.asarray(mesh.triangles[comps[i]],
                                           dtype=np.float32).reshape(-1).tolist()}
                  for i in noisy],
        "bounds": [[float(v) for v in lo], [float(v) for v in hi]],
        "decimated_to": report.decimated_to,
    }


def drop_noise(mesh: trimesh.Trimesh, *, ratio: float = NOISE_RATIO,
               area_share: float = AREA_SHARE):
    """Return ``(mesh without its noise, report)``.

    Removal is a face mask on the original, so every surviving triangle is the
    one that was uploaded — nothing is rebuilt, rewound or re-welded on the way
    through. The report describes the file as it was, fragments still tagged,
    so the caller can say what went rather than only that something did.
    """
    comps, report = classify(mesh, ratio=ratio, area_share=area_share)
    noisy = {f["index"] for f in report.fragments if f["is_noise"]}
    if not noisy:
        return mesh, report
    drop = np.concatenate([comps[i] for i in sorted(noisy)])
    mask = np.ones(len(mesh.faces), dtype=bool)
    mask[drop] = False
    out = mesh.copy()
    out.update_faces(mask)
    out.remove_unreferenced_vertices()
    return out, report


# --------------------------------------------------------------------------- #
#  What the translation search will cost at a given lattice
# --------------------------------------------------------------------------- #
def sweep_cost(extents, pitch: float, clearance: float = 5.0) -> dict:
    """Size of one FFT correlation for a part of these extents at ``pitch``.

    Predicting this matters because the failure mode is otherwise a bare
    ``MemoryError`` from numpy, surfaced to the user as "could not process this
    file" with no hint that the lattice is the problem.

    The estimate is the output array only. ``fftconvolve`` also holds float32
    copies of both operands and their complex spectra, so real peak use is a
    few times this — which is why the budget it is compared against is well
    under physical memory.
    """
    extents = np.asarray(extents, float)
    pitch = float(pitch)
    if pitch <= 0:
        raise ValueError("pitch must be positive")
    # ClearanceGrid pads A by the dilation radius; B is the bare rotated copy
    radius = clearance + 1.5 * pitch
    pad = int(np.ceil(radius / pitch)) + 2
    a = np.ceil(extents / pitch) + 2 + 2 * pad
    b = np.ceil(extents / pitch) + 2
    shape = (a + b - 1).astype(np.int64)
    elements = float(np.prod(shape.astype(float)))
    return {"pitch": pitch,
            "shape": [int(v) for v in shape],
            "elements": elements,
            "bytes": elements * 4.0,
            "within_budget": elements * 4.0 <= SWEEP_MEMORY_BUDGET}


def suggested_pitch(extents, clearance: float = 5.0,
                    budget: int = SWEEP_MEMORY_BUDGET) -> float:
    """The finest lattice whose correlation still fits ``budget``.

    Rounded up to a readable step so the number offered to a user is 1.5 rather
    than 1.4372, and floored at the engine's own default — there is no reason to
    coarsen a small part just because the estimate says it could afford to.
    """
    for pitch in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0):
        if sweep_cost(extents, pitch, clearance)["within_budget"]:
            return pitch
    return 24.0
