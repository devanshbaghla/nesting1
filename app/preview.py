"""Pre-nesting inspection: split a part from its debris and show both.

Denoising used to happen silently on upload. That is the wrong default for
something irreversible: the rule cannot know whether the small thing beside
your part is scanner noise or the pin that belongs to it, and once dropped it
is gone from every number the run reports. So the file is classified, drawn,
and left alone until someone looks at it.

The rule, per spec
------------------
A fragment is noise when **both** hold:

1. it is **detached** from the object — its surface stands clear of it by more
   than a touch tolerance; and
2. its **largest** dimension is under ``ratio`` (25% by default) of the
   **smallest** dimension of the object.

Size alone is not enough, and the drill shows why. It arrives as 23 shells,
ten of which sit between 0.02 mm and 0.8 mm from the housing — a chuck, a
trigger, screws, all modelled as their own shells and all touching the body
they belong to. Anything abutting the object is part of the object however
small it is, so the size test is only asked about pieces that stand apart. The
genuine debris on that part sits 13 mm to 36 mm away.

Both dimensions are axis-aligned box measurements, and "the object" is the
largest fragment rather than the whole file — measuring against the whole
would let a speck 200 mm off the part inflate the very box it is being judged
against, so the further away the debris, the less likely the rule would be to
catch it.

Comparing a fragment's longest axis against the object's shortest is
deliberately strict in the safe direction: a sliver has to be small against
the part's *thinnest* section before it is called debris, so a long thin thing
that might be a real feature survives.

Where the touch tolerance lands is a judgement, not a fact, so every fragment
carries its measured gap into the report and the viewer prints it. A borderline
call is then visible rather than silent.

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
from scipy.spatial import cKDTree

__all__ = ["NOISE_RATIO", "TOUCH_RATIO", "PREVIEW_FACE_BUDGET", "Fragment", "PreviewReport",
           "classify", "geometry_payload", "drop_noise", "attached_set"]

#: a fragment is noise below this share of the object's smallest dimension
NOISE_RATIO = 0.25

#: a fragment closer than this share of the object's smallest dimension counts
#: as touching it, and is kept whatever its size. 1% is loose enough to absorb
#: the export tolerances that leave assembled shells a few hundredths apart,
#: and tight enough that real debris — which is millimetres away, not
#: hundredths — never qualifies.
TOUCH_RATIO = 0.01

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
    touch_tolerance: float = 0.0     # closer than this counts as attached
    ratio: float = NOISE_RATIO
    attached_kept: int = 0           # small, but spared for being attached
    attached_bodies: int = 0         # bodies joined to the object at all
    fragments: list = field(default_factory=list)
    noise_bodies: int = 0
    noise_faces: int = 0
    watertight: bool = False
    decimated_to: int = 0            # 0 when the preview is the real geometry

    @property
    def has_noise(self) -> bool:
        return self.noise_bodies > 0

    def summary(self) -> str:
        if self.bodies <= 1:
            return "single body; no loose fragments to remove"
        spared = ("" if not self.attached_kept else
                  f" {self.attached_kept} small "
                  f"{'body is' if self.attached_kept == 1 else 'bodies are'} "
                  f"kept for touching the object.")
        if not self.has_noise:
            return (f"{self.bodies} bodies, none of them noise — nothing is both "
                    f"detached and under {self.threshold:.3f} across."
                    + spared)
        return (f"{self.noise_bodies} of {self.bodies} bodies look like noise: "
                f"detached from the object and under {self.threshold:.3f} "
                f"across, {100 * self.ratio:.0f}% of its smallest dimension "
                f"({self.object_smallest:.2f}). {self.noise_faces:,} of "
                f"{self.faces:,} triangles." + spared)

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
             touch: float | None = None,
             filename: str = "") -> tuple[list, PreviewReport]:
    """Return ``(components, report)`` — face groups, tagged noise or not."""
    report = PreviewReport(filename=filename, faces=int(len(mesh.faces)),
                           ratio=float(ratio),
                           extents=[float(v) for v in mesh.extents],
                           watertight=bool(mesh.is_watertight))

    comps = components(mesh)
    report.bodies = len(comps)
    if len(comps) <= 1:
        report.object_extents = [float(v) for v in mesh.extents]
        report.object_smallest = float(np.min(mesh.extents))
        report.threshold = report.object_smallest * ratio
        report.touch_tolerance = report.object_smallest * (
            TOUCH_RATIO if touch is None else touch)
        report.fragments = [Fragment(
            index=0, faces=int(len(mesh.faces)),
            extents=[float(v) for v in mesh.extents],
            largest=float(np.max(mesh.extents)), volume=float(mesh.volume),
            centre=[float(v) for v in mesh.bounds.mean(axis=0)],
            is_noise=False, share=None, gap=0.0,
            reason="the object").to_dict()]
        report.attached_bodies = 1
        return comps or [np.arange(len(mesh.faces))], report

    boxes = [_box(mesh, c) for c in comps]
    # the object is the largest fragment: judging a speck against a box its own
    # distance inflated would let the worst debris escape the rule
    obj_i = int(np.argmax([float(np.max(e)) for e, _ in boxes]))
    smallest = float(np.min(boxes[obj_i][0]))
    report.object_extents = [float(v) for v in boxes[obj_i][0]]
    report.object_smallest = smallest
    report.threshold = smallest * ratio
    report.touch_tolerance = smallest * (TOUCH_RATIO if touch is None else touch)

    # Distances are only needed for fragments small enough to be at risk, so
    # the tree over the object's surface is built only if one exists — on a
    # million-face part that is seconds saved on the common clean file.
    at_risk = [i for i, (e, _) in enumerate(boxes)
               if i != obj_i and float(np.max(e)) < report.threshold]
    gaps = _gaps_to(boxes, obj_i, at_risk)

    # Anything too big to be noise is part of the object by definition, so the
    # attached set grows out from all of them at once, not just the largest.
    seeds = [i for i in range(len(comps)) if i not in set(at_risk)]
    attached = attached_set(boxes, seeds, report.touch_tolerance) if at_risk \
        else set(range(len(comps)))
    report.attached_bodies = len(attached)

    for i, (comp, (extents, centre)) in enumerate(zip(comps, boxes)):
        largest = float(np.max(extents))
        gap = 0.0 if i == obj_i else gaps.get(i)
        small = i != obj_i and largest < report.threshold
        detached = i not in attached
        noise = bool(small and detached)

        if i == obj_i:
            reason = "the object"
        elif not small:
            reason = f"too big to be noise ({largest:.3g} across)"
        elif not detached:
            # three ways to be attached, and they read differently to a person
            # deciding whether the list is trustworthy
            if _within(boxes[obj_i], (extents, centre)):
                reason = "enclosed by the object, kept"
            elif gap is not None and gap <= report.touch_tolerance:
                reason = f"touching the object ({gap:.3g} away), kept"
            else:
                reason = ("joined to the object through other bodies, kept"
                          if gap is None else
                          f"joined through other bodies ({gap:.3g} from the "
                          f"main shell), kept")
            report.attached_kept += 1
        else:
            reason = (f"free-standing, {gap:.3g} from the main shell and only "
                      f"{largest:.3g} across")

        report.fragments.append(Fragment(
            index=i, faces=int(len(comp)),
            extents=[float(v) for v in extents],
            largest=largest, volume=0.0,
            centre=[float(v) for v in centre],
            is_noise=noise,
            share=(largest / smallest) if smallest > 0 else None,
            gap=gap, reason=reason,
        ).to_dict())
        if noise:
            report.noise_bodies += 1
            report.noise_faces += int(len(comp))
    return comps, report


def _touching(boxes, i: int, j: int, tol: float) -> bool:
    """Are two boxes within ``tol`` of each other?"""
    (ei, ci), (ej, cj) = boxes[i], boxes[j]
    clear = np.abs(np.asarray(ci) - np.asarray(cj)) - (np.asarray(ei)
                                                       + np.asarray(ej)) / 2.0
    return bool(np.linalg.norm(np.maximum(clear, 0.0)) <= tol)


def attached_set(boxes, seeds, tol: float) -> set:
    """Everything reachable from ``seeds`` by hops between touching bodies.

    Attachment has to be transitive or the rule punishes assemblies. A clip
    welded to a bracket welded to the housing is attached to the part, even
    though its own box is nowhere near the housing's — measuring only against
    the largest shell would call it debris and offer to delete it. On the
    reference headlamp that is the whole difference: 46 bodies flagged when
    attachment is measured to the largest shell alone, and none at all once it
    is followed through the assembly.

    Candidate neighbours come from a tree over box centres. Two boxes can only
    be within ``tol`` if their centres are within half their diagonals plus
    ``tol``, so a radius built from the largest box never misses a pair; the
    exact box test then decides.
    """
    extents = np.array([e for e, _ in boxes], float)
    centres = np.array([c for _, c in boxes], float)
    if not len(centres):
        return set(seeds)
    half_diag = np.linalg.norm(extents, axis=1) / 2.0
    reach = float(half_diag.max()) + tol * np.sqrt(3.0)

    tree = cKDTree(centres)
    reached, frontier = set(seeds), list(seeds)
    while frontier:
        i = frontier.pop()
        for j in tree.query_ball_point(centres[i], half_diag[i] + reach):
            if j not in reached and _touching(boxes, i, j, tol):
                reached.add(j)
                frontier.append(j)
    return reached


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
               touch: float | None = None):
    """Return ``(mesh without its noise, report)``.

    Removal is a face mask on the original, so every surviving triangle is the
    one that was uploaded — nothing is rebuilt, rewound or re-welded on the way
    through. The report describes the file as it was, fragments still tagged,
    so the caller can say what went rather than only that something did.
    """
    comps, report = classify(mesh, ratio=ratio, touch=touch)
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
