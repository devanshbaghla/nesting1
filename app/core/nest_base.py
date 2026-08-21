"""
nest_base.py — STL in, top-N nested arrangements out.

    python nest_base.py part.stl                    # top 10, standard profile
    python nest_base.py part.stl -n 5 -p quick
    python nest_base.py part.stl --objective footprint
    python nest_base.py --describe

Writes one STL and one GLB per recommendation, plus a report:

    part_nest_01.stl ... part_nest_10.stl
    part_nest_01.glb ... part_nest_10.glb   two coloured nodes, orbit in a browser
    part_part.glb                           the input part on its own
    recommendations.json / .md

Why several recommendations rather than one
-------------------------------------------
The best arrangement by bounding volume is not the best by footprint, and on a
part with an interlocking feature the two can be completely different poses —
on the reference part the volume optimum interlocks two uprights at 205,440
mm3 with a 1,540 mm2 footprint, while the footprint optimum simply stacks them
end to end: 980 mm2 but 251 mm tall and the worst volume of the lot. Which one
you want depends on whether you are filling a carton or a print bed, and that
is not something the geometry can tell you. So the driver returns a diverse,
Pareto-annotated set and lets you choose.

Every returned arrangement satisfies the clearance. They differ in how they
trade the three box dimensions against each other, not in whether they are
valid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import trimesh

from .mesh_repair import (DenoiseReport, MeshDenoise, MeshRepair,
                          MeshRepairError, RepairReport)
from .nesting3d import (
    Geometry, ScanlineVoxelizer, MeshAudit, NestResult, PairNester, Preview, Refiner,
    SurfacePairDistance, SurfaceSampleCache, Validation,
)
from .nesting_factory import (
    AlgorithmRegistry, NesterFactory, NestingConfig, PROFILES,
)


# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    rank: int
    label: str
    source: str                    # which objective proposed it
    transform: list                # 4x4, copy B relative to copy A
    extents: list
    volume: float
    footprint: float
    height: float
    gap: float
    refined: bool
    pareto: bool = False
    stl: str = ""
    glb: str = ""
    verified: dict = field(default_factory=dict)
    note: str = ""

    def row(self):
        e = self.extents
        return (f"{self.rank:>2}  {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f}  "
                f"{self.volume:10,.0f}  {self.footprint:8,.0f}  "
                f"{self.gap:6.3f}  {'P' if self.pareto else ' '}  {self.label}")


# --------------------------------------------------------------------------- #
class NestingRecommender:
    """Full pipeline: audit -> self-test -> sweep -> diversify -> refine ->
    rank -> export -> verify -> report."""

    def __init__(self, config: NestingConfig | None = None):
        self.cfg = config or NesterFactory.config("standard")
        self.log_lines: list[str] = []
        self.repair_report: RepairReport | None = None
        self.denoise_report: DenoiseReport | None = None
        self.sample_cache: SurfaceSampleCache | None = None

    def _log(self, msg=""):
        self.log_lines.append(str(msg))
        if self.cfg.verbose:
            print(msg, flush=True)

    # -- 1. load ----------------------------------------------------------- #
    def load(self, path: Path) -> trimesh.Trimesh:
        """Read the STL and hand back a mesh that is safe to voxelise.

        A closed, consistently wound mesh is returned exactly as it was read.
        An open one is repaired here and nowhere else, so every stage below —
        audit, self-tests, both sweeps, refinement, export — sees one agreed
        geometry. Set ``repair=False`` in the config to refuse open meshes
        instead of fixing them, or ``solidify=False`` to keep the strict
        topological repair without the voxel-rebuild fallback.

        :raises MeshRepairError: the mesh is open and cannot be closed safely.
        """
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if self.cfg.denoise:
            # before the repair, so hole filling never tries to close a speck
            mesh, self.denoise_report = MeshDenoise.strip_stray_shells(
                mesh, ratio=self.cfg.denoise_ratio, log=self._log)
        mesh, self.repair_report = MeshRepair.ensure_solid(
            mesh, allow_repair=self.cfg.repair,
            allow_solidify=self.cfg.solidify,
            solidify_pitch=self.cfg.solidify_pitch, log=self._log)
        if self.repair_report.approximated:
            self._log("    NOTE the nested geometry is a rasterisation of the "
                      "input, so every dimension below carries that tolerance")
        return mesh

    # -- 1b. pick the lattice ---------------------------------------------- #
    #: candidate fine pitches, coarsest first. The search stops at the first
    #: one that passes, so a part that tolerates a coarse lattice pays one or
    #: two cheap gate evaluations and nothing more.
    PITCH_LADDER = (4.0, 3.0, 2.0, 1.5, 1.0, 0.5)
    #: never choose a pitch that leaves fewer than this many voxels across the
    #: part's thinnest axis, whatever the volume gate says. A thin wall can
    #: pass on total volume while being one voxel thick, and an interlock
    #: resolved at one voxel is not resolved at all.
    MIN_VOXELS_THINNEST = 12

    def choose_pitch(self, mesh) -> dict:
        """Coarsest lattice this mesh still passes the accuracy gate on.

        Cost is cubic in ``1/pitch``, and the shipped 0.5 mm default was chosen
        for one reference part rather than derived. Measured across seven parts
        the coarsest passing pitch ranges 0.5 mm to 4.0 mm, so a fixed default
        either overpays on most of them or is refused on the rest.

        The gate is **not monotone in pitch** -- hook.stl fails at 1.0 and 1.5
        and passes at 2.0, auto_05_clevis fails at 1.5, 2.0 and 3.0 and passes
        at 4.0 -- because the error depends on how the lattice happens to land
        on the features. So this searches rather than predicts, and it searches
        coarsest-first so the cheap evaluations come first.

        Nothing here loosens the gate: every candidate must pass the same 2%
        volume-error and rotation-robustness test the run would have applied
        anyway. It only stops paying for resolution the gate does not ask for.
        """
        thinnest = float(np.min(mesh.extents))
        chosen, tried = None, []
        for pitch in self.PITCH_LADDER:
            if thinnest / pitch < self.MIN_VOXELS_THINNEST:
                tried.append(f"{pitch} too coarse for a {thinnest:.1f} mm axis")
                continue
            if pitch < self.cfg.fine_pitch:
                break                      # do not refine past the request
            # Screen on the axis-aligned error first: that is one voxelisation,
            # where the full gate is seven (it re-voxelises under six random
            # rotations). Screening first made the search cheap enough to be
            # worth running -- doing the full gate per candidate cost more than
            # a coarser lattice saved on a part where nothing coarser passed.
            axis = ScanlineVoxelizer.volume_error(mesh, pitch)
            if abs(axis) >= self.cfg.voxel_tolerance:
                tried.append(f"{pitch} off by {100*axis:+.2f}%")
                continue
            # only the survivor pays for the rotation-robustness half
            v = Validation.voxeliser(mesh, pitch, tol=self.cfg.voxel_tolerance,
                                     trials=self.cfg.robustness_trials)
            if v["pass"]:
                chosen = pitch
                break
            tried.append(f"{pitch} rotated off by "
                         f"{100*v['rotated']['max_abs']:.2f}%")
        if chosen is None:
            chosen = self.cfg.fine_pitch   # fall back to what was configured
        return {"fine": chosen, "coarse": 2.0 * chosen, "rejected": tried}

    # -- 2. audit + self-tests (gates) ------------------------------------- #
    def audit(self, mesh) -> dict:
        rep = MeshAudit(mesh).assert_usable()
        self._log(f"    {rep['faces']:,} faces  fill ratio {rep['fill_ratio']:.1%}  "
                  f"{'axis-aligned' if rep['already_axis_aligned'] else 'OBB tighter'}")
        if rep["fill_ratio"] > 0.60:
            self._log("    NOTE high fill ratio, little room to interlock")
        return rep

    def self_test(self, mesh) -> dict:
        vox = Validation.voxeliser(mesh, self.cfg.fine_pitch,
                                   tol=self.cfg.voxel_tolerance,
                                   trials=self.cfg.robustness_trials)
        # gate the metric this run will use, not a fixed one
        sph = Validation.sphere_pair(n=40_000,
                                     backend=NesterFactory.backend(self.cfg))
        self._log(f"    voxeliser {100*vox['axis_aligned_error']:+.2f}% axis, "
                  f"{100*vox['rotated']['max_abs']:.2f}% rotated  "
                  f"[{'pass' if vox['pass'] else 'FAIL'}]")
        self._log(f"    sphere pair {sph['got']:.4f} vs {sph['expected']}  "
                  f"[{'pass' if sph['pass'] else 'FAIL'}]")
        if not (vox["pass"] and sph["pass"]):
            # Name the number that failed and what would admit it. The bare
            # "primitive self-tests failed" sent people hunting: setting a
            # coarse fine_pitch on its own always trips this, because the
            # tolerance is a separate knob and still at its 2% default.
            why = []
            tol = 100 * self.cfg.voxel_tolerance
            if not vox["pass"]:
                axis = 100 * vox["axis_aligned_error"]
                rot = 100 * vox["rotated"]["max_abs"]
                worst = max(abs(axis), rot)
                why.append(
                    f"the {self.cfg.fine_pitch} mm lattice reproduces this "
                    f"part's volume to {axis:+.2f}% axis-aligned and "
                    f"{rot:.2f}% under rotation, against a {tol:.0f}% "
                    f"tolerance. Use a finer fine_pitch, or accept the error "
                    f"by raising voxel_tolerance above {worst:.2f}% "
                    f"(the 'fast' profile sets 25% and skips refinement)")
            if not sph["pass"]:
                why.append(
                    f"the distance metric measured {sph['got']:.4f} on two "
                    f"spheres whose true gap is {sph['expected']}, which is a "
                    f"fault in the metric rather than in your file")
            raise RuntimeError("refusing to nest: " + "; ".join(why))
        # A relaxed gate or a skipped refinement changes what the delivered
        # numbers mean, so it is recorded with the result rather than left for
        # someone to infer from the profile name.
        caveats = []
        if self.cfg.voxel_tolerance > 0.02:
            caveats.append(
                f"the voxeliser gate was relaxed to "
                f"{100*self.cfg.voxel_tolerance:.0f}% volume error (default 2%), "
                f"and this lattice came in at "
                f"{100*vox['axis_aligned_error']:+.2f}%")
        if self.cfg.refiner == "none":
            caveats.append(
                "continuous refinement was skipped, so each gap is whatever "
                "the lattice left and is looser than the clearance requested, "
                "not squeezed to it")
        if self.cfg.fine_pitch > 1.0:
            caveats.append(
                f"the lattice is {self.cfg.fine_pitch} mm, so a reported "
                f"dimension carries roughly that quantisation")
        for c in caveats:
            self._log(f"    CAVEAT {c}")
        return {"voxeliser": vox, "sphere_pair": sph, "caveats": caveats}

    # -- 3. sweep ---------------------------------------------------------- #
    def sweep(self, mesh, pitch, orientations) -> list:
        grid = NesterFactory.clearance_grid(mesh, pitch, self.cfg)
        oracle = NesterFactory.oracle(grid)
        out = []
        for label, T in orientations:
            mB = mesh.copy(); mB.apply_transform(T)
            r = oracle.search(mB)
            if r is None:
                continue
            out.append({"label": label, "T": T,
                        "t_volume": r.t_volume, "volume": r.volume,
                        "t_area": r.t_area, "area": r.area})
        return out

    # -- 4. diversify ------------------------------------------------------ #
    def diversify(self, mesh, rows, obj_fn, top_n) -> list:
        """Pick distinct arrangements, not near-duplicates of one another.

        A raw ranking is dominated by micro-variants: the same interlock at
        +/-1.5 degrees appears five times and crowds out genuinely different
        poses. Two candidates count as the same when their rotations are within
        `diversity_angle` AND their bounding boxes agree within
        `diversity_extent` — so a shared rotation with a different translation
        (interlocked versus stacked) still survives as its own option.

        Both objectives propose candidates, because the volume-optimal and
        footprint-optimal translations for one rotation are different poses.
        """
        cands = []
        for r in rows:
            for src, t in (("volume", r["t_volume"]), ("footprint", r["t_area"])):
                mB = mesh.copy(); mB.apply_transform(r["T"])
                e = Geometry.union_extents(mesh.bounds, mB.bounds, t)
                cands.append({"label": r["label"], "source": src, "T": r["T"],
                              "t": np.asarray(t, float), "extents": e,
                              "score": obj_fn(e)})
        cands.sort(key=lambda c: c["score"])

        # de-duplicate near-identical arrangements
        kept = []
        for c in cands:
            dup = False
            for k in kept:
                same_box = np.abs(c["extents"] - k["extents"]).max() < 0.5
                same_pose = (self._rot_angle(c["T"], k["T"]) < self.cfg.diversity_angle
                             and np.abs(c["extents"] - k["extents"]).max()
                             < self.cfg.diversity_extent)
                # an identical bounding box is an identical offer, whatever the
                # rotation that produced it — mirror-equivalent poses would
                # otherwise burn two slots on the same choice
                if same_box or same_pose:
                    dup = True
                    break
            if not dup:
                kept.append(c)

        # Pareto front first, then fill by score.
        #
        # Ranking purely by one objective produces a useless list: the top
        # slots fill with micro-variants of the single best interlock and the
        # genuinely different trade-off never appears at all. On the reference
        # part a pure volume ranking never surfaces the stacked arrangement,
        # even though it has the smallest footprint of anything available.
        # Taking the non-dominated set first guarantees both ends of the
        # volume/footprint trade are offered.
        for c in kept:
            c["_v"] = float(np.prod(c["extents"]))
            c["_a"] = float(c["extents"][0] * c["extents"][1])
        front = [c for c in kept if not any(
            o["_v"] <= c["_v"] and o["_a"] <= c["_a"]
            and (o["_v"] < c["_v"] or o["_a"] < c["_a"])
            for o in kept if o is not c)]
        front.sort(key=lambda c: c["score"])
        rest = [c for c in kept if c not in front]
        rest.sort(key=lambda c: c["score"])
        return (front + rest)[:top_n]

    @staticmethod
    def _rot_angle(T1, T2) -> float:
        R = np.asarray(T1)[:3, :3] @ np.asarray(T2)[:3, :3].T
        return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

    # -- 5. refine --------------------------------------------------------- #
    def refine(self, mesh, cand, obj_fn) -> tuple:
        """Recover the slack the conservative lattice search left behind.

        Returns ``(t, gap, refined)``. With ``refiner='none'`` the lattice pose
        is kept: still feasible, just looser than necessary.
        """
        T, t0 = cand["T"], cand["t"].copy()
        strategy = NesterFactory.strategy(self.cfg)
        mB = mesh.copy(); mB.apply_transform(T)
        n = self.cfg.n_samples if strategy != "none" else 80_000
        # T is handed over so B's side of the distance queries can borrow A's
        # field through it; the factory checks that mB really is mesh moved by T
        dist = NesterFactory.distance(mesh, mB, t0, self.cfg, n_samples=n,
                                      transform=T)
        metric = NesterFactory.metric(self.cfg)

        if strategy == "none":
            return t0, float(dist.exact(t0)), False

        objective = lambda t: obj_fn(
            Geometry.union_extents(mesh.bounds, mB.bounds, t))
        guard = self.cfg.clearance + self.cfg.guard_extra
        coarse_ok = lambda t: dist.sampled(t, coarse=True) >= guard
        t = t0
        if not coarse_ok(t):
            t = Refiner.repair(t, int(np.argmax(np.abs(t))), coarse_ok)

        ref = NesterFactory.refiner(objective, coarse_ok)
        lev = Geometry.axis_leverage(
            mesh.extents, Geometry.union_extents(mesh.bounds, mB.bounds, t))
        bind, scan, free = lev[0][0], lev[1][0], lev[2][0]

        if strategy == "profile":
            span = np.arange(t[scan] - self.cfg.scan_span,
                             t[scan] + self.cfg.scan_span + 1e-9, self.cfg.scan_step)
            rows = ref.profile_sweep(t, scan, bind, span,
                                     t[bind] - 12.0, t[bind] + 6.0)
            if rows:
                t = rows[0][1]
            ref.perturb_check(t, free, (-2.0, -1.0, 1.0, 2.0), bind,
                              t[bind] - 12.0, t[bind] + 8.0)
        else:
            t, _ = ref.descend(t)

        exact_ok = lambda tt: metric(dist, tt) >= self.cfg.clearance
        tight = NesterFactory.refiner(objective, exact_ok).bisect_axis(
            t, bind, t[bind] - 1.5, t[bind], iters=9)
        if tight is not None:
            t = tight
        return t, float(dist.exact(t)), True

    # -- 6. export + verify ------------------------------------------------ #
    def _assembly(self, mesh, M):
        A = mesh.copy()
        B = mesh.copy(); B.apply_transform(M)
        if self.cfg.origin_corner:
            shift = -np.minimum(A.bounds[0], B.bounds[0])
            A.apply_translation(shift); B.apply_translation(shift)
        return trimesh.util.concatenate([A, B]), A, B

    def export_one(self, mesh, rec: Recommendation, path: Path):
        """Write the pair as STL, and as GLB for the interactive viewer.

        Both come out of one ``_assembly`` call. The GLB is written from the
        separate ``A``/``B`` copies rather than the concatenated assembly so
        each stays its own node with its own colour, and it costs nothing
        extra: the geometry is already in memory and this is serialisation
        only. Writing it here instead of re-loading the STL afterwards is what
        removed the render stage from the pipeline.
        """
        asm, A, B = self._assembly(mesh, np.array(rec.transform))
        asm.export(str(path))
        rec.stl = str(path)
        rec.glb = Preview.pair_glb([A, B], str(path.with_suffix(".glb")))

    def verify_one(self, path: str, full: bool, n_faces: int | None = None,
                   mesh=None, transform=None) -> dict:
        """Re-measure the written STL from scratch.

        The file is re-read from disk rather than trusted, which is what makes
        this catch an export or transform bug. It uses the configured distance
        backend: on 'bvh' the sampled/exact split collapses, since one exact
        answer costs less than the cheap approximation did.

        ``n_faces`` is the face count of one copy. A part that is a single
        closed body splits into exactly two components and needs no hint, but
        an assembly — or anything :class:`MeshSolidify` rasterised into several
        shells — does not, and pairing components up by adjacency would be
        guesswork. :meth:`export_one` writes copy A's faces and then copy B's,
        so with the count the boundary between them is known exactly.

        ``mesh`` and ``transform`` are an optimisation, and an optional one.
        Both copies on disk are rigid placements of the same part, so the
        distance field built during the sweep describes them too once it is
        moved into place — which removes the two full cross-queries this method
        would otherwise spend on pruning. The placement is *checked against the
        file*, never assumed: see :meth:`_verify_fields`. A file that does not
        match is exactly the export bug this stage exists to catch, so the
        mismatch is recorded and the unpruned path runs instead.
        """
        chk = trimesh.load(path)
        parts = chk.split(only_watertight=False)
        out = {"bodies": len(parts),
               "watertight": [bool(p.is_watertight) for p in parts],
               "volumes": [float(p.volume) for p in parts],
               "extents": (chk.bounds[1] - chk.bounds[0]).tolist()}

        if n_faces and len(chk.faces) == 2 * n_faces:
            A = chk.submesh([np.arange(n_faces)], append=True)
            B = chk.submesh([np.arange(n_faces, 2 * n_faces)], append=True)
            out["copies_from"] = "face index"
        elif len(parts) == 2:
            A, B = parts
            out["copies_from"] = "connectivity"
        else:
            out["copies_from"] = ""
            out["gap_note"] = (f"{len(parts)} bodies and no face count; the two "
                               f"copies could not be told apart to re-measure "
                               f"the gap")
            return out

        n = 250_000 if full else 60_000
        backend = NesterFactory.backend(self.cfg)
        kw = {}
        if backend is SurfacePairDistance:
            fa, fb, note = self._verify_fields(A, B, mesh, transform)
            kw = {"field": fa, "field_b": fb}
            out["pruned_by_field"] = bool(fa is not None or fb is not None)
            if note:
                out["field_note"] = note
        d = backend(A, B, np.zeros(3), n_samples=n, move=0.0, seed=12345, **kw)
        exact = full or backend is not SurfacePairDistance
        out["gap"] = float(d.exact(np.zeros(3)) if exact
                           else d.sampled(np.zeros(3)))
        out["gap_method"] = "exact" if exact else "sampled"
        return out

    def _verify_fields(self, A, B, mesh, transform):
        """Distance fields for the two written copies, or ``(None, None, why)``.

        The sweep's field describes the part at the origin. Copy A on disk is
        that part translated, copy B is it rotated and translated, so both are
        views of the one array — provided the file really does contain those
        placements.

        That is checked, not assumed, and it has to be: a field placed where
        the geometry is not would under-state nothing and over-state some
        distances, pruning away the very points that hold the true minimum, and
        the gap would come back too large. Since a wrong export is precisely
        what this method is here to detect, the check compares the expected
        triangle centroids against the ones read from the file and declines the
        optimisation on any disagreement.
        """
        if mesh is None or transform is None:
            return None, None, ""
        pool = SurfaceSampleCache.current()
        field = pool.field_for(mesh, self.cfg.fine_pitch) if pool else None
        if field is None:
            return None, None, "no field was built during the sweep"

        M = np.asarray(transform, float)
        # export_one writes _assembly's output, which corner-aligns the pair
        shift = np.zeros(3)
        if self.cfg.origin_corner:
            mB = mesh.copy(); mB.apply_transform(M)
            shift = -np.minimum(mesh.bounds[0], mB.bounds[0])
        place_a = trimesh.transformations.translation_matrix(shift)
        place_b = place_a @ M

        tol = 1e-4 * max(1.0, float(np.max(mesh.extents)))
        for placed, got, which in ((place_a, A, "A"), (place_b, B, "B")):
            if len(got.faces) != len(mesh.faces):
                return None, None, (f"copy {which} has {len(got.faces):,} faces, "
                                    f"the part has {len(mesh.faces):,}")
            want = trimesh.transform_points(mesh.triangles.mean(axis=1), placed)
            err = float(np.abs(want - got.triangles.mean(axis=1)).max())
            if err > tol:
                return None, None, (f"copy {which} on disk is {err:.4g} from its "
                                    f"expected placement; not reusing the field")
        return field.transformed(place_a), field.transformed(place_b), ""

    # -- 7. pareto --------------------------------------------------------- #
    @staticmethod
    def mark_pareto(recs: list[Recommendation]):
        """Flag arrangements not beaten on BOTH volume and footprint."""
        for r in recs:
            r.pareto = not any(
                (o.volume <= r.volume and o.footprint <= r.footprint
                 and (o.volume < r.volume or o.footprint < r.footprint))
                for o in recs if o is not r)

    # -- 8. report --------------------------------------------------------- #
    # -- main -------------------------------------------------------------- #
    def recommend(self, stl_path, out_dir="nest_out", top_n=None) -> list[Recommendation]:
        """Run the pipeline, reusing part A's samples across every candidate.

        The cache is opened here and nowhere deeper because this is the scope
        that owns "one part": every candidate below measures the same fixed
        part A against a differently rotated B, and leaving the block drops the
        entries so no geometry survives into the next job.
        """
        asked = (self.cfg.coarse_pitch, self.cfg.fine_pitch)
        try:
            with SurfaceSampleCache() as pool:
                self.sample_cache = pool
                return self._recommend(stl_path, out_dir, top_n)
        except (RuntimeError, MeshRepairError) as exc:
            # A coarser lattice is only worth choosing if it cannot cost an
            # answer. It can: on electric_drill.stl the 2 mm lattice put the
            # refiner on a pose it could not walk back to feasible, and the run
            # died where the 0.5 mm lattice succeeds. So the speed-up is
            # attempted, and the configured pitch is what actually has to work.
            if (not self.cfg.auto_pitch
                    or (self.cfg.coarse_pitch, self.cfg.fine_pitch) == asked):
                raise
            self._log(f"    the chosen lattice failed ({exc}); retrying at the "
                      f"requested {asked[1]} mm")
            self.cfg.coarse_pitch, self.cfg.fine_pitch = asked
            self.cfg.auto_pitch = False
            with SurfaceSampleCache() as pool:
                self.sample_cache = pool
                return self._recommend(stl_path, out_dir, top_n)

    def _recommend(self, stl_path, out_dir, top_n) -> list[Recommendation]:
        stl_path = Path(stl_path)
        if not stl_path.exists():
            raise FileNotFoundError(stl_path)
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        top_n = top_n or self.cfg.top_n
        t_all = time.time()

        self._log(f"nesting 2 x {stl_path.name}   clearance {self.cfg.clearance} "
                  f"(file units)   top {top_n}")
        self._log(NesterFactory.describe(self.cfg))

        self._log("\n[1/8] load")
        mesh = self.load(stl_path)
        if self.cfg.auto_pitch:
            pick = self.choose_pitch(mesh)
            if pick["fine"] != self.cfg.fine_pitch:
                self._log(f"    lattice: {pick['fine']} mm fine / "
                          f"{pick['coarse']} mm coarse "
                          f"(was {self.cfg.fine_pitch} / {self.cfg.coarse_pitch})")
                for why in pick["rejected"]:
                    self._log(f"      rejected {why}")
                self.cfg.fine_pitch = pick["fine"]
                self.cfg.coarse_pitch = pick["coarse"]

        self._log("[2/8] audit (gate)")
        audit = self.audit(mesh)
        self._log("[3/8] primitive self-tests (gate)")
        tests = self.self_test(mesh)

        self._log("[4/8] coarse orientation sweep")
        orients = NesterFactory.orientations(self.cfg, "coarse")
        self._log(f"    {len(orients)} orientations at pitch {self.cfg.coarse_pitch}")
        rows = self.sweep(mesh, self.cfg.coarse_pitch, orients)
        if not rows:
            raise RuntimeError("no feasible arrangement at any orientation")

        obj_fn = NesterFactory.objective(self.cfg, mesh.extents)
        best_T = min(rows, key=lambda r: obj_fn(
            Geometry.union_extents(mesh.bounds,
                                   self._rot(mesh, r["T"]).bounds, r["t_volume"])))["T"]

        self._log("[5/8] fine local sweep around the leader")
        rows += self.sweep(mesh, self.cfg.fine_pitch,
                           NesterFactory.orientations(self.cfg, "fine", best_T))

        self._log(f"[6/8] diversify -> {top_n} distinct arrangements")
        cands = self.diversify(mesh, rows, obj_fn, top_n)
        self._log(f"    {len(cands)} kept from {2*len(rows)} proposals")

        self._log("[7/8] refine, export and verify each")
        recs = []
        for i, c in enumerate(cands, 1):
            t, gap, refined = self.refine(mesh, c, obj_fn)
            M = np.eye(4); M[:3, :3] = c["T"][:3, :3]; M[:3, 3] = t
            mB = self._rot(mesh, M)
            e = Geometry.union_extents(mesh.bounds, mB.bounds, np.zeros(3))
            rec = Recommendation(
                rank=0, label=c["label"], source=c["source"],
                transform=M.tolist(), extents=e.tolist(),
                volume=float(e.prod()), footprint=float(e[0] * e[1]),
                height=float(e[2]), gap=gap, refined=refined)
            if gap < self.cfg.clearance - 1e-3:
                rec.note = f"REJECTED: gap {gap:.4f} < {self.cfg.clearance}"
                self._log(f"    {i:>2}. {rec.note}")
                continue
            recs.append(rec)
            self._log(f"    {i:>2}. {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f}  "
                      f"vol {rec.volume:10,.0f}  area {rec.footprint:8,.0f}  "
                      f"gap {gap:.3f}")

        if not recs:
            raise RuntimeError("every candidate failed the clearance check")
        recs.sort(key=lambda r: obj_fn(np.array(r.extents)))
        self.mark_pareto(recs)
        for i, r in enumerate(recs, 1):
            r.rank = i
            self.export_one(mesh, r, out_dir / f"{stl_path.stem}_nest_{i:02d}.stl")
            if self.cfg.verify:
                r.verified = self.verify_one(r.stl, full=(i <= 3),
                                             n_faces=len(mesh.faces),
                                             mesh=mesh, transform=r.transform)
            else:
                # recorded, not silently absent: a result that was never
                # checked against the file on disk should say so
                r.verified = {"skipped": True,
                              "why": "verification is off for this profile"}

        self._log("[8/8] report")
        # every recommendation already has its GLB from export_one, so there is
        # nothing left to draw here; the input part gets one too, because the
        # question "is this part even worth nesting" is answered by looking at
        # it, and that used to be what part_views was for
        part_glb = Preview.part_glb(
            mesh, str(out_dir / f"{stl_path.stem}_part.glb"))

        baselines = PairNester(mesh, self.cfg.clearance, verbose=False).baselines()
        self._write_report(out_dir, stl_path, mesh, recs, audit, tests, baselines)

        self._log(f"\n rank   bbox (mm)                 volume    footprint    gap  P  label")
        for r in recs:
            self._log(" " + r.row())
        self._log(f"\ncompleted in {time.time()-t_all:.0f}s -> {out_dir}")
        self._log(f"part model: {part_glb}")
        return recs

    @staticmethod
    def _rot(mesh, T):
        m = mesh.copy(); m.apply_transform(T); return m

    def _write_report(self, out_dir, stl_path, mesh, recs, audit, tests, baselines):
        best_naive = min(baselines.values(), key=lambda r: r["volume"])
        payload = {
            "input": str(stl_path),
            "config": self.cfg.to_dict(),
            "denoise": self.denoise_report.to_dict() if self.denoise_report else None,
            "repair": self.repair_report.to_dict() if self.repair_report else None,
            "approximated": bool(self.repair_report
                                 and self.repair_report.approximated),
            "audit": audit,
            "self_tests": tests,
            "baselines": baselines,
            "recommendations": [asdict(r) for r in recs],
            "log": self.log_lines,
        }
        (out_dir / "recommendations.json").write_text(
            json.dumps(payload, indent=2, default=str))

        L = [f"# Nested arrangements — {stl_path.name}", "",
             f"- part: {audit['faces']:,} faces, volume {audit['volume']:,.1f}, "
             f"fill ratio {audit['fill_ratio']:.1%}",
             f"- clearance: {self.cfg.clearance} (file units)",
             f"- ranked by: {self.cfg.objective}",
             f"- best naive packing: {best_naive['volume']:,.0f}", "",
             "| # | bbox | volume | footprint | height | gap | Pareto | file |",
             "|---|---|---|---|---|---|---|---|"]
        for r in recs:
            e = r.extents
            L.append(f"| {r.rank} | {e[0]:.1f} x {e[1]:.1f} x {e[2]:.1f} "
                     f"| {r.volume:,.0f} | {r.footprint:,.0f} | {e[2]:.1f} "
                     f"| {r.gap:.3f} | {'yes' if r.pareto else ''} "
                     f"| {Path(r.stl).name} |")
        b = recs[0]
        L += ["", f"Best saves {100*(1-b.volume/best_naive['volume']):.1f}% of "
                  f"bounding volume versus the best naive packing.", "",
              "## Transform for copy B, best arrangement", "", "```",
              str(np.round(np.array(b.transform), 4)), "```", "",
              "Rows marked Pareto are not beaten on both volume and footprint "
              "simultaneously; the others are strictly dominated and are kept "
              "only for comparison."]
        (out_dir / "recommendations.md").write_text("\n".join(L))


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Nest 2 copies of an STL; return the top N arrangements.")
    p.add_argument("stl", nargs="?")
    p.add_argument("-o", "--out-dir", default="nest_out")
    p.add_argument("-n", "--top", type=int, default=10)
    p.add_argument("-c", "--clearance", type=float, default=5.0)
    p.add_argument("-p", "--profile", choices=sorted(PROFILES), default="standard")
    p.add_argument("--objective", choices=sorted(AlgorithmRegistry.names("objective")),
                   default="volume")
    p.add_argument("--distance-backend",
                   choices=sorted(AlgorithmRegistry.names("distance_backend")),
                   help="sampled (default) or bvh (needs python-fcl)")
    p.add_argument("--refiner", choices=sorted(AlgorithmRegistry.names("refiner")))
    p.add_argument("--coarse-pitch", type=float)
    p.add_argument("--fine-pitch", type=float)
    p.add_argument("--samples", type=int, dest="n_samples")
    p.add_argument("--repair", dest="repair", action="store_true", default=None,
                   help="repair an open mesh before nesting (the default)")
    p.add_argument("--no-repair", dest="repair", action="store_false",
                   help="refuse an open mesh instead of repairing it")
    p.add_argument("--no-solidify", dest="solidify", action="store_false",
                   default=None,
                   help="refuse a mesh that topological repair cannot close, "
                        "instead of rebuilding it as a voxel solid")
    p.add_argument("--solidify-pitch", type=float,
                   help="voxel pitch for that rebuild (default: from part size)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--describe", action="store_true",
                   help="print the algorithm registry and exit")
    a = p.parse_args(argv)

    if a.describe:
        print(AlgorithmRegistry.describe()); return 0
    if not a.stl:
        p.error("an STL path is required (or use --describe)")

    try:
        cfg = NesterFactory.config(
            a.profile, clearance=a.clearance, objective=a.objective,
            refiner=a.refiner, coarse_pitch=a.coarse_pitch,
            fine_pitch=a.fine_pitch, n_samples=a.n_samples, repair=a.repair,
            solidify=a.solidify, solidify_pitch=a.solidify_pitch,
            top_n=a.top, distance_backend=a.distance_backend,
            verbose=not a.quiet)
    except RuntimeError as exc:              # e.g. bvh selected without fcl
        print(f"\n{exc}", file=sys.stderr)
        return 2
    try:
        NestingRecommender(cfg).recommend(a.stl, a.out_dir, a.top)
    except MeshRepairError as exc:
        # a bad input file, not a bug: say what is wrong and stop, no traceback
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
