"""
nest_base.py — STL in, top-N nested arrangements out.

    python nest_base.py part.stl                    # top 10, standard profile
    python nest_base.py part.stl -n 5 -p quick
    python nest_base.py part.stl --objective footprint
    python nest_base.py --describe

Writes one STL per recommendation plus a contact sheet and a report:

    part_nest_01.stl ... part_nest_10.stl
    part_part_views.png        single-part diagnostic
    part_contact_sheet.png     all candidates side by side
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

from .mesh_repair import MeshRepair, MeshRepairError, RepairReport
from .nesting3d import (
    Geometry, MeshAudit, NestResult, PairNester, Preview, Refiner,
    SurfacePairDistance, Validation,
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
        instead of fixing them.

        :raises MeshRepairError: the mesh is open and cannot be closed safely.
        """
        mesh = trimesh.load(path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        mesh, self.repair_report = MeshRepair.ensure_solid(
            mesh, allow_repair=self.cfg.repair, log=self._log)
        return mesh

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
                                   trials=self.cfg.robustness_trials)
        sph = Validation.sphere_pair(n=40_000)
        self._log(f"    voxeliser {100*vox['axis_aligned_error']:+.2f}% axis, "
                  f"{100*vox['rotated']['max_abs']:.2f}% rotated  "
                  f"[{'pass' if vox['pass'] else 'FAIL'}]")
        self._log(f"    sphere pair {sph['got']:.4f} vs {sph['expected']}  "
                  f"[{'pass' if sph['pass'] else 'FAIL'}]")
        if not (vox["pass"] and sph["pass"]):
            raise RuntimeError("primitive self-tests failed; refusing to nest")
        return {"voxeliser": vox, "sphere_pair": sph}

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
        dist = NesterFactory.distance(mesh, mB, t0, self.cfg, n_samples=n)
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
        asm, _, _ = self._assembly(mesh, np.array(rec.transform))
        asm.export(str(path))
        rec.stl = str(path)

    def verify_one(self, path: str, full: bool) -> dict:
        chk = trimesh.load(path)
        parts = chk.split(only_watertight=False)
        out = {"bodies": len(parts),
               "watertight": [bool(p.is_watertight) for p in parts],
               "volumes": [float(p.volume) for p in parts],
               "extents": (chk.bounds[1] - chk.bounds[0]).tolist()}
        if len(parts) == 2:
            n = 250_000 if full else 60_000
            d = SurfacePairDistance(parts[0], parts[1], np.zeros(3),
                                    n_samples=n, move=0.0, seed=12345)
            out["gap"] = float(d.exact(np.zeros(3)) if full
                               else d.sampled(np.zeros(3)))
            out["gap_method"] = "exact" if full else "sampled"
        return out

    # -- 7. pareto --------------------------------------------------------- #
    @staticmethod
    def mark_pareto(recs: list[Recommendation]):
        """Flag arrangements not beaten on BOTH volume and footprint."""
        for r in recs:
            r.pareto = not any(
                (o.volume <= r.volume and o.footprint <= r.footprint
                 and (o.volume < r.volume or o.footprint < r.footprint))
                for o in recs if o is not r)

    # -- 8. contact sheet -------------------------------------------------- #
    def contact_sheet(self, mesh, recs, path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection

        n = len(recs)
        cols = min(5, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 4.4 * rows),
                                 squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for i, rec in enumerate(recs):
            ax = axes[i // cols][i % cols]
            ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
            _, A, B = self._assembly(mesh, np.array(rec.transform))
            for m, c in ((A, "#1b9e77"), (B, "#d95f02")):
                ax.add_collection(PolyCollection(m.triangles[:, :, [1, 2]],
                                                 facecolors=c, edgecolors="none",
                                                 alpha=0.09))
            lo = np.minimum(A.bounds[0], B.bounds[0])
            hi = np.maximum(A.bounds[1], B.bounds[1])
            ax.add_patch(plt.Rectangle((lo[1], lo[2]), hi[1] - lo[1], hi[2] - lo[2],
                                       fill=False, ec="k", lw=1.0, ls="--"))
            ax.set_xlim(lo[1] - 5, hi[1] + 5); ax.set_ylim(lo[2] - 5, hi[2] + 5)
            ax.set_aspect("equal")
            e = rec.extents
            ax.set_title(f"#{rec.rank}{' *' if rec.pareto else ''}  "
                         f"{e[0]:.0f}x{e[1]:.0f}x{e[2]:.0f}\n"
                         f"{rec.volume:,.0f} mm3 | {rec.footprint:,.0f} mm2",
                         fontsize=8)
        fig.suptitle("Nested candidates (Y-Z view)   * = Pareto-optimal",
                     fontsize=11)
        plt.tight_layout(); plt.savefig(path, dpi=95); plt.close(fig)
        return str(path)

    # -- main -------------------------------------------------------------- #
    def recommend(self, stl_path, out_dir="nest_out", top_n=None) -> list[Recommendation]:
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
            r.verified = self.verify_one(r.stl, full=(i <= 3))

        self._log("[8/8] renders and report")
        Preview.part_views(mesh, str(out_dir / f"{stl_path.stem}_part_views.png"))
        sheet = self.contact_sheet(mesh, recs,
                                   out_dir / f"{stl_path.stem}_contact_sheet.png")
        best = recs[0]
        top_asm, A, B = self._assembly(mesh, np.array(best.transform))
        Preview.render(top_asm.split(only_watertight=False),
                       str(out_dir / f"{stl_path.stem}_best_preview.png"),
                       title=f"best by {self.cfg.objective}: "
                             f"{best.volume:,.0f} mm3, gap {best.gap:.3f}")

        nester = PairNester(mesh, self.cfg.clearance, verbose=False)
        baselines = nester.baselines()
        self._write_report(out_dir, stl_path, mesh, recs, audit, tests, baselines)

        self._log(f"\n rank   bbox (mm)                 volume    footprint    gap  P  label")
        for r in recs:
            self._log(" " + r.row())
        self._log(f"\ncompleted in {time.time()-t_all:.0f}s -> {out_dir}")
        self._log(f"contact sheet: {sheet}")
        return recs

    @staticmethod
    def _rot(mesh, T):
        m = mesh.copy(); m.apply_transform(T); return m

    def _write_report(self, out_dir, stl_path, mesh, recs, audit, tests, baselines):
        best_naive = min(baselines.values(), key=lambda r: r["volume"])
        payload = {
            "input": str(stl_path),
            "config": self.cfg.to_dict(),
            "repair": self.repair_report.to_dict() if self.repair_report else None,
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
    p.add_argument("--refiner", choices=sorted(AlgorithmRegistry.names("refiner")))
    p.add_argument("--coarse-pitch", type=float)
    p.add_argument("--fine-pitch", type=float)
    p.add_argument("--samples", type=int, dest="n_samples")
    p.add_argument("--repair", dest="repair", action="store_true", default=None,
                   help="repair an open mesh before nesting (the default)")
    p.add_argument("--no-repair", dest="repair", action="store_false",
                   help="refuse an open mesh instead of repairing it")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--describe", action="store_true",
                   help="print the algorithm registry and exit")
    a = p.parse_args(argv)

    if a.describe:
        print(AlgorithmRegistry.describe()); return 0
    if not a.stl:
        p.error("an STL path is required (or use --describe)")

    cfg = NesterFactory.config(
        a.profile, clearance=a.clearance, objective=a.objective,
        refiner=a.refiner, coarse_pitch=a.coarse_pitch, fine_pitch=a.fine_pitch,
        n_samples=a.n_samples, repair=a.repair, top_n=a.top,
        verbose=not a.quiet)
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
