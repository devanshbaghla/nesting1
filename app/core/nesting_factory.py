"""
nesting_factory.py — registry and factory for the algorithms in :mod:`nesting3d`.

Two jobs:

* :class:`AlgorithmRegistry` is a catalogue. Every algorithm is registered with
  a status — ``used`` (in the delivered path), ``reference`` (kept to
  cross-validate a faster one), or ``rejected`` (measured and discarded, with
  the number that killed it). Being able to enumerate what was tried and why it
  lost is part of trusting the result.

* :class:`NesterFactory` builds configured components from a profile, so the
  driver never hard-codes an implementation. Swapping the voxeliser, the
  distance metric, the refinement strategy or the objective is a config change.

The objective is genuinely pluggable, which matters here: minimum volume and
minimum footprint are different arrangements for the same part, and neither is
universally correct. Optimising volume when the real constraint is bed area
gives the wrong answer confidently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable

import numpy as np
import trimesh

from .nesting3d import (
    DISABLED_ALGORITHMS, algorithm_enabled,
    HAVE_FCL, BVHPairDistance, ClearanceGrid, Geometry, MeshAudit,
    OrientationSet, Preview, Refiner, ScanlineVoxelizer, SurfaceDistanceField,
    SurfacePairDistance, SurfaceSampleCache, TranslationOracle, Validation,
)

__all__ = ["AlgorithmRegistry", "NestingConfig", "NesterFactory", "OBJECTIVES"]


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
class AlgorithmRegistry:
    """Catalogue of every algorithm, keyed by category and name."""

    _reg: dict[str, dict[str, dict]] = {}

    @classmethod
    def add(cls, category: str, name: str, impl, *, status: str = "used",
            note: str = ""):
        cls._reg.setdefault(category, {})[name] = {
            "impl": impl, "status": status, "note": note}
        return impl

    @classmethod
    def get(cls, category: str, name: str):
        try:
            return cls._reg[category][name]["impl"]
        except KeyError:
            avail = ", ".join(cls.names(category)) or "<none>"
            raise KeyError(f"no {category!r} named {name!r}; available: {avail}")

    @classmethod
    def names(cls, category: str) -> list[str]:
        """Every registered name, including any that are switched off."""
        return sorted(cls._reg.get(category, {}))

    @classmethod
    def enabled_names(cls, category: str) -> list[str]:
        """Names a caller may actually select.

        Offering a switched-off algorithm in a UI and then refusing it on
        submit is a trap: the form defaulted to 'descend' and every upload was
        rejected with the reason it was disabled. Anything user-facing lists
        from here.
        """
        from .nesting3d import algorithm_enabled
        return [n for n in cls.names(category) if algorithm_enabled(n)]

    @classmethod
    def categories(cls) -> list[str]:
        return sorted(cls._reg)

    @classmethod
    def status_of(cls, category: str, name: str) -> str:
        """Registered status, overridden to 'disabled' by the switchboard."""
        from .nesting3d import algorithm_enabled
        meta = cls._reg.get(category, {}).get(name)
        if meta is None:
            return "unknown"
        return meta["status"] if algorithm_enabled(name) else "disabled"

    @classmethod
    def catalogue(cls, status: str | None = None) -> list[tuple]:
        rows = []
        for cat in cls.categories():
            for name, meta in sorted(cls._reg[cat].items()):
                live = cls.status_of(cat, name)
                if status and live != status:
                    continue
                rows.append((cat, name, live, meta["note"]))
        return rows

    @classmethod
    def describe(cls, status: str | None = None) -> str:
        rows = cls.catalogue(status)
        w = max((len(r[1]) for r in rows), default=10)
        out = []
        cur = None
        for cat, name, st, note in rows:
            if cat != cur:
                out.append(f"\n[{cat}]")
                cur = cat
            out.append(f"  {name:<{w}}  {st:<9}  {note}")
        return "\n".join(out).lstrip("\n")


# -- voxelisation ----------------------------------------------------------- #
AlgorithmRegistry.add("voxelizer", "scanline", ScanlineVoxelizer.rasterize,
                      note="z-scanline parity fill; 0.18% volume error at 0.5mm")
AlgorithmRegistry.add(
    "voxelizer", "trimesh_fill",
    lambda mesh, pitch: (mesh.voxelized(pitch).fill().matrix, None),
    status="rejected",
    note="surface voxelisation + flood fill; +79.6% on 3mm walls at 1mm pitch")

# -- clearance -------------------------------------------------------------- #
AlgorithmRegistry.add("clearance", "edt_dilation", ClearanceGrid,
                      note="Euclidean distance transform of free space")

# -- translation search ----------------------------------------------------- #
AlgorithmRegistry.add("oracle", "fft_correlation", TranslationOracle,
                      note="one FFT evaluates every lattice translation")

# -- orientation generators ------------------------------------------------- #
AlgorithmRegistry.add("orientations", "z_family", OrientationSet.z_family,
                      note="Rz sweep x optional X-flip; for slender parts")
AlgorithmRegistry.add("orientations", "so3", OrientationSet.so3,
                      note="ZXZ Euler grid; unbiased but ~15x the cost")
AlgorithmRegistry.add("orientations", "local", OrientationSet.around,
                      note="fine Rz refinement around a winner")

# -- distance backends ------------------------------------------------------ #
AlgorithmRegistry.add("distance_backend", "sampled", SurfacePairDistance,
                      note="dense surface samples + KD-tree; ~164 ms per "
                           "evaluation on the reference part")
AlgorithmRegistry.add("distance_backend", "bvh", BVHPairDistance,
                      status="used" if HAVE_FCL else "unavailable",
                      note="FCL hierarchy traversal; 2.4 ms per evaluation, "
                           "agrees to 9e-16 mm" +
                           ("" if HAVE_FCL else " — needs `pip install python-fcl`"))

# -- distance metrics ------------------------------------------------------- #
AlgorithmRegistry.add("distance", "sampled",
                      lambda d, t: d.sampled(t), note="KD-tree point-to-point; >= true gap")
AlgorithmRegistry.add("distance", "exact",
                      lambda d, t: d.exact(t),
                      note="k-NN sample -> owning face + Ericson point-triangle")
AlgorithmRegistry.add("distance", "reference",
                      lambda d, t: d.exact_reference(t), status="reference",
                      note="radius query over centroids; 26s vs 2s, used to cross-check")

# -- refinement strategies -------------------------------------------------- #
AlgorithmRegistry.add("refiner", "profile", "profile",
                      note="scan one axis, bisect the binding axis; finds cliffs")
AlgorithmRegistry.add("refiner", "descend", "descend",
                      note="coordinate descent; faster, smooth feasible sets only")
AlgorithmRegistry.add("refiner", "none", "none",
                      note="keep the conservative lattice pose; feasible but loose")

# -- validation ------------------------------------------------------------- #
AlgorithmRegistry.add("validation", "voxeliser", Validation.voxeliser,
                      note="analytic volume + random-orientation robustness")
AlgorithmRegistry.add("validation", "sphere_pair", Validation.sphere_pair,
                      note="two spheres; gap is analytically d - 2r")
AlgorithmRegistry.add("validation", "distance_agreement",
                      Validation.distance_agreement,
                      note="fast metric vs independent reference metric")
AlgorithmRegistry.add("validation", "clearance_met", Validation.clearance_met,
                      note="final gate on the delivered pose")

# -- model export ----------------------------------------------------------- #
AlgorithmRegistry.add("export", "part_glb", Preview.part_glb,
                      note="the input part as glTF-binary, for inspection")
AlgorithmRegistry.add("export", "pair_glb", Preview.pair_glb,
                      note="nested pair as glTF-binary; one node per copy")


# --------------------------------------------------------------------------- #
#  Objectives
# --------------------------------------------------------------------------- #
def _volume(e, ref):
    return float(np.prod(e))


def _footprint(e, ref):
    return float(e[0] * e[1])


def _balanced(e, ref):
    """Both objectives, each normalised by the single part, summed.

    Volume and footprint conflict: on the reference part the volume optimum
    interlocks two uprights, while the footprint optimum simply stacks them
    end to end at twice the height. This scores them on one scale so a
    compromise arrangement can win.
    """
    return float(np.prod(e) / np.prod(ref) + (e[0] * e[1]) / (ref[0] * ref[1]))


def _height(e, ref):
    return float(e[2])


OBJECTIVES: dict[str, Callable] = {
    "volume": _volume, "footprint": _footprint,
    "balanced": _balanced, "height": _height,
}
for _n, _f in OBJECTIVES.items():
    AlgorithmRegistry.add("objective", _n, _f,
                          note=(_f.__doc__ or "").strip().split("\n")[0])


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class NestingConfig:
    clearance: float = 5.0
    coarse_pitch: float = 1.0
    fine_pitch: float = 0.5
    coarse_step: float = 15.0
    so3_step: float | None = None
    fine_span: float = 6.0
    fine_step: float = 1.5

    voxelizer: str = "scanline"
    orientations: str = "z_family"
    distance: str = "exact"
    #: which distance implementation to build — "sampled" (no extra
    #: dependency) or "bvh" (needs python-fcl, ~69x faster per evaluation)
    distance_backend: str = "sampled"
    refiner: str = "profile"
    objective: str = "volume"

    n_samples: int = 220_000
    guard_extra: float = 0.08
    scan_span: float = 1.6
    scan_step: float = 0.1

    top_n: int = 10
    diversity_angle: float = 10.0     # deg between distinct rotations
    diversity_extent: float = 3.0     # mm between distinct bounding boxes

    #: Re-read each written STL from disk and re-measure its gap with a fresh
    #: seed. It is the only check that works on the file rather than on the
    #: meshes in memory, so it is the only thing that would catch a bad export
    #: or a mis-applied transform -- an in-memory answer stays perfect through
    #: both. Measured at 3.37 s of an 11.49 s run on sample.stl.
    verify: bool = True
    #: Volume error the voxeliser gate will admit. 2% is what every accuracy
    #: claim in this engine rests on; the 'fast' profile raises it deliberately
    #: and the report says so.
    voxel_tolerance: float = 0.02
    #: Search for the coarsest lattice this part still passes the accuracy gate
    #: on, instead of always using ``fine_pitch``. Cost is cubic in 1/pitch, so
    #: in principle this is the largest speed lever in the engine.
    #:
    #: Off by default, because measured it did not pay. On sample.stl no pitch
    #: coarser than 0.5 mm passes the gate at all, so the search is pure
    #: overhead (1.0x after the cheap screen was added, 0.7x before). On
    #: electric_drill.stl a 2 mm lattice passes the gate and then puts the
    #: refiner on a pose it cannot walk back to feasible, so the fallback
    #: re-runs the whole job at 0.5 mm: 0.8x. The delivered geometry was
    #: identical in both cases -- the cost is time, not accuracy.
    #:
    #: Kept and wired because it is the right lever for large parts, where the
    #: gate does permit a coarse lattice and the cubic saving is real; it just
    #: cannot be the default on parts like these.
    auto_pitch: bool = False
    robustness_trials: int = 12
    cross_check: bool = True
    #: attempt automatic repair when the input mesh is not a closed solid.
    #: Watertight input never reaches the repair path, so this is inert for it.
    repair: bool = True
    #: rebuild the part as a voxel solid when topological repair cannot close
    #: it. Only reached where the run would otherwise abort, and the surface
    #: it produces is an approximation — see MeshSolidify.
    solidify: bool = True
    #: voxel pitch for that rebuild; None picks one from the part's size
    solidify_pitch: float | None = None
    #: drop disconnected debris before nesting. A single-body mesh never
    #: reaches the check, so this is inert for it.
    denoise: bool = True
    #: a fragment is debris below this share of the main fragment's bbox
    #: diagonal. Deliberately timid: a real two-piece assembly keeps both.
    denoise_ratio: float = 0.05
    origin_corner: bool = True
    verbose: bool = True

    def to_dict(self):
        return asdict(self)


PROFILES = {
    # Deliberately inaccurate, for triaging a part rather than quoting it.
    # This is the class of result a coarse voxel packer gives: a coarse
    # lattice, the gate that would refuse it relaxed to suit, and no
    # continuous refinement -- so the delivered gap is whatever the lattice
    # happened to leave, looser than requested, not squeezed to it.
    #
    # Measured against 'quick' on sample.stl: see the note in the README. Use
    # it to see roughly how a part nests; do not quote a clearance from it.
    "fast": dict(coarse_step=45.0, so3_step=None, refiner="none",
                 n_samples=40_000, cross_check=False, robustness_trials=2,
                 fine_step=6.0, coarse_pitch=8.0, fine_pitch=4.0,
                 voxel_tolerance=0.25, auto_pitch=True, verify=False),
    # fast sanity check on a new part
    # 'descend' is switched off (see DISABLED_ALGORITHMS), so quick refines
    # with the profile sweep. That is the more accurate of the two and much
    # dearer: measured 3,059 KD queries against 61 on the same part.
    "quick": dict(coarse_step=30.0, so3_step=None, refiner="profile",
                  n_samples=120_000, cross_check=False, robustness_trials=6,
                  fine_step=3.0),
    # the settings that produced the reference result
    "standard": dict(coarse_step=15.0, so3_step=None, refiner="profile",
                     cross_check=True),
    # the 744-orientation SO(3) sweep is switched off, so this now differs
    # from 'standard' only in robustness_trials: the rotation is assumed from
    # the Z-family rather than proven over SO(3).
    "full": dict(coarse_step=15.0, so3_step=None, refiner="profile",
                 cross_check=True, robustness_trials=20),
}


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
class NesterFactory:
    """Builds configured components. The driver asks for parts, not classes."""

    @staticmethod
    def config(profile: str = "standard", **overrides) -> NestingConfig:
        if profile not in PROFILES:
            raise KeyError(f"unknown profile {profile!r}; "
                           f"choose from {sorted(PROFILES)}")
        base = dict(PROFILES[profile])
        base.update({k: v for k, v in overrides.items() if v is not None})
        cfg = NestingConfig(**base)
        for cat, name in (("voxelizer", cfg.voxelizer),
                          ("orientations", cfg.orientations),
                          ("distance", cfg.distance),
                          ("distance_backend", cfg.distance_backend),
                          ("refiner", cfg.refiner),
                          ("objective", cfg.objective)):
            AlgorithmRegistry.get(cat, name)          # fail fast on typos
        for label in (cfg.refiner, cfg.orientations):
            if not algorithm_enabled(label):
                raise RuntimeError(
                    f"the {label!r} algorithm is switched off: "
                    f"{DISABLED_ALGORITHMS[label]}")
        if cfg.so3_step and not algorithm_enabled("so3"):
            raise RuntimeError(
                "so3_step was given but the SO(3) sweep is switched off: "
                + DISABLED_ALGORITHMS["so3"])
        if cfg.distance_backend == "bvh" and not HAVE_FCL:
            raise RuntimeError(
                "distance_backend='bvh' needs python-fcl, which is not "
                "installed. Run `pip install python-fcl`, or leave the "
                "backend at 'sampled'.")
        return cfg

    # -- components -------------------------------------------------------- #
    @staticmethod
    def audit(mesh: trimesh.Trimesh) -> MeshAudit:
        return MeshAudit(mesh)

    @staticmethod
    def clearance_grid(mesh, pitch: float, cfg: NestingConfig) -> ClearanceGrid:
        radius = ClearanceGrid.safe_radius(cfg.clearance, pitch)
        return AlgorithmRegistry.get("clearance", "edt_dilation")(mesh, pitch, radius)

    @staticmethod
    def oracle(grid: ClearanceGrid) -> TranslationOracle:
        return AlgorithmRegistry.get("oracle", "fft_correlation")(grid)

    @staticmethod
    def orientations(cfg: NestingConfig, stage: str = "coarse", base=None):
        if stage == "coarse":
            gen = AlgorithmRegistry.get("orientations", cfg.orientations)
            cands = list(gen(cfg.coarse_step))
            if cfg.so3_step:
                cands += list(AlgorithmRegistry.get("orientations", "so3")(cfg.so3_step))
            return cands
        if stage == "fine":
            return list(AlgorithmRegistry.get("orientations", "local")(
                base, cfg.fine_span, cfg.fine_step))
        raise ValueError(stage)

    @staticmethod
    def backend(cfg: NestingConfig):
        """The distance class the run is configured to use."""
        return AlgorithmRegistry.get("distance_backend", cfg.distance_backend)

    @staticmethod
    def distance(meshA, meshB, t_ref, cfg: NestingConfig,
                 n_samples: int | None = None, transform=None):
        backend = NesterFactory.backend(cfg)
        if backend is not SurfacePairDistance:
            # BVH answers by hierarchy traversal; there is no KD query to prune
            return backend(meshA, meshB, t_ref, n_samples or cfg.n_samples)
        # the fine sweep's ClearanceGrid already ran this distance transform
        # and left it in the job's cache; a miss simply means no pruning
        pool = SurfaceSampleCache.current()
        field = (pool.field_for(meshA, cfg.fine_pitch)
                 if pool is not None else None)
        field_b = NesterFactory._field_for_b(field, meshA, meshB, transform)
        return backend(meshA, meshB, t_ref, n_samples or cfg.n_samples,
                       field=field, field_b=field_b)

    @staticmethod
    def _field_for_b(field, meshA, meshB, transform):
        """A's field placed on B, when B really is A moved by ``transform``.

        Callers pass ``meshB`` and the transform separately, so nothing stops
        the two disagreeing. A field sitting where the geometry is not would
        exclude points that do hold the minimum and hand back a distance that
        is too large, so the claim is checked against the geometry instead of
        taken on faith — cheap next to the queries it saves, and a mismatch
        only costs the pruning.
        """
        if field is None or transform is None:
            return None
        if len(meshA.faces) != len(meshB.faces):
            return None
        want = trimesh.transform_points(meshA.triangles.mean(axis=1),
                                        np.asarray(transform, float))
        tol = 1e-6 * max(1.0, float(np.max(meshA.extents)))
        if np.abs(want - meshB.triangles.mean(axis=1)).max() > tol:
            return None
        try:
            return field.transformed(transform)
        except ValueError:                     # not a rigid motion
            return None

    @staticmethod
    def metric(cfg: NestingConfig) -> Callable:
        """The callable used as the feasibility test's distance measure."""
        return AlgorithmRegistry.get("distance", cfg.distance)

    @staticmethod
    def objective(cfg: NestingConfig, single_extents) -> Callable:
        fn = AlgorithmRegistry.get("objective", cfg.objective)
        ref = np.asarray(single_extents, float)
        return lambda extents: fn(np.asarray(extents, float), ref)

    @staticmethod
    def refiner(objective_fn: Callable, feasible_fn: Callable) -> Refiner:
        return Refiner(objective_fn, feasible_fn)

    @staticmethod
    def strategy(cfg: NestingConfig) -> str:
        return AlgorithmRegistry.get("refiner", cfg.refiner)

    # -- introspection ----------------------------------------------------- #
    @staticmethod
    def describe(cfg: NestingConfig) -> str:
        rows = [("clearance", f"{cfg.clearance} (file units)"),
                ("voxeliser", cfg.voxelizer),
                ("pitch", f"coarse {cfg.coarse_pitch} / fine {cfg.fine_pitch}"),
                ("orientations", f"{cfg.orientations}"
                                 f"{' + so3' if cfg.so3_step else ''} "
                                 f"@ {cfg.coarse_step} deg"),
                ("distance", f"{cfg.distance} via {cfg.distance_backend}"),
                ("refiner", cfg.refiner),
                ("objective", cfg.objective),
                ("samples", f"{cfg.n_samples:,}"),
                ("recommendations", cfg.top_n)]
        return "\n".join(f"  {k:16s} {v}" for k, v in rows)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect the algorithm registry.")
    p.add_argument("--status", choices=("used", "reference", "rejected"))
    a = p.parse_args()
    print(AlgorithmRegistry.describe(a.status))
