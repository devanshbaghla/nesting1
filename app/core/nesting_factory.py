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
    ClearanceGrid, Geometry, MeshAudit, OrientationSet, Preview, Refiner,
    ScanlineVoxelizer, SurfacePairDistance, TranslationOracle, Validation,
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
        return sorted(cls._reg.get(category, {}))

    @classmethod
    def categories(cls) -> list[str]:
        return sorted(cls._reg)

    @classmethod
    def catalogue(cls, status: str | None = None) -> list[tuple]:
        rows = []
        for cat in cls.categories():
            for name, meta in sorted(cls._reg[cat].items()):
                if status and meta["status"] != status:
                    continue
                rows.append((cat, name, meta["status"], meta["note"]))
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

# -- rendering -------------------------------------------------------------- #
AlgorithmRegistry.add("render", "part_views", Preview.part_views,
                      note="single-part ortho projection; finds pockets")
AlgorithmRegistry.add("render", "pair", Preview.render,
                      note="ortho + isometric of the nested pair")


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
    refiner: str = "profile"
    objective: str = "volume"

    n_samples: int = 220_000
    guard_extra: float = 0.08
    scan_span: float = 1.6
    scan_step: float = 0.1

    top_n: int = 10
    diversity_angle: float = 10.0     # deg between distinct rotations
    diversity_extent: float = 3.0     # mm between distinct bounding boxes

    robustness_trials: int = 12
    cross_check: bool = True
    #: attempt automatic repair when the input mesh is not a closed solid.
    #: Watertight input never reaches the repair path, so this is inert for it.
    repair: bool = True
    origin_corner: bool = True
    verbose: bool = True

    def to_dict(self):
        return asdict(self)


PROFILES = {
    # fast sanity check on a new part
    "quick": dict(coarse_step=30.0, so3_step=None, refiner="descend",
                  n_samples=120_000, cross_check=False, robustness_trials=6,
                  fine_step=3.0),
    # the settings that produced the reference result
    "standard": dict(coarse_step=15.0, so3_step=None, refiner="profile",
                     cross_check=True),
    # adds the 744-orientation sweep, so the rotation is proven not assumed
    "full": dict(coarse_step=15.0, so3_step=30.0, refiner="profile",
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
                          ("refiner", cfg.refiner),
                          ("objective", cfg.objective)):
            AlgorithmRegistry.get(cat, name)          # fail fast on typos
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
    def distance(meshA, meshB, t_ref, cfg: NestingConfig,
                 n_samples: int | None = None) -> SurfacePairDistance:
        return SurfacePairDistance(meshA, meshB, t_ref,
                                   n_samples or cfg.n_samples)

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
                ("distance", cfg.distance),
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
