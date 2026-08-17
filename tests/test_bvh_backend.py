"""The BVH backend must be swappable for the sampled one without lying.

These check agreement rather than speed. The sampled metric is the incumbent
and is trusted, so anywhere the two disagree by more than its own documented
sampling bias, the BVH backend is what is on trial.
"""
import numpy as np
import trimesh

from app.core.nesting3d import (HAVE_FCL, BVHPairDistance, SurfacePairDistance,
                                SurfaceSampleCache, Validation)
from app.core.nesting_factory import AlgorithmRegistry, NesterFactory

SKIP = not HAVE_FCL
N = 120_000


def _pair(angle=0.7):
    a = trimesh.creation.box(extents=(20.0, 30.0, 40.0))
    b = a.copy()
    b.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    return a, b


# --------------------------------------------------------------------------- #
def test_agrees_with_the_sampled_metric():
    if SKIP:
        return
    a, b = _pair()
    t = np.array([0.0, 46.0, 3.0])
    s = SurfacePairDistance(a, b, t, N)
    v = BVHPairDistance(a, b, t)
    for probe in (t, t + [0, 1.2, 0], t + [0.5, -2.0, 1.0], t + [0, 6.0, 0]):
        got, want = v.exact(probe), s.exact(probe)
        assert abs(got - want) < 0.02, f"{got} vs {want} at {probe}"


def test_analytic_sphere_gate_passes():
    """The gate that guards every run, on the backend it will actually use."""
    if SKIP:
        return
    r = Validation.sphere_pair(n=40_000, backend=BVHPairDistance)
    assert r["pass"], r
    assert abs(r["error"]) < 0.05


def test_beats_the_sampled_metric_on_the_analytic_case():
    """Faceting puts the true gap slightly above d-2r; exact should be closer."""
    if SKIP:
        return
    s = Validation.sphere_pair(n=40_000, backend=SurfacePairDistance)
    v = Validation.sphere_pair(n=40_000, backend=BVHPairDistance)
    assert abs(v["error"]) <= abs(s["error"]) + 1e-9, (
        f"BVH error {v['error']} worse than sampled {s['error']}")


def test_overlap_reads_zero_not_negative():
    if SKIP:
        return
    a, b = _pair()
    v = BVHPairDistance(a, b, np.array([0.0, 46.0, 0.0]))
    assert v.exact(np.array([0.0, 46.0, 0.0])) > 0
    for t in ([0.0, 5.0, 0.0], [0.0, 0.0, 0.0], [1.0, -3.0, 2.0]):
        assert v.exact(np.array(t)) == 0.0, "overlap must read 0.0"
    assert v.exact(np.array([0.0, 0.0, 0.0])) < 5.0, "0 must fail a clearance test"


def test_sampled_alias_matches_exact():
    if SKIP:
        return
    a, b = _pair()
    t = np.array([0.0, 46.0, 0.0])
    v = BVHPairDistance(a, b, t)
    assert v.sampled(t) == v.exact(t)
    assert v.sampled(t, coarse=True) == v.exact(t)


def test_interface_parity_with_the_sampled_class():
    if SKIP:
        return
    a, b = _pair()
    t = np.array([0.0, 46.0, 0.0])
    s, v = SurfacePairDistance(a, b, t, 20_000), BVHPairDistance(a, b, t)
    for attr in ("exact", "sampled", "exact_reference", "spacing", "qA", "qB"):
        assert hasattr(v, attr), f"BVH backend is missing {attr}"
        assert hasattr(s, attr)


def test_part_a_hierarchy_is_reused_across_candidates():
    if SKIP:
        return
    a, b = _pair()
    t = np.array([0.0, 46.0, 0.0])
    with SurfaceSampleCache() as pool:
        first = BVHPairDistance(a, b, t)
        for _ in range(4):
            BVHPairDistance(a, b, t)
    # one entry for A, plus nothing else cached by this backend
    assert pool.stats()["reuses"] == 4, pool.stats()
    assert pool.stats()["builds"] == 1, pool.stats()
    assert first.exact(t) > 0


def test_factory_selects_the_backend():
    cfg = NesterFactory.config("quick", distance_backend="sampled")
    assert NesterFactory.backend(cfg) is SurfacePairDistance
    if SKIP:
        return
    cfg = NesterFactory.config("quick", distance_backend="bvh")
    assert NesterFactory.backend(cfg) is BVHPairDistance
    a, b = _pair()
    d = NesterFactory.distance(a, b, np.array([0.0, 46.0, 0.0]), cfg)
    assert isinstance(d, BVHPairDistance)


def test_default_is_unchanged():
    """Nobody gets the new backend by accident."""
    cfg = NesterFactory.config("quick")
    assert cfg.distance_backend == "sampled"
    assert NesterFactory.backend(cfg) is SurfacePairDistance
    for profile in ("quick", "standard", "full"):
        assert NesterFactory.config(profile).distance_backend == "sampled"


def test_unknown_backend_is_rejected():
    try:
        NesterFactory.config("quick", distance_backend="nope")
    except KeyError as exc:
        assert "distance_backend" in str(exc)
    else:
        raise AssertionError("a typo'd backend must fail fast")


def test_missing_fcl_gives_a_clear_error():
    """Simulate an install without python-fcl."""
    import app.core.nesting3d as n3
    saved = n3.fcl
    n3.fcl = None
    try:
        BVHPairDistance(*_pair(), np.zeros(3))
    except RuntimeError as exc:
        assert "python-fcl" in str(exc) and "pip install" in str(exc)
    else:
        raise AssertionError("must refuse without fcl")
    finally:
        n3.fcl = saved


if __name__ == "__main__":
    if SKIP:
        print("python-fcl not installed — BVH tests skipped")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all bvh-backend tests passed")
