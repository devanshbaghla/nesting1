"""SurfaceSampleCache: reuse part A without changing a single number.

The cache is only safe because it advances the RNG exactly as the sampling it
skipped would have. Part B is drawn from the same generator immediately
afterwards, so a hit that left the stream alone would re-draw B from part A's
numbers and quietly change every distance in the run. Most of what is checked
here is that invariant.
"""
import threading

import numpy as np
import trimesh

from app.core.nesting3d import SurfacePairDistance, SurfaceSampleCache

N = 20_000


def _pair():
    a = trimesh.creation.box(extents=(20.0, 30.0, 40.0))
    b = a.copy()
    b.apply_transform(trimesh.transformations.rotation_matrix(0.7, [0, 0, 1]))
    return a, b, np.array([0.0, 55.0, 0.0])


# --------------------------------------------------------------------------- #
def test_cached_run_is_bit_identical():
    a, b, t = _pair()
    plain = SurfacePairDistance(a, b, t, n_samples=N)
    with SurfaceSampleCache():
        first = SurfacePairDistance(a, b, t, n_samples=N)
        second = SurfacePairDistance(a, b, t, n_samples=N)

    for other in (first, second):
        assert np.array_equal(plain.pA, other.pA), "part A samples differ"
        assert np.array_equal(plain.pB, other.pB), "part B samples differ — RNG drifted"
        assert np.array_equal(plain.qA, other.qA)
        assert np.array_equal(plain.qB, other.qB)
        for probe in (t, t + 0.31, t - 1.7):
            assert plain.exact(probe) == other.exact(probe)
            assert plain.sampled(probe) == other.sampled(probe)


def test_part_a_is_built_once_and_shared():
    a, b, t = _pair()
    with SurfaceSampleCache() as pool:
        first = SurfacePairDistance(a, b, t, n_samples=N)
        for _ in range(4):
            nxt = SurfacePairDistance(a, b, t, n_samples=N)
            assert nxt.treeA is first.treeA, "tree A was rebuilt"
            assert nxt.pA is first.pA, "part A was re-sampled"
    assert pool.stats()["builds"] == 1
    assert pool.stats()["reuses"] == 4


def test_a_different_part_a_is_not_reused():
    a, b, t = _pair()
    other = trimesh.creation.box(extents=(21.0, 30.0, 40.0))
    with SurfaceSampleCache() as pool:
        one = SurfacePairDistance(a, b, t, n_samples=N)
        two = SurfacePairDistance(other, b, t, n_samples=N)
        assert two.treeA is not one.treeA
    assert pool.stats()["builds"] == 2


def test_in_place_edit_invalidates():
    """Keyed on mesh content, so a mutated part A must not hit a stale entry."""
    a, b, t = _pair()
    with SurfaceSampleCache() as pool:
        one = SurfacePairDistance(a, b, t, n_samples=N)
        a.vertices[0] += 0.5
        two = SurfacePairDistance(a, b, t, n_samples=N)
        assert two.pA is not one.pA
    assert pool.stats()["builds"] == 2


def test_sample_count_and_seed_are_part_of_the_key():
    a, b, t = _pair()
    with SurfaceSampleCache() as pool:
        SurfacePairDistance(a, b, t, n_samples=N)
        SurfacePairDistance(a, b, t, n_samples=N + 1000)
        SurfacePairDistance(a, b, t, n_samples=N, seed=7)
    assert pool.stats()["builds"] == 3, "entries must not collide"


def test_nothing_leaks_out_of_the_block():
    a, b, t = _pair()
    assert SurfaceSampleCache.current() is None
    with SurfaceSampleCache() as pool:
        SurfacePairDistance(a, b, t, n_samples=N)
        assert SurfaceSampleCache.current() is pool
        assert pool.stats()["live_entries"] == 1
    assert SurfaceSampleCache.current() is None
    assert pool.stats()["live_entries"] == 0, "entries outlived the job"


def test_nested_blocks_restore_the_outer_cache():
    with SurfaceSampleCache() as outer:
        with SurfaceSampleCache() as inner:
            assert SurfaceSampleCache.current() is inner
        assert SurfaceSampleCache.current() is outer


def test_caches_are_per_thread():
    """Concurrent jobs must not share entries — JobStore runs them in threads."""
    a, b, t = _pair()
    seen = {}

    def job(name):
        with SurfaceSampleCache() as pool:
            d = SurfacePairDistance(a, b, t, n_samples=N)
            seen[name] = (pool, d.treeA, pool.stats()["builds"])

    threads = [threading.Thread(target=job, args=(i,)) for i in range(3)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    pools = [v[0] for v in seen.values()]
    trees = [v[1] for v in seen.values()]
    assert len(set(map(id, pools))) == 3, "threads shared a cache"
    assert len(set(map(id, trees))) == 3, "a tree crossed threads"
    assert all(v[2] == 1 for v in seen.values())


def test_uncached_path_still_works():
    a, b, t = _pair()
    assert SurfaceSampleCache.current() is None
    d = SurfacePairDistance(a, b, t, n_samples=N)
    assert d.exact(t) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all sample-cache tests passed")
