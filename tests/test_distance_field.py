"""The retained distance field: it must speed queries up without moving them.

SurfaceDistanceField is an accelerator, not a metric. Every test here is some
form of the same assertion — the answer with the field equals the answer
without it — because the moment that stops being true the field stops being an
optimisation and becomes a silent change to the delivered clearance.

The one test that is not about equality is the one that checks the filter
actually rules points out. An accelerator that is exact but never prunes is
just overhead, and nothing else here would catch that.
"""
import numpy as np
import trimesh

from app.core.nesting3d import (ClearanceGrid, Geometry, ScanlineVoxelizer,
                                SurfaceDistanceField, SurfacePairDistance,
                                SurfaceSampleCache, TransformedDistanceField)

PITCH = 0.5
N = 40_000


def _true_distance(mesh, pts):
    """Exact point-to-surface distance, brute force over every triangle.

    ``trimesh.proximity`` needs rtree, which is not a dependency here, and the
    fixtures are boxes — twelve triangles, so the honest answer is cheaper than
    an acceleration structure anyway.
    """
    tris = np.asarray(mesh.triangles)
    n, m = len(pts), len(tris)
    d = Geometry.point_triangle_distance(
        np.repeat(np.asarray(pts, float), m, axis=0),
        np.tile(tris, (n, 1, 1)))
    return d.reshape(n, m).min(axis=1)


def _part():
    return trimesh.creation.box(extents=(20.0, 30.0, 40.0))


def _pair(mesh=None, gap=6.0):
    mesh = mesh or _part()
    T = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    mB = mesh.copy(); mB.apply_transform(T)
    return mesh, mB, np.array([0.0, 0.0, float(mesh.extents[2]) + gap])


def _field(mesh, pitch=PITCH, margin=12.0):
    occ, i0, _ = ScanlineVoxelizer.solid(mesh, pitch)
    return SurfaceDistanceField.build(occ, i0, pitch,
                                      int(np.ceil(margin / pitch)))


# --------------------------------------------------------------------------- #
#  the bound
# --------------------------------------------------------------------------- #
def test_lower_bound_never_exceeds_the_truth():
    """The whole filter rests on this: the bound must never be optimistic."""
    mesh = _part()
    field = _field(mesh)
    rng = np.random.default_rng(0)
    lo, hi = mesh.bounds[0] - 10.0, mesh.bounds[1] + 10.0
    pts = rng.uniform(lo, hi, size=(4000, 3))

    truth = _true_distance(mesh, pts)
    assert (field.lower_bound(pts) <= truth + 1e-9).all(), \
        "lower_bound rose above the true surface distance"


def test_bound_holds_outside_the_grid():
    """Reads past the edge clamp, and a clamped read must still under-state."""
    mesh = _part()
    field = _field(mesh, margin=4.0)              # deliberately tight
    far = np.array([[0.0, 0.0, 500.0], [300.0, -400.0, 0.0]])
    truth = _true_distance(mesh, far)
    assert (field.lower_bound(far) <= truth).all()


def test_tolerance_is_a_whole_voxel_diagonal():
    field = _field(_part(), pitch=0.5)
    assert np.isclose(field.tolerance, np.sqrt(3) * 0.5)


# --------------------------------------------------------------------------- #
#  equality with the unaccelerated path
# --------------------------------------------------------------------------- #
def test_sampled_is_unchanged():
    mesh, mB, t0 = _pair()
    field = _field(mesh)
    with SurfaceSampleCache():
        ref = SurfacePairDistance(mesh, mB, t0, n_samples=N)
        acc = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    for i in range(20):
        t = t0 + np.array([0.2 * i, -0.15 * i, -0.4 * i])
        for coarse in (True, False):
            assert ref.sampled(t, coarse) == acc.sampled(t, coarse), \
                f"sampled diverged at {t} coarse={coarse}"


def test_exact_is_unchanged():
    """exact() keeps the KD path; the field must not perturb it either."""
    mesh, mB, t0 = _pair()
    field = _field(mesh)
    with SurfaceSampleCache():
        ref = SurfacePairDistance(mesh, mB, t0, n_samples=N)
        acc = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    for i in range(8):
        t = t0 + np.array([0.0, 0.0, -0.5 * i])
        assert ref.exact(t) == acc.exact(t)


def test_the_pruned_point_sets_are_identical():
    """qB feeds subB feeds every coarse gate; a different set is a different run."""
    mesh, mB, t0 = _pair()
    field = _field(mesh)
    with SurfaceSampleCache():
        ref = SurfacePairDistance(mesh, mB, t0, n_samples=N)
        acc = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    assert np.array_equal(ref.qA, acc.qA)
    assert np.array_equal(ref.qB, acc.qB)
    assert np.array_equal(ref.qfA, acc.qfA)
    assert np.array_equal(ref.qfB, acc.qfB)
    assert np.array_equal(ref.subB, acc.subB)


def test_equality_holds_when_the_parts_touch():
    """Contact is where the filter keeps the most points and margins are thinnest."""
    mesh, mB, _ = _pair()
    t = np.array([0.0, 0.0, float(mesh.extents[2])])       # flush
    field = _field(mesh)
    with SurfaceSampleCache():
        ref = SurfacePairDistance(mesh, mB, t, n_samples=N)
        acc = SurfacePairDistance(mesh, mB, t, n_samples=N, field=field)
    assert ref.sampled(t, True) == acc.sampled(t, True)
    assert ref.exact(t) == acc.exact(t)


def test_a_coarse_field_still_gives_the_same_answer():
    """Accuracy of the field changes how much it prunes, never the result."""
    mesh, mB, t0 = _pair()
    with SurfaceSampleCache():
        ref = SurfacePairDistance(mesh, mB, t0, n_samples=N)
        for pitch in (2.0, 1.0, 0.5):
            acc = SurfacePairDistance(mesh, mB, t0, n_samples=N,
                                      field=_field(mesh, pitch=pitch))
            assert ref.sampled(t0, True) == acc.sampled(t0, True), pitch
            assert np.array_equal(ref.qB, acc.qB), pitch


def test_no_field_is_the_old_path():
    mesh, mB, t0 = _pair()
    with SurfaceSampleCache():
        d = SurfacePairDistance(mesh, mB, t0, n_samples=N)
    assert d.field is None
    assert d.sampled(t0) > 0


# --------------------------------------------------------------------------- #
#  it has to actually prune
# --------------------------------------------------------------------------- #
def test_the_filter_rules_most_points_out():
    """Exactness is free if you keep everything; this is the part that pays."""
    mesh, mB, t0 = _pair(gap=8.0)
    field = _field(mesh)
    with SurfaceSampleCache():
        d = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    pts = d.subB + t0
    lb = field.lower_bound(pts)
    seeds = np.argpartition(lb, d._SEEDS)[:d._SEEDS]
    upper = float(d.treeA.query(pts[seeds])[0].min())
    kept = (lb <= upper).mean()
    assert kept < 0.5, f"filter kept {kept:.1%} of the cloud; not worth its gather"


def test_tiny_clouds_skip_the_gather():
    """Below the seed count the filter cannot win, so it must not be attempted."""
    mesh, mB, t0 = _pair()
    field = _field(mesh)
    with SurfaceSampleCache():
        d = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    few = d.subB[:8] + t0
    assert d.min_distance_to(d.treeA, few, field) == \
        float(d.treeA.query(few)[0].min())


# --------------------------------------------------------------------------- #
#  the grid keeps what it used to throw away
# --------------------------------------------------------------------------- #
def test_clearance_grid_retains_its_transform():
    mesh = _part()
    cg = ClearanceGrid(mesh, 1.0, ClearanceGrid.safe_radius(5.0, 1.0))
    assert cg.field is not None
    assert cg.field.distance.shape == cg.grid.shape
    assert np.array_equal(cg.origin, cg.field.origin)
    assert cg.field.distance.dtype == np.float32


def test_the_dilated_grid_is_what_it_always_was():
    """The sweep must not notice this change; grid is still dt <= radius."""
    from scipy.ndimage import distance_transform_edt

    mesh = _part()
    pitch, radius = 1.0, ClearanceGrid.safe_radius(5.0, 1.0)
    cg = ClearanceGrid(mesh, pitch, radius)

    occ, i0, _ = ScanlineVoxelizer.solid(mesh, pitch)
    pad = int(np.ceil(radius / pitch)) + 2
    padded = np.zeros(np.array(occ.shape) + 2 * pad, bool)
    padded[pad:-pad, pad:-pad, pad:-pad] = occ
    expected = distance_transform_edt(~padded, sampling=pitch) <= radius

    assert np.array_equal(cg.grid, expected), "the dilation changed"
    assert np.array_equal(cg.origin, i0 - pad)


def test_the_grid_publishes_its_field_for_reuse():
    """Built once during the sweep, borrowed later by the refiner."""
    mesh = _part()
    with SurfaceSampleCache() as pool:
        assert pool.field_for(mesh, 1.0) is None
        cg = ClearanceGrid(mesh, 1.0, ClearanceGrid.safe_radius(5.0, 1.0))
        assert pool.field_for(mesh, 1.0) is cg.field
    assert SurfaceSampleCache.current() is None


def test_a_published_field_is_not_evicted():
    """Coarse runs first; the finer field must survive it."""
    mesh = _part()
    with SurfaceSampleCache() as pool:
        first = _field(mesh, pitch=0.5)
        pool.publish_field(mesh, 0.5, first)
        pool.publish_field(mesh, 0.5, _field(mesh, pitch=0.5))
        assert pool.field_for(mesh, 0.5) is first


def test_fields_are_keyed_by_pitch():
    mesh = _part()
    with SurfaceSampleCache() as pool:
        pool.publish_field(mesh, 1.0, _field(mesh, pitch=1.0))
        assert pool.field_for(mesh, 1.0) is not None
        assert pool.field_for(mesh, 0.5) is None


# --------------------------------------------------------------------------- #
#  the transformed view
# --------------------------------------------------------------------------- #
def _rigid(angle=np.pi / 3, axis=(0.3, 1.0, 0.2), t=(4.0, -7.0, 130.0)):
    M = trimesh.transformations.rotation_matrix(angle, axis)
    M[:3, 3] = t
    return M


def test_a_view_bounds_the_moved_part():
    """The whole point: one field, read through a rigid motion, still bounds."""
    mesh = _part()
    field = _field(mesh, margin=12.0)
    M = _rigid()
    moved = mesh.copy(); moved.apply_transform(M)
    view = field.transformed(M)

    rng = np.random.default_rng(7)
    pts = rng.uniform(moved.bounds[0] - 4, moved.bounds[1] + 4, size=(1500, 3))
    truth = _true_distance(moved, pts)
    assert (view.lower_bound(pts) <= truth + 1e-9).all(), \
        "the view's lower bound rose above the true distance"


def test_a_view_is_no_looser_than_the_field_it_came_from():
    """Moving the query must not cost accuracy beyond the declared tolerance."""
    mesh = _part()
    field = _field(mesh, margin=12.0)
    M = _rigid()
    moved = mesh.copy(); moved.apply_transform(M)
    view = field.transformed(M)

    rng = np.random.default_rng(11)
    here = rng.uniform(mesh.bounds[0] - 4, mesh.bounds[1] + 4, size=(1200, 3))
    there = trimesh.transform_points(here, M)
    over_field = (field.values(here) - _true_distance(mesh, here)).max()
    over_view = (view.values(there) - _true_distance(moved, there)).max()
    assert over_field <= field.tolerance, over_field
    assert over_view <= view.tolerance, over_view


def test_identity_view_matches_the_field():
    field = _field(_part())
    view = field.transformed(np.eye(4))
    rng = np.random.default_rng(3)
    pts = rng.uniform(-30, 60, size=(500, 3))
    assert np.allclose(field.values(pts), view.values(pts), atol=1e-9)


def test_a_view_shares_the_array_rather_than_copying_it():
    field = _field(_part())
    view = field.transformed(_rigid())
    assert view.field is field
    assert view.nbytes == 0


def test_a_non_rigid_transform_is_refused():
    """A scale would stretch distance and silently invalidate every bound."""
    field = _field(_part())
    for bad in (np.diag([2.0, 1.0, 1.0, 1.0]),          # scale
                np.array([[1, 0.4, 0, 0], [0, 1, 0, 0],  # shear
                          [0, 0, 1, 0], [0, 0, 0, 1]], float)):
        try:
            field.transformed(bad)
        except ValueError as exc:
            assert "rigid" in str(exc)
        else:
            raise AssertionError("a non-rigid transform must be refused")


def test_a_malformed_transform_is_refused():
    field = _field(_part())
    try:
        field.transformed(np.eye(3))
    except ValueError as exc:
        assert "4x4" in str(exc)
    else:
        raise AssertionError("a 3x3 matrix must be refused")


# --------------------------------------------------------------------------- #
#  both-sided pruning
# --------------------------------------------------------------------------- #
def test_mask_within_matches_a_full_query():
    mesh, mB, t0 = _pair()
    field = _field(mesh)
    with SurfaceSampleCache():
        d = SurfacePairDistance(mesh, mB, t0, n_samples=N, field=field)
    pts = d.pB + t0
    for band in (1.0, 4.0, 13.0):
        want = d.treeA.query(pts)[0] <= band
        assert np.array_equal(d.mask_within(d.treeA, pts, band, field), want), band
        assert np.array_equal(d.mask_within(d.treeA, pts, band, None), want), band


def test_both_sided_pruning_leaves_every_number_alone():
    """field_b prunes the A-side; qA, qB and the gap must not move."""
    mesh = _part()
    M = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    M[:3, 3] = [0.0, 0.0, float(mesh.extents[2]) + 6.0]
    mB = mesh.copy(); mB.apply_transform(M)
    fa = _field(mesh)
    fb = fa.transformed(M)

    with SurfaceSampleCache():
        plain = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N, move=0.0)
        a_only = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N,
                                     move=0.0, field=fa)
        both = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N,
                                   move=0.0, field=fa, field_b=fb)
    for other, tag in ((a_only, "A only"), (both, "A and B")):
        assert np.array_equal(plain.qA, other.qA), tag
        assert np.array_equal(plain.qB, other.qB), tag
        assert np.array_equal(plain.subB, other.subB), tag
        assert plain.exact(np.zeros(3)) == other.exact(np.zeros(3)), tag


def test_field_b_alone_is_allowed():
    """Either side may be omitted; only that side loses its pruning."""
    mesh = _part()
    M = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    M[:3, 3] = [0.0, 0.0, float(mesh.extents[2]) + 6.0]
    mB = mesh.copy(); mB.apply_transform(M)
    with SurfaceSampleCache():
        plain = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N, move=0.0)
        b_only = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N,
                                     move=0.0, field_b=_field(mesh).transformed(M))
    assert np.array_equal(plain.qA, b_only.qA)
    assert plain.exact(np.zeros(3)) == b_only.exact(np.zeros(3))


# --------------------------------------------------------------------------- #
#  exact() under the filter
# --------------------------------------------------------------------------- #
def _stack(mesh, gap):
    M = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    M[:3, 3] = [0.0, 0.0, float(mesh.extents[2]) + gap]
    return M


def _three_ways(mesh, M, move=6.0):
    """The same pair built with no field, A's field, and both fields."""
    mB = mesh.copy(); mB.apply_transform(M)
    fa = _field(mesh, margin=14.0)
    fb = fa.transformed(M)
    with SurfaceSampleCache():
        return (SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N, move=move),
                SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N, move=move,
                                    field=fa),
                SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N, move=move,
                                    field=fa, field_b=fb))


def test_exact_is_identical_on_both_sides():
    """The number shipped to the user must not move by a single bit."""
    mesh = _part()
    plain, a_only, both = _three_ways(mesh, _stack(mesh, 6.0))
    for i in range(15):
        t = np.array([0.3 * np.sin(i), 0.5 * np.cos(i), -0.4 * i])
        want = plain.exact(t)
        assert a_only.exact(t) == want, f"A-side filter moved exact() at {t}"
        assert both.exact(t) == want, f"B-side filter moved exact() at {t}"


def test_exact_is_identical_at_contact():
    """Where the band is thinnest and the filter keeps the most."""
    mesh = _part()
    M = _stack(mesh, 0.0)
    plain, a_only, both = _three_ways(mesh, M, move=0.0)
    for dz in (0.0, -0.05, 0.05, -0.2):
        t = np.array([0.0, 0.0, dz])
        want = plain.exact(t)
        assert a_only.exact(t) == want, dz
        assert both.exact(t) == want, dz


def test_exact_is_identical_at_every_band():
    """`band` widens the selection; the filter must track it, not a constant."""
    mesh = _part()
    plain, _, both = _three_ways(mesh, _stack(mesh, 6.0))
    t = np.array([0.0, 0.0, -1.0])
    for band in (0.0, 0.3, 0.8, 2.0, 8.0):
        assert both.exact(t, band=band) == plain.exact(t, band=band), band


def test_exact_is_identical_for_every_k():
    mesh = _part()
    plain, _, both = _three_ways(mesh, _stack(mesh, 6.0))
    t = np.array([0.0, 0.0, -1.0])
    for k in (4, 12, 24, 48):
        assert both.exact(t, k=k) == plain.exact(t, k=k), k


def test_a_coarse_field_does_not_move_exact():
    """Field accuracy changes how much is pruned, never the result."""
    mesh = _part()
    M = _stack(mesh, 6.0)
    mB = mesh.copy(); mB.apply_transform(M)
    t = np.array([0.0, 0.0, -1.0])
    with SurfaceSampleCache():
        plain = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N)
        want = plain.exact(t)
        for pitch in (2.0, 1.0, 0.5):
            fa = _field(mesh, pitch=pitch, margin=14.0)
            got = SurfacePairDistance(mesh, mB, np.zeros(3), n_samples=N,
                                      field=fa, field_b=fa.transformed(M))
            assert got.exact(t) == want, pitch


def test_exact_still_beats_the_sampled_bound():
    """Sanity that the filter did not turn exact() into the sampled metric."""
    mesh = _part()
    _, _, both = _three_ways(mesh, _stack(mesh, 6.0))
    t = np.array([0.0, 0.0, -1.0])
    assert both.exact(t) <= both.sampled(t) + 1e-12


# --------------------------------------------------------------------------- #
#  the factory only lends B a field when B really is A moved
# --------------------------------------------------------------------------- #
def test_factory_supplies_b_a_field_for_a_true_placement():
    from app.core.nesting_factory import NesterFactory

    mesh = _part()
    M = _stack(mesh, 6.0)
    mB = mesh.copy(); mB.apply_transform(M)
    cfg = NesterFactory.config("quick", verbose=False)
    with SurfaceSampleCache() as pool:
        pool.publish_field(mesh, cfg.fine_pitch, _field(mesh, pitch=cfg.fine_pitch))
        d = NesterFactory.distance(mesh, mB, np.zeros(3), cfg, n_samples=N,
                                   transform=M)
    assert d.field is not None and d.field_b is not None


def test_factory_refuses_a_transform_that_does_not_match():
    """A wrong claim must cost the pruning, not the answer."""
    from app.core.nesting_factory import NesterFactory

    mesh = _part()
    mB = mesh.copy(); mB.apply_transform(_stack(mesh, 6.0))
    cfg = NesterFactory.config("quick", verbose=False)
    with SurfaceSampleCache() as pool:
        pool.publish_field(mesh, cfg.fine_pitch, _field(mesh, pitch=cfg.fine_pitch))
        lying = _stack(mesh, 25.0)
        d = NesterFactory.distance(mesh, mB, np.zeros(3), cfg, n_samples=N,
                                   transform=lying)
        plain = NesterFactory.distance(mesh, mB, np.zeros(3), cfg, n_samples=N)
    assert d.field_b is None, "a mismatched placement must not lend B a field"
    assert d.exact(np.zeros(3)) == plain.exact(np.zeros(3))


def test_factory_refuses_a_non_rigid_transform():
    from app.core.nesting_factory import NesterFactory

    mesh = _part()
    mB = mesh.copy(); mB.apply_transform(_stack(mesh, 6.0))
    cfg = NesterFactory.config("quick", verbose=False)
    with SurfaceSampleCache() as pool:
        pool.publish_field(mesh, cfg.fine_pitch, _field(mesh, pitch=cfg.fine_pitch))
        d = NesterFactory.distance(mesh, mB, np.zeros(3), cfg, n_samples=N,
                                   transform=np.diag([2.0, 1.0, 1.0, 1.0]))
    assert d.field_b is None


def test_factory_without_a_transform_is_the_old_path():
    from app.core.nesting_factory import NesterFactory

    mesh = _part()
    mB = mesh.copy(); mB.apply_transform(_stack(mesh, 6.0))
    cfg = NesterFactory.config("quick", verbose=False)
    with SurfaceSampleCache() as pool:
        pool.publish_field(mesh, cfg.fine_pitch, _field(mesh, pitch=cfg.fine_pitch))
        d = NesterFactory.distance(mesh, mB, np.zeros(3), cfg, n_samples=N)
    assert d.field is not None and d.field_b is None


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all distance-field tests passed")
