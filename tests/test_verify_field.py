"""verify_one reusing the sweep's distance field, without giving up its job.

verify_one exists to distrust the pipeline: it re-reads the written STL and
re-measures the gap, which is what catches an export or transform bug. Reusing
a distance field placed by the *claimed* transform would quietly undo that — a
field sitting where the geometry is not would prune away the points holding the
true minimum and hand back a gap that is too large, hiding exactly the class of
bug the stage is for.

So the placement is checked against the file before the field is used. These
tests cover both halves of that bargain: the gap is identical when the export
is right, and the optimisation stands down (still reporting the true gap) when
it is wrong.
"""
import numpy as np
import trimesh

from app.core.nest_base import NestingRecommender
from app.core.nesting3d import ClearanceGrid, SurfaceSampleCache
from app.core.nesting_factory import NesterFactory

PART = (20.0, 30.0, 40.0)


def _recommender():
    cfg = NesterFactory.config("quick", verbose=False)
    return NestingRecommender(cfg), cfg


def _part():
    return trimesh.creation.box(extents=PART)


def _stacked(gap=6.0):
    """Transform putting a second copy squarely above the first."""
    M = np.eye(4)
    M[:3, 3] = [0.0, 0.0, PART[2] + gap]
    return M


def _write(rec, mesh, M, tmp):
    asm, _, _ = rec._assembly(mesh, M)
    asm.export(str(tmp))
    return tmp


def _seeded_pool(rec, mesh):
    """A cache holding the field the sweep would have published."""
    pool = SurfaceSampleCache()
    pool.__enter__()
    ClearanceGrid(mesh, rec.cfg.fine_pitch,
                  ClearanceGrid.safe_radius(rec.cfg.clearance, rec.cfg.fine_pitch))
    return pool


# --------------------------------------------------------------------------- #
def test_gap_is_identical_with_and_without_the_field(tmp_path=None):
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh, M = _part(), _stacked()
    _write(rec, mesh, M, tmp)

    pool = _seeded_pool(rec, mesh)
    try:
        plain = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces))
        fast = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces),
                              mesh=mesh, transform=M)
    finally:
        pool.__exit__()

    assert fast["pruned_by_field"] is True, fast.get("field_note")
    assert not plain.get("pruned_by_field")
    assert fast["gap"] == plain["gap"], (fast["gap"], plain["gap"])
    assert fast["gap_method"] == plain["gap_method"] == "exact"
    assert np.isclose(fast["gap"], 6.0, atol=0.05), fast["gap"]


def test_a_wrong_transform_declines_the_field_and_still_measures_right():
    """The bug verify_one exists to catch must not be maskable by this."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh, M = _part(), _stacked()
    _write(rec, mesh, M, tmp)

    lying = M.copy()
    lying[:3, 3] = [0.0, 0.0, PART[2] + 25.0]      # claims a 25 mm gap

    pool = _seeded_pool(rec, mesh)
    try:
        truth = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces))
        got = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces),
                             mesh=mesh, transform=lying)
    finally:
        pool.__exit__()

    assert got["pruned_by_field"] is False
    assert "expected placement" in got.get("field_note", "")
    assert got["gap"] == truth["gap"], "the true gap must survive a bad claim"
    assert np.isclose(got["gap"], 6.0, atol=0.05)


def test_a_rotated_placement_is_reused_too():
    """Copy B is rotated, so its field is a view through a full rigid motion."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh = _part()
    M = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    M[:3, 3] = [0.0, 0.0, PART[2] + 7.0]
    _write(rec, mesh, M, tmp)

    pool = _seeded_pool(rec, mesh)
    try:
        plain = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces))
        fast = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces),
                              mesh=mesh, transform=M)
    finally:
        pool.__exit__()

    assert fast["pruned_by_field"] is True, fast.get("field_note")
    assert fast["gap"] == plain["gap"]


def test_without_a_field_it_is_the_old_path():
    """No sweep, no field: verify must still work, just unpruned."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh, M = _part(), _stacked()
    _write(rec, mesh, M, tmp)

    with SurfaceSampleCache():                  # opened, but nothing published
        out = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces),
                             mesh=mesh, transform=M)
    assert out["pruned_by_field"] is False
    assert "no field" in out["field_note"]
    assert np.isclose(out["gap"], 6.0, atol=0.05)


def test_no_cache_at_all_is_safe():
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh, M = _part(), _stacked()
    _write(rec, mesh, M, tmp)
    out = rec.verify_one(str(tmp), full=True, n_faces=len(mesh.faces),
                         mesh=mesh, transform=M)
    assert out["pruned_by_field"] is False
    assert np.isclose(out["gap"], 6.0, atol=0.05)


def test_the_sampled_path_agrees_too():
    """full=False reports the sampled metric; it must not move either."""
    import tempfile, pathlib
    tmp = pathlib.Path(tempfile.mkdtemp()) / "pair.stl"
    rec, _ = _recommender()
    mesh, M = _part(), _stacked()
    _write(rec, mesh, M, tmp)

    pool = _seeded_pool(rec, mesh)
    try:
        plain = rec.verify_one(str(tmp), full=False, n_faces=len(mesh.faces))
        fast = rec.verify_one(str(tmp), full=False, n_faces=len(mesh.faces),
                              mesh=mesh, transform=M)
    finally:
        pool.__exit__()
    assert fast["gap_method"] == "sampled"
    assert fast["gap"] == plain["gap"]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all verify-field tests passed")
