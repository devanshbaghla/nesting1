"""The voxel-rebuild fallback: what it closes, what it refuses, and what it costs.

:class:`MeshRepair` handles meshes that are nearly solid, and it handles more
than you would expect — a box with an entire wall missing is a planar boundary
loop, and a triangle fan across it is exactly right. The fallback exists for
what is left: openings a fan cannot honestly cap, assemblies of shells that
never knit, anything where the surface has to be rebuilt rather than mended.

The refusals matter as much as the rebuilds. Rasterising a flat sheet yields a
watertight one-voxel slab, and rasterising a shell with a gaping hole yields a
hollow husk. Both are closed, plausible, and not the part.
"""
import json

import numpy as np
import trimesh

from app.core.mesh_repair import (MeshRepair, MeshRepairError, MeshSolidify,
                                  SolidifyReport)


def _box(extents=(20.0, 30.0, 40.0)):
    return trimesh.creation.box(extents=extents)


def _holed_sphere(radius=12.0, drop=20):
    """A sphere with a patch cut out — an opening no fan can honestly cap.

    This is the shape :class:`MeshRepair` gives up on, so it is the input that
    actually reaches the fallback.
    """
    s = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    keep = np.ones(len(s.faces), bool)
    keep[:drop] = False
    return trimesh.Trimesh(vertices=s.vertices.copy(), faces=s.faces[keep],
                           process=False)


def _flat_patch():
    return trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)


def _unwelded(mesh):
    return trimesh.Trimesh(vertices=mesh.triangles.reshape(-1, 3),
                           faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3),
                           process=False)


# --------------------------------------------------------------------------- #
#  what it closes
# --------------------------------------------------------------------------- #
def test_a_closed_solid_rasterises_to_itself():
    """The control: with nothing to fix, the rebuild must reproduce the part."""
    box = _box((20.0, 20.0, 20.0))
    solid, rep = MeshSolidify.solidify(box)
    assert rep.ok, rep.summary()
    assert solid.is_watertight and solid.is_winding_consistent
    assert np.isclose(solid.volume, 8000.0, rtol=0.05), solid.volume


def test_an_opening_narrower_than_the_grid_is_bridged():
    """The fallback's whole job: close what a triangle fan cannot."""
    holed = _holed_sphere()
    assert not holed.is_watertight
    solid, rep = MeshSolidify.solidify(holed, pitch=1.0)
    assert rep.ok, rep.summary()
    assert solid.is_watertight
    assert MeshRepair.open_edge_count(solid) == 0
    assert rep.interior_voxels > rep.surface_voxels, "the interior did not fill"
    # a 12mm sphere, quantised outward by up to half a voxel on the radius
    assert 7176 <= solid.volume <= 7176 * 1.25, solid.volume


def test_output_is_manifold_not_merely_closed():
    """Watertight means every edge carries exactly two faces, pinches included.

    A staircase of diagonally-touching voxels is what rasterising any sloped
    surface produces, and it is where a naive filled/empty boundary stops
    being 2-manifold. Bridging happens to dissolve most pinches on its way
    past, so this runs with it off — the guarantee has to hold either way.
    """
    for close in (0, 1):
        solid, rep = MeshSolidify.solidify(
            trimesh.creation.icosphere(subdivisions=3, radius=12.0),
            pitch=1.0, close_voxels=close)
        assert rep.ok, rep.summary()
        counts = np.unique(
            np.unique(solid.edges_sorted, axis=0, return_counts=True)[1])
        assert counts.tolist() == [2], (
            f"non-manifold edges at close_voxels={close}: {counts}")


def test_a_pinched_grid_still_extracts_a_manifold_surface():
    """The guarantee at its worst case, built by hand rather than hoped for.

    A filled solid rarely pinches, so a fixture that happens to produce one is
    a fixture that can quietly stop producing one. This states the invariant
    on a grid that definitely does: a diagonal staircase, one voxel thick.
    """
    occ = np.zeros((10, 10, 4), bool)
    for i in range(1, 8):
        occ[i, i, 1] = occ[i, i, 2] = True             # edge-touching diagonal
    assert MeshSolidify.close_diagonal_contacts(occ) > 0
    solid = MeshSolidify.boundary_mesh(occ, np.eye(4))
    trimesh.repair.fix_winding(solid)
    counts = np.unique(
        np.unique(solid.edges_sorted, axis=0, return_counts=True)[1])
    assert counts.tolist() == [2], f"non-manifold edges remain: {counts}"
    assert solid.is_watertight


def test_diagonal_contacts_are_closed_on_the_grid():
    """The 2x2 checkerboard is the pinch; it must not survive the pass."""
    occ = np.zeros((4, 4, 4), bool)
    occ[1, 1, 1] = occ[2, 2, 1] = True         # touching along one edge only
    filled = MeshSolidify.close_diagonal_contacts(occ)
    assert filled >= 1
    assert occ[1, 2, 1] or occ[2, 1, 1], "the pinch was not widened"


def test_a_solid_grid_needs_no_pinch_fixing():
    occ = np.zeros((6, 6, 6), bool)
    occ[1:5, 1:5, 1:5] = True
    assert MeshSolidify.close_diagonal_contacts(occ) == 0


def test_separate_bodies_stay_separate():
    """Two parts a clear distance apart must not be welded into one."""
    a = _box((10.0, 10.0, 10.0))
    b = _box((10.0, 10.0, 10.0)); b.apply_translation([40.0, 0, 0])
    solid, rep = MeshSolidify.solidify(trimesh.util.concatenate([a, b]))
    assert rep.ok, rep.summary()
    assert rep.bodies_after == 2
    assert solid.is_watertight


# --------------------------------------------------------------------------- #
#  the occupancy grid
# --------------------------------------------------------------------------- #
def test_occupancy_fills_the_interior():
    box = _box((20.0, 20.0, 20.0))
    solid, surface, _ = MeshSolidify.occupancy(box, pitch=1.0)
    assert solid.sum() > surface.sum()
    assert np.isclose(solid.sum(), 21 ** 3, rtol=0.15), solid.sum()


def test_a_leaking_fill_is_detected():
    """An opening the fill escapes through leaves a shell, and it must be caught."""
    solid, surface, _ = MeshSolidify.occupancy(_holed_sphere(), pitch=0.3)
    assert MeshSolidify._fill_leaked(solid)
    solid, _, _ = MeshSolidify.occupancy(_box((20.0, 20.0, 20.0)), pitch=0.5)
    assert not MeshSolidify._fill_leaked(solid)


def test_pitch_scales_with_the_part():
    """Comparable relative fidelity for parts of very different sizes."""
    small = MeshSolidify.default_pitch(_box((10.0, 10.0, 10.0)))
    large = MeshSolidify.default_pitch(_box((1000.0, 1000.0, 1000.0)))
    assert np.isclose(large / small, 100.0, rtol=0.01), (small, large)


def test_pitch_is_coarsened_to_fit_the_voxel_budget():
    """A very large part must not try to allocate an unbounded grid."""
    mesh = _box((4000.0, 4000.0, 4000.0))
    pitch = MeshSolidify.default_pitch(mesh)
    assert np.prod(mesh.extents / pitch) <= MeshSolidify.MAX_VOXELS


def test_explicit_pitch_is_honoured():
    _, rep = MeshSolidify.solidify(_box((20.0, 20.0, 20.0)), pitch=2.0)
    assert rep.ok and rep.pitch == 2.0
    assert np.isclose(rep.error_mm(), 2.0 * np.sqrt(3) / 2)


def test_finer_pitch_is_more_faithful():
    box = _box((20.0, 20.0, 20.0))
    coarse, rc = MeshSolidify.solidify(box, pitch=2.0)
    fine, rf = MeshSolidify.solidify(box, pitch=0.5)
    assert rc.ok and rf.ok
    assert rf.error_mm() < rc.error_mm()
    assert abs(fine.volume - 8000.0) < abs(coarse.volume - 8000.0)


# --------------------------------------------------------------------------- #
#  what it refuses
# --------------------------------------------------------------------------- #
def test_a_flat_patch_is_refused():
    """Thickening a sheet is invention, not repair."""
    patch = _flat_patch()
    out, rep = MeshSolidify.solidify(patch)
    assert not rep.ok
    assert "flat" in rep.rejected_because, rep.rejected_because
    assert out is patch, "the original must come back untouched on failure"


def test_a_gaping_hole_is_refused_rather_than_hollowed():
    """Too wide to bridge: the honest answer is no, not a husk."""
    out, rep = MeshSolidify.solidify(_holed_sphere(), pitch=0.3)
    assert not rep.ok
    assert "hollow shell" in rep.rejected_because, rep.rejected_because


def test_the_gate_still_refuses_a_flat_patch():
    """End to end: the fallback must not open a hole in the gate."""
    try:
        MeshRepair.ensure_solid(_flat_patch())
    except MeshRepairError as exc:
        assert "Traceback" not in str(exc)
        assert "try it again" in str(exc)
    else:
        raise AssertionError("a flat patch must not pass the gate")


def test_a_refused_rebuild_explains_itself():
    try:
        MeshRepair.ensure_solid(_holed_sphere(), solidify_pitch=0.3)
    except MeshRepairError as exc:
        assert "hollow shell" in str(exc), str(exc)
        assert exc.report.solidify is not None and not exc.report.solidify.ok
    else:
        raise AssertionError("a leaking rebuild must not pass the gate")


# --------------------------------------------------------------------------- #
#  how it fits the gate
# --------------------------------------------------------------------------- #
def test_a_valid_solid_never_reaches_the_fallback():
    m = _box()
    out, rep = MeshRepair.ensure_solid(m)
    assert out is m
    assert rep.solidify is None and not rep.approximated


def test_a_repairable_mesh_is_repaired_not_rasterised():
    """Unwelded vertices must take the cheap exact path, not the lossy one."""
    box = _box()
    out, rep = MeshRepair.ensure_solid(_unwelded(box))
    assert rep.ok and not rep.approximated
    assert rep.solidify is None
    assert np.isclose(out.volume, box.volume), "the exact surface was lost"


def test_the_fallback_is_refusable():
    """--no-solidify must restore the strict behaviour exactly."""
    try:
        MeshRepair.ensure_solid(_holed_sphere(), allow_solidify=False)
    except MeshRepairError as exc:
        assert exc.report is not None and exc.report.solidify is None
    else:
        raise AssertionError("allow_solidify=False must still refuse")


def test_an_approximated_repair_says_so():
    out, rep = MeshRepair.ensure_solid(_holed_sphere(), solidify_pitch=1.0)
    assert rep.ok and rep.approximated
    assert "approximation" in rep.summary()
    assert rep.after["watertight"] and rep.after["open_edges"] == 0
    assert "voxel remesh" in rep.steps
    assert out.is_watertight


def test_report_is_serialisable():
    _, rep = MeshRepair.ensure_solid(_holed_sphere(), solidify_pitch=1.0)
    d = rep.to_dict()
    json.dumps(d)                              # must not raise
    assert d["approximated"] and d["ok"]
    assert d["solidify"]["ok"] and d["solidify"]["error_mm"] > 0
    assert d["solidify"]["summary"]


def test_empty_report_summarises_cleanly():
    assert "no voxel remesh needed" in SolidifyReport().summary()


def test_config_exposes_the_switches():
    from app.core.nesting_factory import NesterFactory
    cfg = NesterFactory.config("standard")
    assert cfg.solidify is True and cfg.solidify_pitch is None
    assert NesterFactory.config("standard", solidify=False).solidify is False
    assert "solidify" in cfg.to_dict()


# --------------------------------------------------------------------------- #
#  the point of it all
# --------------------------------------------------------------------------- #
def test_the_result_voxelises_as_a_solid():
    """The reason the gate exists: parity fill must be well defined."""
    from app.core.nesting3d import ScanlineVoxelizer

    solid, rep = MeshSolidify.solidify(_holed_sphere(), pitch=0.5,
                                       close_voxels=4)
    assert rep.ok, rep.summary()
    occ, _, used = ScanlineVoxelizer.solid(solid, pitch=0.5)
    assert used.is_watertight
    assert np.isclose(occ.sum() * 0.5 ** 3, 7176.0, rtol=0.20), occ.sum()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all solidify tests passed")
