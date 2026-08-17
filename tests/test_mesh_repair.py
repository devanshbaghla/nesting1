"""Repair gate: what gets fixed, what gets refused, and what stays untouched.

Each case is a defect that real STLs actually arrive with. The refusal cases
matter as much as the repairs — a mesh that gets "closed" into the wrong shape
is worse than one that is rejected, because the numbers downstream look fine.
"""
import numpy as np
import trimesh

from app.core.mesh_repair import MeshRepair, MeshRepairError


def _box():
    return trimesh.creation.box(extents=(20.0, 30.0, 40.0))


def _unwelded(mesh):
    """Every triangle its own island — what a naive STL writer produces."""
    return trimesh.Trimesh(vertices=mesh.triangles.reshape(-1, 3),
                           faces=np.arange(len(mesh.faces) * 3).reshape(-1, 3),
                           process=False)


def _holed(mesh, n=1):
    """Drop n faces to leave a genuine hole."""
    keep = np.ones(len(mesh.faces), bool)
    keep[:n] = False
    return trimesh.Trimesh(vertices=mesh.vertices.copy(),
                           faces=mesh.faces[keep], process=False)


# --------------------------------------------------------------------------- #
def test_watertight_mesh_is_untouched():
    """The whole point of the gate: valid input must not be rewritten."""
    m = _box()
    out, rep = MeshRepair.ensure_solid(m)
    assert out is m, "a valid solid must be returned as-is, not copied"
    assert not rep.attempted and not rep.repaired and rep.ok
    assert out.volume == m.volume


def test_unwelded_vertices_are_repaired():
    m = _unwelded(_box())
    assert not m.is_watertight
    out, rep = MeshRepair.ensure_solid(m)
    assert rep.repaired and rep.ok and out.is_watertight
    assert "weld duplicate vertices" in rep.steps
    assert np.isclose(out.volume, 20 * 30 * 40, rtol=1e-6)


def test_hole_is_filled():
    m = _holed(_box(), n=1)
    assert not m.is_watertight and MeshRepair.open_edge_count(m) > 0
    out, rep = MeshRepair.ensure_solid(m)
    assert rep.repaired and out.is_watertight
    assert MeshRepair.open_edge_count(out) == 0
    assert np.isclose(out.volume, 20 * 30 * 40, rtol=1e-3)


def test_duplicate_and_degenerate_faces_are_dropped():
    m = _box()
    faces = np.vstack([m.faces, m.faces[:2], [[0, 0, 0]]])
    broken = trimesh.Trimesh(vertices=m.vertices.copy(), faces=faces, process=False)
    out, rep = MeshRepair.ensure_solid(broken)
    assert out.is_watertight and rep.ok
    assert len(out.faces) == len(m.faces)


def test_inverted_solid_is_flipped():
    """Closed but wound inside-out: volume comes back negative."""
    m = _box()
    flipped = trimesh.Trimesh(vertices=m.vertices.copy(),
                              faces=m.faces[:, ::-1], process=False)
    out, _ = MeshRepair.ensure_solid(flipped)
    assert out.volume > 0 and np.isclose(out.volume, 20 * 30 * 40, rtol=1e-6)


def test_whole_missing_face_is_filled_to_the_right_shape():
    """A box short one entire side is still unambiguously that box."""
    m = _holed(_box(), n=2)
    out, rep = MeshRepair.ensure_solid(m)
    assert rep.repaired and out.is_watertight
    assert np.isclose(out.volume, 20 * 30 * 40, rtol=1e-3), out.volume


def test_open_surface_patch_is_refused():
    """A flat sheet is not a solid; capping its outline is not a repair."""
    grid = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    try:
        MeshRepair.ensure_solid(grid)
    except MeshRepairError as exc:
        msg = str(exc)
        assert "closed solid" in msg and "try it again" in msg
        assert "Traceback" not in msg
    else:
        raise AssertionError("a flat patch must not pass the gate")


def test_zero_volume_fill_is_rejected():
    """Two stacked sheets close into a flat "solid" — watertight but not a part.

    This is the case the volume check exists for: hole filling succeeds, the
    mesh reports watertight, and the result encloses nothing.
    """
    v = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float)
    sheet = trimesh.Trimesh(vertices=np.vstack([v, v]),
                            faces=np.array([[0, 1, 2], [0, 2, 3],
                                            [4, 6, 5], [4, 7, 6]]),
                            process=False)
    try:
        out, rep = MeshRepair.ensure_solid(sheet)
    except MeshRepairError as exc:
        assert "closed solid" in str(exc) or "no positive volume" in str(exc)
        return
    raise AssertionError(f"a zero-volume shell must not pass: {rep.summary()}")


def test_repair_is_refusable():
    m = _unwelded(_box())
    try:
        MeshRepair.ensure_solid(m, allow_repair=False)
    except MeshRepairError as exc:
        assert "switched off" in str(exc)
    else:
        raise AssertionError("allow_repair=False must refuse an open mesh")


def test_report_is_serialisable():
    _, rep = MeshRepair.ensure_solid(_unwelded(_box()))
    d = rep.to_dict()
    import json
    json.loads(json.dumps(d))
    assert d["ok"] and d["repaired"] and d["summary"]


def test_voxelisation_survives_an_open_mesh():
    """The end the whole module exists for: rasterise an open mesh safely."""
    from app.core.nesting3d import ScanlineVoxelizer

    m = _unwelded(_box())
    occ, i0, used = ScanlineVoxelizer.solid(m, pitch=1.0)
    assert used.is_watertight
    filled = occ.sum() * 1.0 ** 3
    assert np.isclose(filled, 20 * 30 * 40, rtol=0.05), filled


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all repair tests passed")
