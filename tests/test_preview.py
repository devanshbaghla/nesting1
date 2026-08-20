"""The pre-nesting inspection: what gets called noise, and what must not.

The rule is size only — a fragment goes if its largest dimension is under a
share of the object's smallest. Distance was tried as a second condition and
removed: a headlamp carries hundreds of quarter-millimetre shells flush against
the body, and they are debris whatever they touch.

So the tests that matter are the ones pinning what the threshold is measured
against, and that removal takes exactly the flagged faces and nothing else. The
deletion is applied to the file the run will read, so there is no second chance.
"""
import pathlib

import numpy as np
import trimesh

import app.preview as _pv
from app.preview import (AREA_SHARE, NOISE_RATIO, SWEEP_MEMORY_BUDGET,
                         components, geometry_payload, suggested_pitch,
                         sweep_cost)

OBJ = (40.0, 60.0, 100.0)

#: Tests pin the rule, not the shipped defaults, so they state both
#: thresholds. At this ratio the size limit is 10 mm on the fixture object,
#: and the area limit is loose enough that the dimension test is what decides.
RATIO, AREA = 0.25, 0.01


def classify(mesh, *, ratio=RATIO, area_share=AREA, **kw):
    return _pv.classify(mesh, ratio=ratio, area_share=area_share, **kw)


def drop_noise(mesh, *, ratio=RATIO, area_share=AREA):
    return _pv.drop_noise(mesh, ratio=ratio, area_share=area_share)


def _obj():
    return trimesh.creation.box(extents=OBJ)


def _speck(size=2.0, at=(200.0, 0.0, 0.0)):
    s = trimesh.creation.box(extents=(size, size, size))
    s.apply_translation(at)
    return s


def _scene(*meshes):
    return trimesh.util.concatenate(list(meshes))


# --------------------------------------------------------------------------- #
#  the two halves of the rule
# --------------------------------------------------------------------------- #
def test_a_small_detached_fragment_is_noise():
    _, r = classify(_scene(_obj(), _speck()))
    assert r.bodies == 2
    assert r.noise_bodies == 1
    assert np.isclose(r.object_smallest, 40.0)
    assert np.isclose(r.threshold, 40.0 * RATIO)


def test_a_big_fragment_is_not_noise_however_far_away():
    """Only size decides, so distance cannot condemn a large fragment."""
    _, r = classify(_scene(_obj(), _speck(size=30.0, at=(900.0, 0, 0))))
    assert r.noise_bodies == 0
    assert "over the" in r.fragments[1]["reason"]


def test_a_small_touching_fragment_is_still_noise():
    """Touching the part does not save a speck; this is the point of the change.

    A 2 mm nub welded to the face of the box is debris by size, and size is the
    only test. An earlier version spared it for being attached, which left the
    sub-millimetre shells on a real headlamp in place.
    """
    nub = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))     # sitting on the top face
    _, r = classify(_scene(_obj(), nub))
    assert r.bodies == 2
    assert r.noise_bodies == 1, r.fragments[1]["reason"]


def test_distance_does_not_change_the_verdict():
    """Two identical specks, one welded on and one far off — both go."""
    near = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))
    far = _speck(size=2.0, at=(300.0, 0.0, 0.0))
    _, r = classify(_scene(_obj(), near, far))
    assert r.noise_bodies == 2, [f["reason"] for f in r.fragments]


def test_the_ratio_moves_the_threshold():
    """One ratio cannot suit a 20 mm bracket and a 400 mm housing alike."""
    scene = _scene(_obj(), _speck(size=6.0))              # 6 mm on a 40 mm min
    assert classify(scene, ratio=0.25)[1].noise_bodies == 1     # limit 10
    assert classify(scene, ratio=0.10)[1].noise_bodies == 0     # limit 4
    assert classify(scene, ratio=0.02)[1].threshold == 40.0 * 0.02


def test_a_small_body_carrying_real_surface_is_kept():
    """The guard that stopped a 6,512 mm2 socket being called debris.

    Its box is small against the object, so the dimension test condemns it. Its
    surface is a real share of the part, so the area test saves it -- and that
    is the test that survives the ratio being set wrongly.
    """
    scene = _scene(_obj(), _speck(size=6.0))
    _, r = classify(scene, ratio=0.25, area_share=1e-6)
    assert r.noise_bodies == 0
    assert r.kept_for_area == 1
    assert "carries" in r.fragments[1]["reason"]


def test_both_tests_must_agree_before_anything_is_removed():
    scene = _scene(_obj(), _speck(size=6.0))
    assert classify(scene, ratio=0.25, area_share=0.01)[1].noise_bodies == 1
    assert classify(scene, ratio=0.25, area_share=1e-9)[1].noise_bodies == 0
    assert classify(scene, ratio=0.001, area_share=0.01)[1].noise_bodies == 0


def test_area_is_reported_per_body():
    _, r = classify(_scene(_obj(), _speck()))
    for f in r.fragments:
        assert f["area"] > 0
        assert 0 < f["area_share"] <= 1
    assert r.total_area > 0 and r.area_limit > 0


def test_the_shipped_defaults_are_conservative():
    """2%, not 25%: the 25% limit deleted a quarter of a real headlamp."""
    assert NOISE_RATIO == 0.02
    assert AREA_SHARE == 1e-4


def test_the_gap_is_reported_but_unused():
    """Distance is context for the reader, not an input to the test."""
    _, r = classify(_scene(_obj(), _speck(at=(300.0, 0, 0))))
    flagged = [f for f in r.fragments if f["is_noise"]][0]
    assert flagged["gap"] > 0
    assert "from the main shell" in flagged["reason"]


def test_the_object_itself_is_never_noise():
    _, r = classify(_scene(_obj(), _speck()))
    obj = max(r.fragments, key=lambda f: f["largest"])
    assert not obj["is_noise"]
    assert obj["reason"] == "the object"
    assert obj["gap"] == 0.0


def test_a_single_body_has_nothing_to_do():
    comps, r = classify(_obj())
    assert r.bodies == 1 and r.noise_bodies == 0
    assert not r.has_noise
    assert len(comps) == 1
    assert "single body" in r.summary()


def test_the_largest_fragment_is_the_reference_not_the_whole_file():
    """A distant speck inflates the file's box; the rule must ignore that.

    Measured against the whole file the smallest dimension would be 40 (the
    box is only displaced in x), but on a part displaced in every axis the
    inflation is what lets debris escape. Pinning the reference to the largest
    body keeps the threshold stable however far the speck has drifted.
    """
    near = classify(_scene(_obj(), _speck(at=(80.0, 80.0, 80.0))))[1]
    far = classify(_scene(_obj(), _speck(at=(9000.0, 9000.0, 9000.0))))[1]
    assert np.isclose(near.threshold, far.threshold)
    assert near.noise_bodies == far.noise_bodies == 1


# --------------------------------------------------------------------------- #
#  removal is exact
# --------------------------------------------------------------------------- #
def test_components_partition_the_faces_exactly():
    """split() duplicates bridging faces; face indices must not."""
    mesh = _scene(_obj(), _speck(), _speck(at=(-200.0, 0, 0)))
    comps = components(mesh)
    allf = np.concatenate(comps)
    assert len(allf) == len(mesh.faces)
    assert len(np.unique(allf)) == len(mesh.faces)


def test_drop_noise_removes_exactly_the_flagged_faces():
    mesh = _scene(_obj(), _speck())
    before = len(mesh.faces)
    out, r = drop_noise(mesh)
    assert len(out.faces) == before - r.noise_faces
    assert out.body_count == 1
    # the surviving geometry is the object, untouched
    assert np.allclose(sorted(out.extents), sorted(OBJ))


def test_drop_noise_is_a_no_op_when_there_is_none():
    mesh = _obj()
    out, r = drop_noise(mesh)
    assert out is mesh and not r.has_noise


def test_removal_takes_a_touching_nub_too():
    nub = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))
    mesh = _scene(_obj(), nub)
    out, r = drop_noise(mesh)
    assert len(out.faces) == len(mesh.faces) - r.noise_faces
    assert np.allclose(sorted(out.extents), sorted(OBJ))


# --------------------------------------------------------------------------- #
#  what the viewer receives
# --------------------------------------------------------------------------- #
def test_payload_splits_every_triangle_into_one_group_or_the_other():
    mesh = _scene(_obj(), _speck())
    comps, r = classify(mesh)
    p = geometry_payload(mesh, comps, r)
    part = len(p["part"]) // 9
    noise = sum(len(n["positions"]) // 9 for n in p["noise"])
    assert part + noise == len(mesh.faces)
    assert noise == r.noise_faces


def test_payload_positions_are_flat_triples():
    mesh = _scene(_obj(), _speck())
    comps, r = classify(mesh)
    p = geometry_payload(mesh, comps, r)
    assert len(p["part"]) % 9 == 0            # 3 vertices x 3 floats per face
    assert all(len(n["positions"]) % 9 == 0 for n in p["noise"])
    assert len(p["bounds"]) == 2 and len(p["bounds"][0]) == 3


def test_payload_is_json_safe():
    """`inf` is not JSON; a single-body part used to produce it."""
    import json
    for mesh in (_obj(), _scene(_obj(), _speck())):
        comps, r = classify(mesh)
        json.dumps(r.to_dict())
        json.dumps(geometry_payload(mesh, comps, r))


def test_noise_is_never_decimated():
    """The fragments under inspection go over whole, however small the budget."""
    mesh = _scene(_obj(), _speck())
    comps, r = classify(mesh)
    p = geometry_payload(mesh, comps, r, budget=12)
    assert sum(len(n["positions"]) // 9 for n in p["noise"]) == r.noise_faces


# --------------------------------------------------------------------------- #
#  connectivity: the definition that decides everything downstream
# --------------------------------------------------------------------------- #
def test_faces_joined_only_at_a_vertex_are_one_body():
    """Edge adjacency splits these; shared-vertex connectivity does not.

    Two triangles meeting at a single point are one shell to any person
    looking at them. `mesh.face_adjacency` disagrees, because it only links
    faces across an edge used by exactly two faces. On a headlamp housing with
    266,562 boundary edges that difference reported 7,392 bodies with a median
    size of one triangle, against a true 351 — and since the threshold is
    measured from the largest body, it made the whole rule meaningless.
    """
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],      # first triangle
                  [1, 1, 0], [2, 1, 0]], float)          # shares only vertex 3
    mesh = trimesh.Trimesh(vertices=v, faces=np.array([[0, 1, 2], [2, 3, 4]]),
                           process=False)
    assert len(components(mesh)) == 1


def test_component_count_agrees_with_trimesh():
    mesh = _scene(_obj(), _speck(), _speck(at=(-300.0, 0, 0)))
    assert len(components(mesh)) == mesh.body_count == 3


def test_an_open_shell_is_still_one_body():
    """Boundary edges must not shatter a component."""
    box = _obj()
    keep = np.ones(len(box.faces), bool)
    keep[:2] = False                                    # remove a whole face
    open_box = trimesh.Trimesh(vertices=box.vertices.copy(),
                               faces=box.faces[keep], process=False)
    assert not open_box.is_watertight
    assert len(components(open_box)) == 1


# --------------------------------------------------------------------------- #
#  drawing a mesh too big to send whole
# --------------------------------------------------------------------------- #
def test_decimation_preserves_the_surface():
    """Decimating a triangle soup shreds it; the payload must weld first.

    The unwelded route kept 27% of the surface area on a real part and drew
    something that looked shattered. Anything close to that here means the
    weld was skipped again.
    """
    sphere = trimesh.creation.icosphere(subdivisions=5)   # 20,480 faces
    comps, r = classify(sphere)
    p = geometry_payload(sphere, comps, r, budget=2000)
    drawn = np.array(p["part"]).reshape(-1, 3, 3)
    assert r.decimated_to and len(drawn) <= 2200, len(drawn)

    def area(t):
        return float(np.linalg.norm(
            np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)
    kept = area(drawn) / sphere.area
    assert kept > 0.9, f"decimation kept only {kept:.0%} of the surface"


def test_a_small_mesh_is_sent_whole():
    mesh = _obj()
    comps, r = classify(mesh)
    p = geometry_payload(mesh, comps, r)
    assert r.decimated_to == 0
    assert len(p["part"]) // 9 == len(mesh.faces)


# --------------------------------------------------------------------------- #
#  what a lattice will cost the translation search
# --------------------------------------------------------------------------- #
def test_sweep_cost_grows_as_the_cube_of_resolution():
    """Halving the pitch is eight times the memory, which is the whole trap.

    Measured on a part big enough for the law to show: the grid also carries an
    additive pad for the clearance dilation, and on a 40 mm fixture that pad
    dominates and the ratio comes out at 6.6 rather than 8.
    """
    big = [294.01, 340.07, 423.09]
    coarse = sweep_cost(big, 1.0)
    fine = sweep_cost(big, 0.5)
    assert 7.0 < fine["elements"] / coarse["elements"] < 9.0


def test_padding_dominates_on_a_small_part():
    """...and the estimate must stay honest there, not just asymptotically."""
    ratio = sweep_cost(OBJ, 1.0)["elements"] / sweep_cost(OBJ, 2.0)["elements"]
    assert 5.0 < ratio < 8.0, ratio


def test_a_large_part_at_a_fine_pitch_is_over_budget():
    """The headlamp that failed: 294 x 340 x 423 mm at 0.5 mm wants 10.8 GB."""
    cost = sweep_cost([294.01, 340.07, 423.09], 0.5)
    assert not cost["within_budget"]
    assert 10.0 < cost["bytes"] / 2 ** 30 < 12.0
    assert cost["shape"] == [1209, 1393, 1725]


def test_a_small_part_at_a_fine_pitch_is_fine():
    assert sweep_cost([28.0, 35.0, 123.0], 0.5)["within_budget"]


def test_the_suggestion_always_fits():
    for ext in ([294.01, 340.07, 423.09], [899.0, 208.4, 187.9],
                [28.0, 35.0, 123.0], [2000.0, 2000.0, 2000.0]):
        pitch = suggested_pitch(ext)
        assert sweep_cost(ext, pitch)["within_budget"], (ext, pitch)


def test_the_suggestion_is_not_needlessly_coarse():
    """One step finer should be over budget, unless already at the finest step."""
    for ext in ([294.01, 340.07, 423.09], [899.0, 208.4, 187.9]):
        pitch = suggested_pitch(ext)
        assert sweep_cost(ext, pitch / 2)["bytes"] > SWEEP_MEMORY_BUDGET, ext


def test_a_non_positive_pitch_is_refused():
    for bad in (0.0, -1.0):
        try:
            sweep_cost(OBJ, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-positive pitch must be refused")


# --------------------------------------------------------------------------- #
#  the per-body endpoint the viewer clicks
# --------------------------------------------------------------------------- #
def _client_with(mesh):
    """A preview job holding `mesh`, and its id.

    These go through the real endpoints, so they exercise the *shipped*
    defaults rather than the loose thresholds the unit tests pin. A 0.5 mm
    speck is debris by those defaults; a 2 mm one is not.
    """
    import tempfile, warnings
    warnings.filterwarnings("ignore")
    from fastapi.testclient import TestClient
    from app.main import app

    path = pathlib.Path(tempfile.mkdtemp()) / "part.stl"
    mesh.export(str(path))
    client = TestClient(app)
    with open(path, "rb") as fh:
        r = client.post("/api/preview",
                        files={"file": ("part.stl", fh, "application/octet-stream")},
                        data={"profile": "quick"})
    assert r.status_code == 200, r.text
    return client, r.json()["job_id"], r.json()["preview"]


def test_each_body_can_be_fetched_on_its_own():
    client, job, rep = _client_with(_scene(_obj(), _speck(size=0.5)))
    for frag in rep["fragments"]:
        r = client.get(f"/api/preview/{job}/body/{frag['index']}")
        assert r.status_code == 200
        body = r.json()
        assert len(body["positions"]) // 9 == frag["faces"]
        assert body["fragment"]["reason"] == frag["reason"]
        assert len(body["bounds"]) == 2


def test_a_body_is_sent_at_full_resolution():
    """Clicking a fragment is how you see its shape; it must not be decimated."""
    client, job, rep = _client_with(_scene(_obj(), _speck(size=0.5)))
    noise = [f for f in rep["fragments"] if f["is_noise"]][0]
    body = client.get(f"/api/preview/{job}/body/{noise['index']}").json()
    assert len(body["positions"]) // 9 == noise["faces"]


def test_an_unknown_body_is_a_clean_404():
    client, job, rep = _client_with(_scene(_obj(), _speck(size=0.5)))
    r = client.get(f"/api/preview/{job}/body/999")
    assert r.status_code == 404
    assert "999" in r.json()["detail"]


def test_body_indices_follow_the_file_after_denoise():
    """The analysis is cached; denoising has to invalidate it."""
    client, job, rep = _client_with(_scene(_obj(), _speck(size=0.5)))
    assert rep["bodies"] == 2
    before = client.get(f"/api/preview/{job}/body/1").json()
    assert before["positions"]

    client.post(f"/api/preview/{job}/denoise")
    after = client.get(f"/api/preview/{job}/geometry").json()
    assert after["noise"] == []
    assert client.get(f"/api/preview/{job}/body/1").status_code == 404


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all preview tests passed")
