"""The pre-nesting inspection: what gets called noise, and what must not.

The rule has two halves and the interesting tests are the ones that check the
halves are actually independent — that a tiny fragment welded to the part
survives, and that a distant one only dies if it is also small. Getting either
wrong deletes real geometry, and the deletion is applied to the file the run
will read, so there is no second chance.
"""
import numpy as np
import trimesh

from app.preview import (NOISE_RATIO, TOUCH_RATIO, attached_set, classify,
                         components, drop_noise, geometry_payload)

OBJ = (40.0, 60.0, 100.0)          # smallest dimension 40 -> noise under 10


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
    assert np.isclose(r.threshold, 40.0 * NOISE_RATIO)


def test_a_big_detached_fragment_is_not_noise():
    """Distance alone must not condemn anything."""
    _, r = classify(_scene(_obj(), _speck(size=30.0)))
    assert r.noise_bodies == 0
    assert "too big" in r.fragments[1]["reason"]


def test_a_small_touching_fragment_is_kept():
    """The refinement that matters: attached geometry is part of the part.

    A 2 mm nub on the face of the box is well under the size threshold, and
    would be deleted by size alone. It is welded to the object, so it stays.
    """
    nub = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))     # sitting on the top face
    _, r = classify(_scene(_obj(), nub))
    assert r.bodies == 2
    assert r.noise_bodies == 0, r.fragments[1]["reason"]
    assert r.attached_kept == 1
    assert "touching" in r.fragments[1]["reason"]


def test_the_gap_decides_between_two_identical_specks():
    """Same size, different distance — only the far one goes."""
    near = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))
    far = _speck(size=2.0, at=(300.0, 0.0, 0.0))
    _, r = classify(_scene(_obj(), near, far))
    flagged = [f for f in r.fragments if f["is_noise"]]
    assert len(flagged) == 1
    assert flagged[0]["gap"] > r.touch_tolerance
    assert r.attached_kept == 1


def test_touch_tolerance_scales_with_the_object():
    _, r = classify(_scene(_obj(), _speck()))
    assert np.isclose(r.touch_tolerance, 40.0 * TOUCH_RATIO)


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


def test_removal_keeps_a_touching_nub():
    nub = _speck(size=2.0, at=(0.0, 0.0, OBJ[2] / 2))
    mesh = _scene(_obj(), nub)
    out, r = drop_noise(mesh)
    assert len(out.faces) == len(mesh.faces), "a welded nub must survive"


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
#  attachment is transitive
# --------------------------------------------------------------------------- #
def test_a_speck_attached_through_a_chain_is_kept():
    """A clip on a bracket on the housing is attached to the part.

    The speck's own box is nowhere near the object's, so measuring attachment
    against the largest shell alone condemns it. On the reference headlamp that
    single change is the difference between 46 bodies offered for deletion and
    none: every one of the 46 was joined to the part through other bodies.
    """
    bridge = trimesh.creation.box(extents=(30.0, 4.0, 4.0))
    bridge.apply_translation((OBJ[0] / 2 + 15.0, 0.0, 0.0))   # touches the box
    tip = _speck(size=2.0, at=(OBJ[0] / 2 + 31.0, 0.0, 0.0))  # touches the bridge
    _, r = classify(_scene(_obj(), bridge, tip))

    assert r.bodies == 3
    assert r.noise_bodies == 0, [f["reason"] for f in r.fragments]
    assert r.attached_bodies == 3
    assert any("through other bodies" in f["reason"] for f in r.fragments)


def test_a_chain_that_does_not_reach_still_leaves_noise():
    """Transitivity must not become 'keep everything'."""
    bridge = trimesh.creation.box(extents=(30.0, 4.0, 4.0))
    bridge.apply_translation((OBJ[0] / 2 + 15.0, 0.0, 0.0))
    stray = _speck(size=2.0, at=(400.0, 0.0, 0.0))            # touches nothing
    _, r = classify(_scene(_obj(), bridge, stray))
    assert r.noise_bodies == 1
    flagged = [f for f in r.fragments if f["is_noise"]][0]
    assert "free-standing" in flagged["reason"]


def test_attached_set_walks_the_whole_chain():
    boxes = [(np.array([10.0, 10, 10]), np.array([0.0, 0, 0])),
             (np.array([10.0, 10, 10]), np.array([10.0, 0, 0])),
             (np.array([10.0, 10, 10]), np.array([20.0, 0, 0])),
             (np.array([10.0, 10, 10]), np.array([500.0, 0, 0]))]
    reached = attached_set(boxes, [0], tol=0.01)
    assert reached == {0, 1, 2}, reached


def test_attachment_seeds_from_every_big_body_not_just_the_largest():
    """A speck touching a second large body is attached to the part."""
    other = trimesh.creation.box(extents=(30.0, 30.0, 30.0))
    other.apply_translation((300.0, 0.0, 0.0))
    tip = _speck(size=2.0, at=(300.0, 0.0, 16.0))             # sits on `other`
    _, r = classify(_scene(_obj(), other, tip))
    assert r.noise_bodies == 0, [f["reason"] for f in r.fragments]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all preview tests passed")
