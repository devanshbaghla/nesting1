"""Upload path for imperfect STLs: repaired ones run, hopeless ones say why."""
import time
from pathlib import Path

import numpy as np
import trimesh
from fastapi.testclient import TestClient

from app.main import app

STL = Path(__file__).resolve().parent.parent / "data" / "sample.stl"


def _post(client, name, mesh_bytes, **form):
    data = {"clearance": 5.0, "top_n": 2, "profile": "quick",
            "objective": "volume", "refiner": "none"}
    data.update(form)
    return client.post("/api/jobs",
                       files={"file": (name, mesh_bytes, "model/stl")}, data=data)


def _holed_bytes(path, n=3):
    """The same part with n triangles deleted, leaving genuine holes.

    Note this is deliberately *not* an unwelded mesh: trimesh welds vertices
    when it loads a file, so that defect — the most common one in the wild —
    never survives the round trip and cannot be tested through the API. It is
    covered in-memory by tests/test_mesh_repair.py instead.
    """
    m = trimesh.load(str(path), force="mesh")
    keep = np.ones(len(m.faces), bool)
    keep[::len(m.faces) // n] = False
    broken = trimesh.Trimesh(vertices=m.vertices.copy(), faces=m.faces[keep],
                             process=False)
    assert not broken.is_watertight
    return broken.export(file_type="stl")


def _patch_bytes():
    """A flat two-triangle sheet: not a solid under any repair."""
    grid = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]], float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]), process=False)
    return grid.export(file_type="stl")


def test_open_mesh_is_repaired_and_accepted():
    c = TestClient(app)
    r = _post(c, "open.stl", _holed_bytes(STL))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mesh"]["repaired"] is True
    rep = body["mesh"]["repair"]
    assert rep["ok"] and "fill holes" in rep["steps"]
    assert rep["before"]["open_edges"] > 0 and rep["after"]["open_edges"] == 0

    jid = body["job_id"]
    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert j["status"] == "done", j.get("error")
    assert len(j["recommendations"]) == 2
    for rec in j["recommendations"]:
        assert rec["gap"] >= 5.0 - 1e-3, "clearance violated on a repaired mesh"
    assert any("not a closed solid" in line for line in j["log"]), \
        "the user must be told their mesh was modified"
    print("repaired upload nested OK:", rep["summary"])


def test_unrepairable_mesh_is_refused_clearly():
    c = TestClient(app)
    r = _post(c, "patch.stl", _patch_bytes())
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert isinstance(detail, str), "the UI shows detail verbatim; it must be text"
    assert "closed solid" in detail
    assert "Traceback" not in detail and "Error" not in detail
    assert len(detail) < 500, "an error panel is not a place for an essay"
    print("refusal message:", detail)


def test_watertight_upload_is_unchanged():
    """The regression guard: a valid solid must not be touched or annotated."""
    c = TestClient(app)
    with STL.open("rb") as fh:
        r = _post(c, "sample.stl", fh.read())
    assert r.status_code == 200, r.text
    mesh = r.json()["mesh"]
    assert mesh["repaired"] is False
    assert mesh["repair"]["attempted"] is False
    original = trimesh.load(str(STL), force="mesh")
    assert np.isclose(mesh["volume"], round(float(original.volume), 1))
    assert mesh["faces"] == len(original.faces)


if __name__ == "__main__":
    test_watertight_upload_is_unchanged()
    print("ok   watertight upload unchanged")
    test_unrepairable_mesh_is_refused_clearly()
    print("ok   unrepairable refused")
    test_open_mesh_is_repaired_and_accepted()
    print("ok   open mesh repaired and nested")
