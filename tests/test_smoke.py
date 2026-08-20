"""End-to-end smoke test: upload, poll to completion, fetch every artefact."""
import io, time, zipfile
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

STL = Path(__file__).resolve().parent.parent / "data" / "sample.stl"

GLB_MAGIC = b"glTF"


def test_pipeline():
    c = TestClient(app)
    assert c.get("/api/health").json()["ok"]
    assert c.get("/").status_code == 200
    assert len(c.get("/api/algorithms").json()) > 15

    with STL.open("rb") as fh:
        r = c.post("/api/jobs", files={"file": ("sample.stl", fh, "model/stl")},
                   data={"clearance": 5.0, "top_n": 3, "profile": "quick",
                         "objective": "volume", "refiner": "none"})
    assert r.status_code == 200, r.text
    jid = r.json()["job_id"]

    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert j["status"] == "done", j.get("error")
    recs = j["recommendations"]
    assert len(recs) == 3
    for rec in recs:
        assert rec["gap"] >= 5.0 - 1e-3, "clearance violated"
        assert rec["glb"].endswith(".glb"), "every recommendation carries a model"

    # the interactive model, which replaced seven server-rendered PNG views
    m = c.get(f"/api/jobs/{jid}/rec/1/model.glb")
    assert m.status_code == 200
    assert m.content[:4] == GLB_MAGIC, "not a glTF-binary payload"
    assert m.headers["content-type"] == "model/gltf-binary"

    st = c.get(f"/api/jobs/{jid}/rec/1/stl")
    assert st.status_code == 200 and len(st.content) > 1000

    names = zipfile.ZipFile(
        io.BytesIO(c.get(f"/api/jobs/{jid}/all.zip").content)).namelist()
    assert sum(n.endswith(".glb") for n in names) == 3
    assert sum(n.endswith(".stl") for n in names) == 3
    assert not [n for n in names if n.endswith(".png")], "no images anywhere"
    print("OK", len(recs), "recommendations;", names)


def test_the_model_has_two_separately_coloured_copies():
    """The two copies must stay distinguishable, or an interlock is unreadable."""
    import trimesh
    from app.core.nesting3d import Preview

    a = trimesh.creation.box((20, 10, 10))
    b = trimesh.creation.box((20, 10, 10)); b.apply_translation([25, 0, 0])
    out = Path(__file__).resolve().parent / "_smoke_pair.glb"
    try:
        Preview.pair_glb([a, b], str(out))
        assert out.read_bytes()[:4] == GLB_MAGIC
        scene = trimesh.load(str(out))
        assert sorted(scene.geometry) == ["copy_A", "copy_B"]
        colours = {tuple(g.visual.material.baseColorFactor[:3])
                   for g in scene.geometry.values()}
        assert len(colours) == 2, f"copies share a colour: {colours}"
    finally:
        out.unlink(missing_ok=True)


def test_no_module_imports_matplotlib():
    """The dependency is gone, not merely unused."""
    import subprocess, sys, textwrap
    probe = textwrap.dedent("""
        import sys
        class Blocked:
            def find_module(self, name, path=None):
                if name.split('.')[0] in ('matplotlib', 'mpl_toolkits'):
                    raise ImportError(f'{name} must not be imported')
        sys.meta_path.insert(0, Blocked())
        import app.main, app.jobs, app.core.nest_base, app.core.nesting3d
        print('clean')
    """)
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-c", probe], cwd=root,
                       capture_output=True, text=True)
    assert "clean" in r.stdout, r.stdout + r.stderr


if __name__ == "__main__":
    test_the_model_has_two_separately_coloured_copies()
    test_no_module_imports_matplotlib()
    test_pipeline()
