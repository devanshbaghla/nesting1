"""End-to-end smoke test: upload, poll to completion, fetch every artefact."""
import io, time, zipfile
from pathlib import Path
from fastapi.testclient import TestClient
from app.main import app

STL = Path(__file__).resolve().parent.parent / "data" / "sample.stl"


def test_pipeline(tmp_path=None):
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

    for view in ("iso", "top", "bottom", "front"):
        im = c.get(f"/api/jobs/{jid}/rec/1/image/{view}")
        assert im.status_code == 200 and im.content[:4] == b"\x89PNG"
    st = c.get(f"/api/jobs/{jid}/rec/1/stl")
    assert st.status_code == 200 and len(st.content) > 1000
    z = c.get(f"/api/jobs/{jid}/rec/1/views.zip")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert sum(n.endswith(".png") for n in names) >= 4
    assert zipfile.ZipFile(io.BytesIO(c.get(f"/api/jobs/{jid}/all.zip").content).__class__ and io.BytesIO(c.get(f"/api/jobs/{jid}/all.zip").content)).namelist()
    print("OK", len(recs), "recommendations;", names)


if __name__ == "__main__":
    test_pipeline()
