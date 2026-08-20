"""
FastAPI application: upload an STL, get the top N nested arrangements back,
each as an interactive glTF-binary model you can orbit in the browser.

    uvicorn app.main:app --reload

Routes
------
GET  /                                       upload page + results UI
POST /api/preview                            upload an STL, inspect it, nest nothing
GET  /api/preview/{id}/geometry              triangles, split into part and noise
GET  /api/preview/{id}/classify              re-run the rule at another ratio
GET  /api/preview/{id}/body/{i}              one body's triangles, for highlighting
POST /api/preview/{id}/denoise               drop the noise fragments
POST /api/preview/{id}/nest                  accept the part and queue the run
POST /api/jobs                               upload an STL, returns a job id
GET  /api/jobs                               list jobs
GET  /api/jobs/{id}                          status, progress, recommendations
DELETE /api/jobs/{id}                        remove a job and its artefacts
GET  /api/jobs/{id}/rec/{rank}/model.glb     the nested pair, two coloured nodes
GET  /api/jobs/{id}/rec/{rank}/stl           download the nested STL
GET  /api/jobs/{id}/report                   full JSON report
GET  /api/algorithms                         the algorithm registry
GET  /api/health
"""

from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from . import config, preview
from .core.nesting3d import HAVE_FCL
from .core.nesting_factory import PROFILES, AlgorithmRegistry
from .jobs import store, validate_upload

BASE = Path(__file__).resolve().parent
app = FastAPI(title="STL Nesting Service", version="1.0.0",
              description="Nest two copies of a part with a guaranteed "
                          "minimum surface clearance.")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


# --------------------------------------------------------------------------- #
#  Pages
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "profiles": sorted(PROFILES),
        "objectives": AlgorithmRegistry.names("objective"),
        "refiners": AlgorithmRegistry.names("refiner"),
        "backends": [b for b in AlgorithmRegistry.names("distance_backend")
                     if b != "bvh" or HAVE_FCL],
        "defaults": {"profile": config.DEFAULT_PROFILE,
                     "clearance": config.DEFAULT_CLEARANCE,
                     "top_n": config.DEFAULT_TOP_N,
                     "backend": config.DEFAULT_DISTANCE_BACKEND},
    })


# --------------------------------------------------------------------------- #
#  Jobs
# --------------------------------------------------------------------------- #
@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    clearance: float = Form(config.DEFAULT_CLEARANCE),
    top_n: int = Form(config.DEFAULT_TOP_N),
    profile: str = Form(config.DEFAULT_PROFILE),
    objective: str = Form("volume"),
    refiner: str = Form("descend"),
    distance_backend: str = Form(config.DEFAULT_DISTANCE_BACKEND),
):
    params = _validated_params(file, clearance, top_n, profile, objective,
                               refiner, distance_backend)
    name = Path(file.filename or "part.stl").name
    job = store.create(name, params)
    dest = job.dir / name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        stats = validate_upload(dest)
    except ValueError as exc:
        shutil.rmtree(job.dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc

    job.audit = stats
    _note_intake(job, stats)
    store.submit(job)
    return {"job_id": job.id, "status": job.status, "mesh": stats}


def _validated_params(file: UploadFile, clearance: float, top_n: int,
                      profile: str, objective: str, refiner: str,
                      distance_backend: str) -> dict:
    """Reject a bad request before a byte is written to disk."""
    name = Path(file.filename or "part.stl").name
    if not name.lower().endswith(".stl"):
        raise HTTPException(400, "please upload a .stl file")
    if profile not in PROFILES:
        raise HTTPException(400, f"unknown profile; choose from {sorted(PROFILES)}")
    if objective not in AlgorithmRegistry.names("objective"):
        raise HTTPException(400, "unknown objective")
    if refiner not in AlgorithmRegistry.names("refiner"):
        raise HTTPException(400, "unknown refiner")
    if distance_backend not in AlgorithmRegistry.names("distance_backend"):
        raise HTTPException(400, "unknown distance backend")
    if distance_backend == "bvh" and not HAVE_FCL:
        raise HTTPException(400, "the bvh distance backend needs python-fcl, "
                                 "which is not installed on this server")
    if not 1 <= top_n <= 20:
        raise HTTPException(400, "top_n must be between 1 and 20")
    if clearance < 0:
        raise HTTPException(400, "clearance must be positive")
    return {"clearance": clearance, "top_n": top_n, "profile": profile,
            "objective": objective, "refiner": refiner,
            "distance_backend": distance_backend}


def _note_intake(job, stats: dict) -> None:
    """Put anything intake changed about the mesh into the job's own log."""
    if stats.get("denoised"):
        job.log.append(f"input mesh carried loose fragments; "
                       f"{stats['denoise']['summary']}")
    if stats.get("repaired"):
        # the nested output is derived from the repaired solid, not the file
        # that was uploaded — say so where the user is already looking
        job.log.append(f"input mesh was not a closed solid; "
                       f"{stats['repair']['summary']}")
    if stats.get("approximated"):
        # a rasterised part is a different claim from a repaired one: every
        # dimension below carries the grid tolerance, so it gets its own line
        job.log.append(
            f"the nested geometry is a rasterisation of the upload, accurate "
            f"to +/-{stats['repair']['solidify']['error_mm']:.3f} (file "
            f"units); dimensions and clearances inherit that tolerance")


# --------------------------------------------------------------------------- #
#  Preview: look at the part, and its debris, before committing to a run
# --------------------------------------------------------------------------- #
@app.post("/api/preview")
async def create_preview(
    file: UploadFile = File(...),
    clearance: float = Form(config.DEFAULT_CLEARANCE),
    top_n: int = Form(config.DEFAULT_TOP_N),
    profile: str = Form(config.DEFAULT_PROFILE),
    objective: str = Form("volume"),
    refiner: str = Form("descend"),
    distance_backend: str = Form(config.DEFAULT_DISTANCE_BACKEND),
):
    """Upload and inspect, without starting anything.

    The job directory and TTL machinery are reused, so the file is stored once
    and the eventual run reads the same bytes the preview drew — including any
    denoising the user agreed to.
    """
    params = _validated_params(file, clearance, top_n, profile, objective,
                               refiner, distance_backend)
    name = Path(file.filename or "part.stl").name
    job = store.create(name, params)
    job.status = job.stage = "preview"
    dest = job.dir / name
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    try:
        mesh = preview.load_mesh(dest)
    except Exception as exc:
        shutil.rmtree(job.dir, ignore_errors=True)
        store._jobs.pop(job.id, None)
        raise HTTPException(422, f"could not parse as STL: {exc}") from exc
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        shutil.rmtree(job.dir, ignore_errors=True)
        store._jobs.pop(job.id, None)
        raise HTTPException(422, "no triangles found in the file")

    _, report = preview.classify(mesh, filename=name)
    _advise_pitch(report, mesh, params["clearance"])
    job.audit = {"preview": report.to_dict()}
    return {"job_id": job.id, "preview": report.to_dict()}


#: analysed meshes, keyed by job. Splitting a 1.4M-face part takes over a
#: second, and the viewer asks again for every body the user clicks; the file's
#: mtime is the key's second half so denoising invalidates it for free.
_ANALYSED: dict[str, tuple] = {}


def _analysed(job, ratio: float | None = None):
    """``(mesh, components, report)`` for a job, re-used across requests.

    The key carries the ratio as well as the file's mtime, so dragging the
    threshold re-classifies without re-reading a 71 MB STL from disk.
    """
    path = _source_stl(job)
    ratio = preview.NOISE_RATIO if ratio is None else float(ratio)
    stamp = (path.stat().st_mtime_ns, round(ratio, 6))
    hit = _ANALYSED.get(job.id)
    if hit and hit[0] == stamp:
        return hit[1:]
    mesh = hit[1] if hit and hit[0][0] == stamp[0] else preview.load_mesh(path)
    comps, report = preview.classify(mesh, ratio=ratio, filename=job.filename)
    _ANALYSED[job.id] = (stamp, mesh, comps, report)
    return mesh, comps, report


def _ratio(value: float) -> float:
    if not 0.0 < value < 1.0:
        raise HTTPException(400, "ratio must be between 0 and 1 (exclusive)")
    return float(value)



def _advise_pitch(report, mesh, clearance: float) -> None:
    """Attach what each candidate lattice would cost the translation search.

    The sweep allocates one array per orientation sized by the part and the
    pitch, and on a large part at the default 0.5 mm that is 10.8 GB — which
    numpy reports as a bare allocation failure with nothing to act on. Costing
    the options here means the choice is made with the numbers in view.
    """
    report.suggested_pitch = preview.suggested_pitch(mesh.extents, clearance)
    report.pitch_options = [
        preview.sweep_cost(mesh.extents, p, clearance)
        for p in (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]


@app.get("/api/preview/{job_id}/geometry")
def preview_geometry(job_id: str, ratio: float = preview.NOISE_RATIO):
    """Triangles for the viewer, split into the part and each noise fragment."""
    job = _job(job_id)
    mesh, comps, report = _analysed(job, _ratio(ratio))
    return JSONResponse(preview.geometry_payload(mesh, comps, report))


@app.get("/api/preview/{job_id}/classify")
def preview_reclassify(job_id: str, ratio: float = preview.NOISE_RATIO):
    """Re-run the rule at a different threshold, without redrawing anything."""
    job = _job(job_id)
    _, _, report = _analysed(job, _ratio(ratio))
    job.audit = {**job.audit, "preview": report.to_dict()}
    return {"job_id": job.id, "preview": report.to_dict()}


@app.get("/api/preview/{job_id}/body/{index}")
def preview_body(job_id: str, index: int,
                 ratio: float = preview.NOISE_RATIO):
    """One body's triangles, at full resolution, for highlighting it.

    Fetched on demand rather than shipped with the rest: a part can have
    hundreds of bodies and the viewer only ever highlights one at a time. Not
    decimated either — the whole point of clicking a 0.4 mm fragment is to see
    its actual shape.
    """
    job = _job(job_id)
    mesh, comps, report = _analysed(job, _ratio(ratio))
    if not 0 <= index < len(comps):
        raise HTTPException(404, f"no body {index}; this part has {len(comps)}")
    faces = comps[index]
    tris = np.asarray(mesh.triangles[faces], dtype=np.float32)
    lo, hi = tris.reshape(-1, 3).min(axis=0), tris.reshape(-1, 3).max(axis=0)
    return JSONResponse({
        "index": index,
        "positions": tris.reshape(-1).tolist(),
        "bounds": [[float(v) for v in lo], [float(v) for v in hi]],
        "fragment": report.fragments[index],
    })


@app.post("/api/preview/{job_id}/denoise")
def preview_denoise(job_id: str,
                    ratio: float = preview.NOISE_RATIO):
    """Drop the fragments the rule called noise, and rewrite the upload.

    Destructive by intent — this is the button that says yes. The file the run
    will read is replaced, so what was drawn is what gets nested.
    """
    job = _job(job_id)
    path = _source_stl(job)
    mesh = preview.load_mesh(path)
    cleaned, report = preview.drop_noise(mesh, ratio=_ratio(ratio))
    if not report.has_noise:
        return {"job_id": job.id, "removed": 0,
                "preview": report.to_dict(), "note": "nothing matched the rule"}

    cleaned.export(str(path))
    _ANALYSED.pop(job.id, None)
    _, after = preview.classify(preview.load_mesh(path), ratio=_ratio(ratio),
                                filename=job.filename)
    job.audit = {"preview": after.to_dict(), "denoised_from": report.to_dict()}
    job.log.append(f"removed {report.noise_bodies} noise "
                   f"{'fragment' if report.noise_bodies == 1 else 'fragments'} "
                   f"({report.noise_faces:,} triangles) before nesting")
    return {"job_id": job.id, "removed": report.noise_bodies,
            "removed_faces": report.noise_faces, "preview": after.to_dict()}


@app.post("/api/preview/{job_id}/nest")
def preview_nest(job_id: str, coarse_pitch: float | None = None,
                 fine_pitch: float | None = None):
    """Accept the part as previewed and queue the run.

    The lattice is settled here rather than at upload because this is the
    first point where the part's size is known, and size is what decides
    whether a pitch is affordable. A pitch that cannot fit is refused now,
    with the number, instead of failing minutes later inside the sweep.
    """
    job = _job(job_id)
    if job.status != "preview":
        raise HTTPException(409, f"job is already {job.status}")
    pitches = _checked_pitches(job, coarse_pitch, fine_pitch)
    job.params.update(pitches)
    try:
        stats = validate_upload(_source_stl(job))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    job.audit = {**job.audit, **stats}
    _note_intake(job, stats)
    job.status, job.stage = "queued", "queued"
    if pitches.get("coarse_pitch") or pitches.get("fine_pitch"):
        job.log.append(
            f"voxel pitch set to coarse {pitches['coarse_pitch']} / "
            f"fine {pitches['fine_pitch']} (file units)")
    store.submit(job)
    return {"job_id": job.id, "status": job.status, "mesh": stats,
            "pitch": pitches}


def _checked_pitches(job, coarse: float | None, fine: float | None) -> dict:
    """Validate a requested lattice against what the sweep can allocate."""
    if coarse is None and fine is None:
        return {}
    mesh = preview.load_mesh(_source_stl(job))
    clearance = float(job.params["clearance"])
    coarse = float(coarse if coarse else fine * 2)
    fine = float(fine if fine else coarse / 2)
    if not 0 < fine <= coarse:
        raise HTTPException(
            400, "fine pitch must be positive and no larger than the coarse "
                 f"pitch (got fine {fine}, coarse {coarse})")
    for label, pitch in (("coarse", coarse), ("fine", fine)):
        cost = preview.sweep_cost(mesh.extents, pitch, clearance)
        if not cost["within_budget"]:
            suggest = preview.suggested_pitch(mesh.extents, clearance)
            raise HTTPException(
                400,
                f"a {label} pitch of {pitch} needs "
                f"{cost['bytes'] / 2 ** 30:.1f} GB for one translation "
                f"search on this part ({cost['shape'][0]} x "
                f"{cost['shape'][1]} x {cost['shape'][2]} voxels), over the "
                f"{preview.SWEEP_MEMORY_BUDGET / 2 ** 30:.0f} GB budget. "
                f"Try {suggest} or coarser.")
    return {"coarse_pitch": coarse, "fine_pitch": fine}


def _source_stl(job) -> Path:
    path = job.dir / job.filename
    if not path.exists():
        raise HTTPException(404, "the uploaded file is no longer on disk")
    return path


@app.get("/api/jobs")
def list_jobs():
    return [{"id": j.id, "filename": j.filename, "status": j.status,
             "created": j.created, "n": len(j.recommendations)}
            for j in store.all()]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = _job(job_id)
    return job.public()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job(job_id)
    shutil.rmtree(job.dir, ignore_errors=True)
    store._jobs.pop(job_id, None)
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/report")
def report(job_id: str):
    job = _job(job_id)
    path = job.dir / "recommendations.json"
    if not path.exists():
        raise HTTPException(404, "report not ready")
    return JSONResponse(json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
#  Per-recommendation artefacts
# --------------------------------------------------------------------------- #
@app.get("/api/jobs/{job_id}/rec/{rank}/model.glb")
def rec_model(job_id: str, rank: int):
    """The nested pair as glTF-binary, for the viewer in the results page.

    Written during export, not on request: the pipeline serialises it while
    both copies are still in memory, so this route only ever reads a file off
    disk. It replaced seven server-rendered PNG viewpoints -- the browser can
    reach any of them, and more, by orbiting.
    """
    job, rec = _rec(job_id, rank)
    name = rec.get("glb")
    if not name:
        raise HTTPException(404, "this job has no model for that rank")
    path = job.dir / name
    if not path.exists():
        raise HTTPException(404, "model not found")
    return FileResponse(path, media_type="model/gltf-binary",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{job_id}/rec/{rank}/stl")
def rec_stl(job_id: str, rank: int):
    job, rec = _rec(job_id, rank)
    path = job.dir / rec["stl"]
    if not path.exists():
        raise HTTPException(404, "STL not found")
    stem = Path(job.filename).stem
    return FileResponse(path, media_type="model/stl",
                        filename=f"{stem}_nested_{rank:02d}.stl")


@app.get("/api/jobs/{job_id}/all.zip")
def all_zip(job_id: str):
    """Every recommendation: STLs, interactive GLB models and the report."""
    job = _job(job_id)
    if job.status != "done":
        raise HTTPException(409, "job is not finished")
    stem = Path(job.filename).stem
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in job.recommendations:
            for key in ("stl", "glb"):
                name = rec.get(key)
                if name and (job.dir / name).exists():
                    zf.write(job.dir / name, name)
        for extra in ("recommendations.json", "recommendations.md"):
            p = job.dir / extra
            if p.exists():
                zf.write(p, extra)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{stem}_all_nests.zip"'})


# --------------------------------------------------------------------------- #
#  Meta
# --------------------------------------------------------------------------- #
@app.get("/api/algorithms")
def algorithms(status: str | None = None):
    return [{"category": c, "name": n, "status": s, "note": note}
            for c, n, s, note in AlgorithmRegistry.catalogue(status)]


@app.get("/api/health")
def health():
    return {"ok": True, "jobs": len(store.all()),
            "workers": config.JOB_WORKERS, "version": app.version}


@app.on_event("startup")
def _startup():
    store.purge_expired()


# --------------------------------------------------------------------------- #
def _job(job_id: str):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job id")
    return job


def _rec(job_id: str, rank: int):
    job = _job(job_id)
    for rec in job.recommendations:
        if rec["rank"] == rank:
            return job, rec
    raise HTTPException(404, f"no recommendation ranked {rank}")
