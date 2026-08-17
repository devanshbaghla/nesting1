"""
FastAPI application: upload an STL, get the top N nested arrangements back
with isometric previews and downloadable orthographic views.

    uvicorn app.main:app --reload

Routes
------
GET  /                                       upload page + results UI
POST /api/jobs                               upload an STL, returns a job id
GET  /api/jobs                               list jobs
GET  /api/jobs/{id}                          status, progress, recommendations
DELETE /api/jobs/{id}                        remove a job and its artefacts
GET  /api/jobs/{id}/rec/{rank}/image/{view}  PNG, view in iso|top|bottom|front|...
GET  /api/jobs/{id}/rec/{rank}/stl           download the nested STL
GET  /api/jobs/{id}/rec/{rank}/views.zip     top + bottom + front PNGs, zipped
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

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               StreamingResponse)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from . import config
from .core.nesting3d import HAVE_FCL
from .core.nesting_factory import PROFILES, AlgorithmRegistry
from .jobs import store, validate_upload
from .renders import DOWNLOADABLE, VIEWS, render, render_sheet

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

    params = {"clearance": clearance, "top_n": top_n, "profile": profile,
              "objective": objective, "refiner": refiner,
              "distance_backend": distance_backend}
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
    if stats.get("repaired"):
        # the nested output is derived from the repaired solid, not the file
        # that was uploaded — say so where the user is already looking
        job.log.append(f"input mesh was not a closed solid; "
                       f"{stats['repair']['summary']}")
    store.submit(job)
    return {"job_id": job.id, "status": job.status, "mesh": stats}


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
@app.get("/api/jobs/{job_id}/rec/{rank}/image/{view}")
def rec_image(job_id: str, rank: int, view: str):
    """Isometric views are rendered up front; the rest on first request.

    Rendering all seven views for every recommendation would add roughly a
    minute to a ten-candidate job for images most people never open, so only
    the isometric is eager. Everything else is cached to disk on first hit.
    """
    job, rec = _rec(job_id, rank)
    if view not in VIEWS:
        raise HTTPException(400, f"view must be one of {list(VIEWS)}")
    png = job.dir / f"rec_{rank:02d}_{view}.png"
    if not png.exists():
        stl = job.dir / rec["stl"]
        if not stl.exists():
            raise HTTPException(404, "STL missing for this recommendation")
        render(stl, view, png, dpi=config.RENDER_DPI,
               label=f"#{rank}  {view.upper()}")
    return FileResponse(png, media_type="image/png",
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


@app.get("/api/jobs/{job_id}/rec/{rank}/views.zip")
def rec_views_zip(job_id: str, rank: int, sheet: bool = True):
    """Top, bottom and front as PNGs, plus a combined sheet, zipped."""
    job, rec = _rec(job_id, rank)
    stl = job.dir / rec["stl"]
    stem = Path(job.filename).stem
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for view in DOWNLOADABLE:
            png = job.dir / f"rec_{rank:02d}_{view}.png"
            if not png.exists():
                render(stl, view, png, dpi=config.RENDER_DPI,
                       label=f"#{rank}  {view.upper()}")
            zf.write(png, f"{stem}_nest{rank:02d}_{view}.png")
        if sheet:
            sh = job.dir / f"rec_{rank:02d}_sheet.png"
            if not sh.exists():
                render_sheet(stl, sh, dpi=config.RENDER_DPI,
                             label=f"#{rank}  {rec['extents'][0]}x"
                                   f"{rec['extents'][1]}x{rec['extents'][2]} mm  "
                                   f"gap {rec['gap']} mm")
            zf.write(sh, f"{stem}_nest{rank:02d}_all_views.png")
        zf.writestr(f"{stem}_nest{rank:02d}_metrics.json",
                    json.dumps(rec, indent=2))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{stem}_nest{rank:02d}_views.zip"'})


@app.get("/api/jobs/{job_id}/all.zip")
def all_zip(job_id: str):
    """Every recommendation: STLs, isometric previews and the report."""
    job = _job(job_id)
    if job.status != "done":
        raise HTTPException(409, "job is not finished")
    stem = Path(job.filename).stem
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rec in job.recommendations:
            for key in ("stl", "iso"):
                p = job.dir / rec[key]
                if p.exists():
                    zf.write(p, p.name)
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
