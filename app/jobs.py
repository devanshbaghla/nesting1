"""
Background job execution for the nesting engine.

Nesting takes minutes, not milliseconds, so uploads return a job id
immediately and the browser polls. The executor is deliberately single-worker
by default: the engine is CPU-bound on one core (FFT correlation, then KD-tree
queries), so running jobs concurrently makes every one of them slower without
improving throughput. Queue them instead.
"""

from __future__ import annotations

import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import trimesh

from . import config
from .core.mesh_repair import (DenoiseReport, MeshDenoise, MeshRepair,
                               MeshRepairError)
from .core.nest_base import NestingRecommender
from .core.nesting_factory import NesterFactory

# stage headings emitted by the engine, used to turn its log into a progress bar
_STAGES = ["load", "audit", "self-tests", "coarse sweep", "fine sweep",
           "diversify", "refine + export", "report"]


@dataclass
class Job:
    id: str
    filename: str
    params: dict
    status: str = "queued"           # queued | running | done | failed
    stage: str = "queued"
    progress: float = 0.0
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished: str | None = None
    error: str | None = None
    log: list = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    baselines: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return config.DATA_DIR / self.id

    def public(self) -> dict:
        d = asdict(self)
        d["log"] = self.log[-40:]
        return d


class JobStore:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=config.JOB_WORKERS)

    def create(self, filename: str, params: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, params=params)
        job.dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def submit(self, job: Job):
        self._pool.submit(self._run, job)

    def purge_expired(self):
        cutoff = datetime.now(timezone.utc) - timedelta(hours=config.JOB_TTL_HOURS)
        with self._lock:
            stale = [j for j in self._jobs.values()
                     if datetime.fromisoformat(j.created) < cutoff
                     and j.status in ("done", "failed")]
            for j in stale:
                shutil.rmtree(j.dir, ignore_errors=True)
                self._jobs.pop(j.id, None)
        return len(stale)

    # -- worker ------------------------------------------------------------ #
    def _run(self, job: Job):
        job.status, job.stage = "running", "starting"
        t0 = time.time()
        try:
            cfg = NesterFactory.config(
                job.params["profile"],
                clearance=job.params["clearance"],
                objective=job.params["objective"],
                refiner=job.params.get("refiner"),
                distance_backend=job.params.get("distance_backend"),
                top_n=job.params["top_n"],
                verbose=False,
            )
            rec = _WebRecommender(cfg, job)
            results = rec.recommend(job.dir / job.filename, job.dir,
                                    job.params["top_n"])

            # No render pass: export_one already wrote each pair as a GLB
            # while its two copies were still in memory, so there is nothing
            # left to draw. This stage used to rasterise one isometric PNG per
            # recommendation and was the most expensive thing after refinement.
            job.stage = "collecting results"
            job.progress = 0.97
            payload = []
            for r in results:
                payload.append({
                    "rank": r.rank, "label": r.label, "source": r.source,
                    "extents": [round(v, 2) for v in r.extents],
                    "volume": round(r.volume, 1),
                    "footprint": round(r.footprint, 1),
                    "height": round(r.height, 2),
                    "gap": round(r.gap, 4),
                    "refined": r.refined, "pareto": r.pareto,
                    "transform": r.transform,
                    "stl": Path(r.stl).name,
                    "glb": Path(r.glb).name if r.glb else "",
                    "verified": r.verified,
                })
            job.recommendations = payload
            job.audit = rec.audit_data
            job.baselines = rec.baseline_data
            job.status, job.stage, job.progress = "done", "complete", 1.0
            job.log.append(f"finished in {time.time() - t0:.0f}s")
        except MeshRepairError as exc:
            # a bad input file, not a bug — the message is already written for
            # the user, and a traceback in the job log would only obscure it
            job.status = "failed"
            job.stage = "failed"
            job.error = str(exc)
            job.log.append("ERROR " + str(exc))
        except Exception as exc:
            job.status = "failed"
            job.stage = "failed"
            job.error = str(exc)
            job.log.append("ERROR " + str(exc))
            job.log.extend(traceback.format_exc().splitlines()[-6:])
        finally:
            job.finished = datetime.now(timezone.utc).isoformat()


class _WebRecommender(NestingRecommender):
    """Recommender that reports progress into a Job instead of stdout."""

    def __init__(self, cfg, job: Job):
        super().__init__(cfg)
        self.job = job
        self.audit_data: dict = {}
        self.baseline_data: dict = {}

    def _log(self, msg=""):
        msg = str(msg)
        self.job.log.append(msg)
        stripped = msg.strip()
        if stripped.startswith("[") and "/8]" in stripped:
            try:
                n = int(stripped[1:stripped.index("/")])
                self.job.progress = round(0.9 * n / 8, 3)
                self.job.stage = _STAGES[min(n - 1, len(_STAGES) - 1)]
            except (ValueError, IndexError):
                pass

    def audit(self, mesh):
        self.audit_data = super().audit(mesh)
        return self.audit_data

    def _write_report(self, out_dir, stl_path, mesh, recs, audit, tests, baselines):
        self.baseline_data = baselines
        return super()._write_report(out_dir, stl_path, mesh, recs, audit,
                                     tests, baselines)


def validate_upload(path: Path, repair: bool = True,
                    denoise: bool = True) -> dict:
    """Reject unusable uploads before a worker slot is spent on them.

    Disconnected debris is stripped and an open mesh is repaired here rather
    than at the start of the job, and the cleaned solid is written back over
    the uploaded copy. Two reasons: the user learns about it in the upload
    response instead of minutes later in a failed job, and the worker then
    loads a mesh that needs no work — one clean-up per upload, and the STL on
    disk matches the results derived from it.

    :raises ValueError: unusable file. ``MeshRepairError`` is a ``ValueError``,
        so an unrepairable mesh surfaces through the same 422 as the rest.
    """
    size_mb = path.stat().st_size / 1e6
    if size_mb > config.MAX_UPLOAD_MB:
        raise ValueError(f"file is {size_mb:.1f} MB, limit is {config.MAX_UPLOAD_MB} MB")
    try:
        mesh = trimesh.load(str(path), force="mesh")
    except Exception as exc:
        raise ValueError(f"could not parse as STL: {exc}") from exc
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        raise ValueError("no triangles found in the file")
    if config.MAX_FACES and len(mesh.faces) > config.MAX_FACES:
        raise ValueError(f"{len(mesh.faces):,} faces exceeds the "
                         f"{config.MAX_FACES:,} limit; decimate first")

    noise = DenoiseReport()
    if denoise:
        mesh, noise = MeshDenoise.strip_stray_shells(mesh)
    mesh, report = MeshRepair.ensure_solid(mesh, allow_repair=repair)
    if report.repaired or noise.changed:
        mesh.export(str(path))

    return {"faces": int(len(mesh.faces)),
            "extents": [round(float(v), 2) for v in mesh.extents],
            "volume": round(float(mesh.volume), 1),
            "fill_ratio": round(float(mesh.volume / np.prod(mesh.extents)), 4),
            "repaired": bool(report.repaired),
            "approximated": bool(report.approximated),
            "repair": report.to_dict(),
            "denoised": bool(noise.changed),
            "denoise": noise.to_dict()}


store = JobStore()
