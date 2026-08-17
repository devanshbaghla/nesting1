# STL Nesting Service

Upload an STL, get back the top *N* ways to arrange **two copies** of that part
in the smallest bounding box, with a guaranteed minimum surface-to-surface
clearance. Each arrangement comes with an isometric preview, downloadable
top / bottom / front views, and the nested STL.

![contact sheet](docs/example_contact_sheet.png)

---

## Quick start

```bash
git clone <this repo> && cd nesting-app
./run.sh                       # creates .venv, installs, serves on :8000
```

Then open <http://localhost:8000>. Or manually:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Command line, no server:

```bash
python cli.py part.stl -n 10 -o results/
python cli.py --describe            # list every algorithm in the registry
```

Run the end-to-end test (uploads `data/sample.stl`, checks every endpoint):

```bash
python -m tests.test_smoke
```

---

## What you get per run

| Artefact | Where |
|---|---|
| `*_nest_01.stl` … `*_nest_10.stl` | one nested pair per recommendation |
| isometric preview | rendered eagerly, shown in the results grid |
| top / bottom / front PNGs | rendered on demand, **"2D views"** button → ZIP |
| `recommendations.json` / `.md` | metrics, transforms, audit, validation results |
| `all.zip` | everything at once |

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | upload an STL → `{job_id}` |
| `GET` | `/api/jobs/{id}` | status, progress, log, recommendations |
| `GET` | `/api/jobs/{id}/rec/{rank}/image/{view}` | PNG; `iso`, `top`, `bottom`, `front`, `back`, `left`, `right` |
| `GET` | `/api/jobs/{id}/rec/{rank}/stl` | download that arrangement |
| `GET` | `/api/jobs/{id}/rec/{rank}/views.zip` | top + bottom + front + combined sheet |
| `GET` | `/api/jobs/{id}/all.zip` | every STL, preview and the report |
| `GET` | `/api/jobs/{id}/report` | full JSON report |
| `GET` | `/api/algorithms` | the algorithm registry |
| `DELETE` | `/api/jobs/{id}` | delete a job and its files |

Interactive docs at `/docs`.

```bash
curl -F file=@part.stl -F clearance=5 -F top_n=10 -F profile=quick \
     http://localhost:8000/api/jobs
```

---

## Options

| Option | Values | Notes |
|---|---|---|
| `clearance` | float | minimum surface gap, **in the STL's own units** — STL carries no unit information |
| `top_n` | 1–20 | how many arrangements to return |
| `profile` | `quick` · `standard` · `full` | search breadth; see timings below |
| `objective` | `volume` · `footprint` · `balanced` · `height` | what "best" means |
| `refiner` | `none` · `descend` · `profile` | how hard to squeeze toward the exact clearance |

**Timings** (single core, 4k-triangle part): `quick` ≈ 1–3 min · `standard`
≈ 10 min · `full` adds a 744-orientation SO(3) sweep. Cost scales with
`top_n`, since each candidate needs its own sample set and KD-trees.

---

## How it works

```
upload → audit (gate) → self-tests (gate) → coarse orientation sweep
       → fine sweep → diversify → refine each → export → verify → render
```

1. **Voxelise** by z-scanline parity fill on a shared lattice — exact solid
   occupancy, 0.18% volume error at 0.5 mm.
2. **Dilate** copy A by the clearance with a Euclidean distance transform, so
   "keep 5 mm apart" becomes a binary overlap test.
3. **Search translations** by FFT correlation. One transform evaluates *every*
   lattice translation, so the translation optimum is global for each rotation.
4. **Sweep rotations**, then refine locally around the winner.
5. **Refine continuously** against exact point-to-triangle distance, recovering
   the slack the deliberately conservative lattice search left behind.
6. **Verify** by reloading the written file and re-measuring with a fresh seed.

Full algorithm documentation lives in `app/core/nesting3d.py`; the registry of
what was used, what is kept only as a cross-check, and what was measured and
rejected is in `app/core/nesting_factory.py`.

### Why several recommendations

Minimum volume and minimum footprint are **different arrangements**. On the
sample part the volume optimum interlocks two uprights (205,440 mm³, 1,540 mm²
footprint) while the footprint optimum simply stacks them end to end (980 mm²,
but 251 mm tall and the worst volume available). Which you want depends on
whether you are filling a carton or a print bed, and geometry cannot tell you
that. So the service returns a diverse, Pareto-annotated set: rows marked
Pareto are not beaten on both volume *and* footprint simultaneously.

---

## Things worth knowing

**Watertight meshes only.** Solid voxelisation by parity is undefined for an
open surface, so non-watertight uploads are rejected at validation rather than
silently producing a wrong answer. Repair in your CAD tool first.

**`refiner=none` leaves gaps loose.** It keeps the conservative lattice pose —
still valid, just further apart than requested (6–12 mm where you asked for 5).
Use `descend` or `profile` to squeeze to a true 5.00. `profile` is the only one
that reliably finds the feasibility cliff where an interlock engages;
coordinate descent can settle on the wrong side of it.

**Verification is not uniform across ranks.** Ranks 1–3 are re-measured with
the exact point-to-triangle metric; the rest get a faster sampled check. The
`verified.gap_method` field records which. Don't read a rank-8 gap as being
held to the same standard as rank 1.

**Two copies only.** Extending to *N* is not a small change: the FFT oracle
assumes one fixed part and one mover, so *N* needs either sequential placement
against a growing occupancy grid (fast, greedy, no longer globally optimal) or
branch-and-bound over placement order.

**Not provably globally optimal.** The rotation grid is discrete and copy A is
pinned at identity, so global rotations of the pair are not searched. The
honest claim is "best found by a well-covered search."

**Single worker by default.** The engine is CPU-bound on one core, so
concurrent jobs make each slower without improving throughput. Jobs queue.
Raise `NEST_JOB_WORKERS` only if you have cores to spare.

**Jobs are in-memory.** Restarting the server loses job records, though files
survive under `data/jobs/`. For production, back `JobStore` with Redis or a
database and move rendering to a task queue.

---

## Configuration

Environment variables, all optional:

```
NEST_DATA_DIR        job artefact directory      (default data/jobs)
NEST_MAX_UPLOAD_MB   upload size limit           (64)
NEST_MAX_FACES       triangle limit              (400000)
NEST_JOB_WORKERS     concurrent jobs             (1)
NEST_JOB_TTL_HOURS   artefact retention          (24)
NEST_DEFAULT_PROFILE quick | standard | full     (quick)
NEST_RENDER_DPI      PNG resolution              (110)
```

---

## Layout

```
app/
  main.py              FastAPI routes
  jobs.py              job store, worker pool, progress bridge
  renders.py           painter's-algorithm iso + orthographic renderer
  config.py            settings
  core/
    nesting3d.py       all geometry algorithms as classes
    nesting_factory.py registry + factory
    nest_base.py       recommendation driver
  templates/index.html
  static/{app.js,style.css}
tests/test_smoke.py    end-to-end: upload → poll → fetch every artefact
cli.py                 command-line entry point
```

Rendering uses a painter's algorithm rather than an interactive 3D backend:
servers have no GPU, and matplotlib's mplot3d does not depth-sort reliably
across two interpenetrating bodies — parts of the rear copy leak in front of
the near one, which on a nesting result reads as a broken interlock.

## Licence

MIT.
