#!/usr/bin/env python3
"""Make an STL watertight — repair it if possible, rebuild it if not.

    python solidify.py part.stl                     # -> part_solid.stl
    python solidify.py part.stl -o closed.stl
    python solidify.py part.stl --pitch 0.4         # finer voxel rebuild
    python solidify.py part.stl --check             # report only, write nothing
    python solidify.py *.stl -d solids/             # batch

Two mechanisms, in order of preference:

1. :class:`~app.core.mesh_repair.MeshRepair` — weld duplicate vertices, drop
   slivers and duplicate faces, agree the winding, fill small holes. This
   preserves the exact surface and fixes the overwhelming majority of "open"
   STLs, which are open only because the exporter never welded anything.

2. :class:`~app.core.mesh_repair.MeshSolidify` — rasterise the geometry to a
   voxel grid, flood-fill the interior, and emit the filled/empty boundary.
   Closed and 2-manifold by construction, so it works on input that cannot be
   repaired at all: an assembly of open shells, a zero-thickness patch, a
   scan. The surface is quantised to the pitch, so this is an approximation of
   the part and is reported as one.

Exit status is 0 when every file ended up watertight, 1 otherwise, so this can
gate a batch in a script.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

from app.core.mesh_repair import (MeshDenoise, MeshRepair, MeshRepairError,
                                  MeshSolidify)


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def describe(mesh: trimesh.Trimesh) -> str:
    return (f"{len(mesh.faces):,} faces, {mesh.body_count} "
            f"{'body' if mesh.body_count == 1 else 'bodies'}, "
            f"{MeshRepair.open_edge_count(mesh):,} open edges, "
            f"extents {np.round(mesh.extents, 2).tolist()}")


def solidify_file(src: Path, dst: Path | None, *, pitch: float | None,
                  denoise: bool, allow_solidify: bool, check: bool,
                  quiet: bool) -> bool:
    """Return True when ``src`` is (or was made) a watertight solid."""
    def log(msg=""):
        if not quiet:
            print(msg, flush=True)

    t0 = time.time()
    log(f"\n{src.name}")
    try:
        mesh = load_mesh(src)
    except Exception as exc:
        print(f"  cannot read: {exc}", file=sys.stderr)
        return False
    log(f"  in : {describe(mesh)}")

    if denoise:
        mesh, _ = MeshDenoise.strip_stray_shells(
            mesh, log=lambda m: log(f"  {m.strip()}"))

    state = MeshRepair.diagnose(mesh)
    if state["watertight"] and state["winding_consistent"] and state["volume"] > 0:
        log("  already a closed solid; nothing to do")
        if dst and not check:
            mesh.export(str(dst))
            log(f"  -> {dst}")
        return True

    if check:
        log(f"  NOT a closed solid: {state['open_edges']:,} open edges, "
            f"winding {'consistent' if state['winding_consistent'] else 'inconsistent'}"
            f", {state['bodies']:,} "
            f"{'body' if state['bodies'] == 1 else 'bodies'}")
        return False

    try:
        solid, report = MeshRepair.ensure_solid(
            mesh, allow_solidify=allow_solidify, solidify_pitch=pitch,
            log=lambda m: log(f"  {m.strip()}"))
    except MeshRepairError as exc:
        print(f"  FAILED: {exc}", file=sys.stderr)
        return False

    log(f"  out: {describe(solid)}, volume {solid.volume:,.1f}")
    if report.approximated:
        log(f"  NOTE surface is a rasterisation, accurate to "
            f"+/-{report.solidify.error_mm():.3f} (file units)")
    if dst:
        dst.parent.mkdir(parents=True, exist_ok=True)
        solid.export(str(dst))
        log(f"  -> {dst}  [{time.time() - t0:.1f}s]")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Make one or more STL files watertight.")
    p.add_argument("stl", nargs="+", help="input STL file(s)")
    p.add_argument("-o", "--out",
                   help="output path; only valid with a single input file")
    p.add_argument("-d", "--out-dir",
                   help="write <name>_solid.stl into this directory")
    p.add_argument("--pitch", type=float,
                   help="voxel pitch for the rebuild (default: from part size)")
    p.add_argument("--no-solidify", action="store_true",
                   help="topological repair only; fail rather than rasterise")
    p.add_argument("--no-denoise", action="store_true",
                   help="keep stray fragments instead of dropping them")
    p.add_argument("--check", action="store_true",
                   help="report watertightness and write nothing")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)

    inputs = [Path(s) for s in a.stl]
    if a.out and len(inputs) > 1:
        p.error("a single output path needs a single input file")

    ok = True
    for src in inputs:
        if a.check:
            dst = None
        elif a.out:
            dst = Path(a.out)
        elif a.out_dir:
            dst = Path(a.out_dir) / f"{src.stem}_solid.stl"
        else:
            dst = src.with_name(f"{src.stem}_solid.stl")
        ok &= solidify_file(src, dst, pitch=a.pitch, denoise=not a.no_denoise,
                            allow_solidify=not a.no_solidify, check=a.check,
                            quiet=a.quiet)
    if not a.quiet:
        print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
