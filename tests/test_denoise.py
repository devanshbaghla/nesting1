"""Stray-fragment removal: kill the debris, keep the part.

Debris is dangerous precisely because it is not an error — the mesh stays
watertight and every check passes. What it does is stretch the axis-aligned
box that every reported number is measured against, so the nesting quietly
optimises for mostly empty air. These tests pin both halves: the debris goes,
and geometry that merely looks small does not.
"""
import numpy as np
import trimesh

from app.core.mesh_repair import MeshDenoise
from app.core.nesting_factory import NesterFactory


def _part():
    return trimesh.creation.box(extents=(20.0, 30.0, 40.0))


def _with_speck(offset=(200.0, 0.0, 0.0), size=0.4):
    part = _part()
    speck = trimesh.creation.box(extents=(size, size, size))
    speck.apply_translation(offset)
    return trimesh.util.concatenate([part, speck]), part


# --------------------------------------------------------------------------- #
def test_single_body_is_untouched():
    m = _part()
    out, rep = MeshDenoise.strip_stray_shells(m)
    assert out is m and not rep.changed
    assert rep.summary() == "no stray fragments found"


def test_distant_speck_is_removed():
    dirty, clean = _with_speck()
    assert dirty.is_watertight, "the speck does not break any existing check"
    out, rep = MeshDenoise.strip_stray_shells(dirty)
    assert rep.changed and len(rep.dropped) == 1
    assert out.body_count == 1
    assert np.allclose(out.extents, clean.extents), out.extents


def test_the_bounding_box_is_what_gets_fixed():
    """The whole point: one 0.4 mm speck was inflating the box 11-fold."""
    dirty, clean = _with_speck()
    before = float(np.prod(dirty.extents))
    out, rep = MeshDenoise.strip_stray_shells(dirty)
    after = float(np.prod(out.extents))
    assert before / after > 10, f"expected a big shrink, got {before/after:.2f}x"
    assert np.isclose(after, float(np.prod(clean.extents)))
    assert rep.bbox_shrink() > 0.9
    assert "bounding box down" in rep.summary()


def test_a_real_two_piece_assembly_is_kept():
    """A genuine second body must survive — this is the false-positive guard."""
    a = _part()
    b = _part()
    b.apply_translation([60.0, 0.0, 0.0])
    both = trimesh.util.concatenate([a, b])
    out, rep = MeshDenoise.strip_stray_shells(both)
    assert not rep.changed, rep.summary()
    assert out.body_count == 2


def test_a_small_but_not_tiny_body_is_kept():
    """10% of the part's diagonal is over the 5% threshold, so it stays."""
    part = _part()
    diag = float(np.linalg.norm(part.extents))
    side = 0.10 * diag / np.sqrt(3)
    small = trimesh.creation.box(extents=(side, side, side))
    small.apply_translation([80.0, 0.0, 0.0])
    out, rep = MeshDenoise.strip_stray_shells(
        trimesh.util.concatenate([part, small]))
    assert not rep.changed, f"dropped a legitimate body: {rep.summary()}"


def test_threshold_is_tunable():
    part = _part()
    diag = float(np.linalg.norm(part.extents))
    side = 0.10 * diag / np.sqrt(3)
    small = trimesh.creation.box(extents=(side, side, side))
    small.apply_translation([80.0, 0.0, 0.0])
    dirty = trimesh.util.concatenate([part, small])
    assert not MeshDenoise.strip_stray_shells(dirty, ratio=0.05)[1].changed
    assert MeshDenoise.strip_stray_shells(dirty, ratio=0.30)[1].changed


def test_several_specks_all_go():
    part = _part()
    bits = [part]
    for i in range(5):
        s = trimesh.creation.box(extents=(0.3, 0.3, 0.3))
        s.apply_translation([50.0 + 30 * i, 20.0, -40.0])
        bits.append(s)
    out, rep = MeshDenoise.strip_stray_shells(trimesh.util.concatenate(bits))
    assert len(rep.dropped) == 5
    assert out.body_count == 1
    assert np.allclose(out.extents, part.extents)


def test_result_stays_watertight():
    dirty, _ = _with_speck()
    out, _ = MeshDenoise.strip_stray_shells(dirty)
    assert out.is_watertight and out.volume > 0


def test_report_is_serialisable():
    import json
    dirty, _ = _with_speck()
    _, rep = MeshDenoise.strip_stray_shells(dirty)
    d = json.loads(json.dumps(rep.to_dict()))
    assert d["changed"] and d["summary"] and d["bbox_shrink"] > 0.9
    assert d["faces_before"] > d["faces_after"]


def test_config_exposes_the_switches():
    cfg = NesterFactory.config("quick")
    assert cfg.denoise is True and cfg.denoise_ratio == 0.05
    assert NesterFactory.config("quick", denoise=False).denoise is False
    assert NesterFactory.config("quick", denoise_ratio=0.2).denoise_ratio == 0.2


def test_face_limit_is_unlimited_by_default():
    from app import config
    assert config.MAX_FACES == 0, "0 means no ceiling"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all denoise tests passed")
