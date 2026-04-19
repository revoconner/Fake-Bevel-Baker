import numpy as np

from fake_bevel_baker.bvh import BVH, LOCAL_MAX_HITS


def _make_stacked_planes(z_values):
    """Each plane = one triangle covering (-1..2) in x/y at given z."""
    verts = []
    tris = []
    for i, z in enumerate(z_values):
        base = len(verts)
        verts.extend([[-1.0, -1.0, z], [2.0, -1.0, z], [-1.0, 2.0, z]])
        tris.append([base, base + 1, base + 2])
    return np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.uint32)


def test_two_plane_multi_hit():
    v, t = _make_stacked_planes([0.0, 0.5])
    bvh = BVH(v, t)
    hits = bvh.multi_hit_single([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], tmin=0.0, tmax=10.0)
    assert len(hits) == 2
    ts = [h["tfar"] for h in hits]
    assert ts == sorted(ts), f"hits not sorted by distance: {ts}"
    # First plane hit at t=1.0, second near t=1.5
    assert abs(ts[0] - 1.0) < 1e-3
    assert abs(ts[1] - 1.5) < 1e-3
    assert hits[0]["primID"] == 0
    assert hits[1]["primID"] == 1


def test_caps_at_max_hits():
    v, t = _make_stacked_planes([0.0, 0.5, 1.0, 1.5, 2.0])
    bvh = BVH(v, t)
    # Only 4 hits returned even though 5 planes exist
    hits = bvh.multi_hit_single([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], tmin=0.0, tmax=10.0)
    assert len(hits) == LOCAL_MAX_HITS == 4


def test_tmax_clips_hits():
    v, t = _make_stacked_planes([0.0, 0.5, 1.0])
    bvh = BVH(v, t)
    # tmax lets only the first two planes through (at cum distances ~1.0 and ~1.5)
    hits = bvh.multi_hit_single([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], tmin=0.0, tmax=1.6)
    assert len(hits) == 2


def test_ray_misses_returns_empty():
    v, t = _make_stacked_planes([0.0])
    bvh = BVH(v, t)
    # Ray points away from the plane
    hits = bvh.multi_hit_single([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], tmin=0.0, tmax=10.0)
    assert hits == []


def test_batch_multi_hit_shapes():
    v, t = _make_stacked_planes([0.0, 0.5, 1.0])
    bvh = BVH(v, t)
    origins = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 5.0], [0.5, 0.5, -1.0]], dtype=np.float32)
    dirs = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    r = bvh.multi_hit(origins, dirs, tmin=0.0, tmax=10.0, max_hits=4)
    assert r.tfar.shape == (3, 4)
    assert r.primID.shape == (3, 4)
    assert r.Ng.shape == (3, 4, 3)
    assert r.hit_count.tolist() == [3, 0, 3]


def test_hits_sorted_and_distinct():
    v, t = _make_stacked_planes([0.0, 0.2, 0.4, 0.6])
    bvh = BVH(v, t)
    hits = bvh.multi_hit_single([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], tmin=0.0, tmax=10.0)
    assert len(hits) == 4
    ts = [h["tfar"] for h in hits]
    # Strictly increasing
    assert all(ts[i] < ts[i + 1] for i in range(3))
    # Primitives hit in order
    assert [h["primID"] for h in hits] == [0, 1, 2, 3]


def test_tmin_advances_origin():
    v, t = _make_stacked_planes([0.0, 1.0])
    bvh = BVH(v, t)
    # Origin z=-2, first plane at distance 2.0, second at 3.0.
    # tmin=2.5 skips the first plane; only the second should hit (cumulative t=3.0).
    hits = bvh.multi_hit_single([0.0, 0.0, -2.0], [0.0, 0.0, 1.0], tmin=2.5, tmax=10.0)
    assert len(hits) == 1
    assert abs(hits[0]["tfar"] - 3.0) < 1e-3


def test_cube_mesh_multi_hit_through():
    from fake_bevel_baker.mesh_prep import prepare_mesh
    from pathlib import Path
    prep = prepare_mesh(Path(__file__).parent / "fixtures" / "cube.obj", angle_deg=30.0)
    bvh = BVH(prep.split_positions, prep.split_faces)
    # Cube fixture spans roughly x,z in [-13.15, 13.15], y in [0, 26.3]
    # Shoot through the cube along +X
    hits = bvh.multi_hit_single([-50.0, 13.0, 0.0], [1.0, 0.0, 0.0], tmin=0.0, tmax=200.0)
    # Entry + exit face = 2 hits
    assert len(hits) >= 2, f"expected at least 2 hits through cube, got {len(hits)}"
