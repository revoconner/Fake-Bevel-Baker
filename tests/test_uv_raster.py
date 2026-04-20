import numpy as np

from fake_bevel_baker.uv_raster import rasterize_uvs


def test_half_texture_lower_left():
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 256, 256)

    total = res.valid.size
    covered = int(res.valid.sum())
    # u + v < 1 triangle covers ~half minus the diagonal strip.
    # Allow some slack for edge-pixel inclusion.
    assert 0.48 * total <= covered <= 0.52 * total, f"{covered}/{total}"

    # Top-right corner must be empty
    assert not res.valid[255, 255]
    # Bottom-left corner must be covered
    assert res.valid[0, 0]
    # Bary in corner should be ~ (1, 0, 0) for vertex 0
    assert np.argmax(res.bary[0, 0]) == 0


def test_full_quad_two_triangles_covers_whole_image():
    uvs = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 64, 64)
    assert res.valid.all(), "full quad should cover every texel"
    # tri_idx must be either 0 or 1 everywhere
    assert ((res.tri_idx == 0) | (res.tri_idx == 1)).all()


def test_barycentrics_sum_to_one():
    uvs = np.array(
        [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 128, 128)
    s = res.bary[res.valid].sum(axis=-1)
    assert np.allclose(s, 1.0, atol=1e-4)


def test_empty_pixels_are_minus_one():
    uvs = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 64, 64)
    # Triangle is in the lower-left 20% region; top-right pixel must be -1
    assert res.tri_idx[63, 63] == -1
    # And the valid mask / tri_idx == -1 should agree
    assert (res.valid == (res.tri_idx >= 0)).all()


def test_uv_outside_image_is_clipped():
    # Triangle with one vertex outside [0,1] shouldn't crash
    uvs = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 32, 32)
    # This triangle covers the full image (u+v <= 2 clipped to image)
    assert res.valid.all()


def test_conservative_covers_strictly_more_than_center_only():
    # Thin triangle whose edges pass close to texel centers: conservative
    # raster must include texels that center-only would miss.
    uvs = np.array(
        [[0.10, 0.10], [0.90, 0.10], [0.50, 0.11]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res_c = rasterize_uvs(uvs, faces, 256, 256, conservative=True)
    res_n = rasterize_uvs(uvs, faces, 256, 256, conservative=False)
    assert res_c.valid.sum() > res_n.valid.sum()
    # Conservative must be a superset of center-only.
    assert np.all(res_c.valid[res_n.valid])


def test_conservative_island_reaches_uv_boundary():
    # A triangle whose vertex sits at u=0.5 exactly. Conservative raster
    # should include texels on both sides of the half-line, so the island
    # edge is not inset from the true UV boundary.
    uvs = np.array(
        [[0.00, 0.00], [0.50, 0.00], [0.50, 1.00]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    res = rasterize_uvs(uvs, faces, 128, 128, conservative=True)
    # Center column x=63 has center at u=(63.5)/128 ~ 0.496 -- inside.
    # Next column x=64 center at u ~ 0.504 -- conservative should still
    # include it because its left edge is at u=0.5 exactly (overlap).
    assert res.valid[10, 64], "conservative should reach the 0.5 UV boundary"


def test_cube_fixture_has_six_islands_coverage():
    from pathlib import Path
    from fake_bevel_baker.mesh_prep import prepare_mesh

    prep = prepare_mesh(
        Path(__file__).parent / "fixtures" / "cube.obj", angle_deg=30.0
    )
    res = rasterize_uvs(prep.tangent_uvs, prep.tangent_faces, 256, 256)
    covered = int(res.valid.sum())
    # Cube UV cross unwrap covers 6/14 of the UV square = ~43%
    assert 0.35 * res.valid.size <= covered <= 0.5 * res.valid.size, covered

    # Every covered texel should have a valid triangle index
    assert (res.tri_idx[res.valid] >= 0).all()
    assert (res.tri_idx[res.valid] < prep.tangent_faces.shape[0]).all()
