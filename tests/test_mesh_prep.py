from pathlib import Path

import numpy as np

from fake_bevel_baker.mesh_prep import prepare_mesh, split_by_angle

FIXTURES = Path(__file__).parent / "fixtures"


def test_cube_splits_8_to_24_at_45deg():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=45.0)
    assert prep.split_positions.shape == (24, 3)
    assert prep.split_normals.shape == (24, 3)
    assert prep.split_faces.shape == (12, 3)
    assert prep.split_faces.min() >= 0


def test_cube_split_normals_are_face_axis_aligned_at_45deg():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=45.0)
    abs_n = np.abs(prep.split_normals)
    # exactly one coordinate should be ~1, others ~0
    near_one = np.isclose(abs_n.max(axis=1), 1.0, atol=1e-4)
    sum_of_abs = abs_n.sum(axis=1)
    assert near_one.all(), "each split normal should be axis-aligned"
    assert np.allclose(sum_of_abs, 1.0, atol=1e-4), "normals should be unit axis vectors"


def test_cube_fully_smoothed_at_high_threshold():
    # With a 120deg threshold, all 90deg edges are "soft": every face groups
    # with its neighbors at every shared vertex, collapsing to 8 smooth verts.
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=120.0)
    assert prep.split_positions.shape == (8, 3)
    # All 8 smooth normals must be the body-diagonal unit vectors (+-1/sqrt(3))
    expected = 1.0 / np.sqrt(3.0)
    assert np.allclose(np.abs(prep.split_normals), expected, atol=1e-4)


def test_cube_face_normals_are_unit():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    lens = np.linalg.norm(prep.face_normals, axis=1)
    assert np.allclose(lens, 1.0, atol=1e-5)


def test_cube_tangent_uvs_present_and_in_unit_square():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    assert prep.tangent_uvs.shape[1] == 2
    assert prep.tangent_uvs.min() >= -1e-5
    assert prep.tangent_uvs.max() <= 1.0 + 1e-5


def test_split_faces_reference_valid_indices():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    assert prep.split_faces.max() < prep.split_positions.shape[0]
    assert prep.tangent_faces.max() < prep.tangent_positions.shape[0]


def test_split_and_tangent_face_counts_match():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    assert prep.split_faces.shape == prep.tangent_faces.shape


def test_boundary_edges_are_treated_as_hard():
    # A single triangle: all three edges have only one incident face, so
    # every edge is hard -> vertices cannot merge with anything, 3 split verts.
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    sp, sn, sf = split_by_angle(positions, faces, angle_deg=30.0)
    assert sp.shape == (3, 3)
    # Single face, normal is +Z
    assert np.allclose(sn, np.array([[0, 0, 1]]), atol=1e-5)
    assert set(sf.flatten().tolist()) == {0, 1, 2}
