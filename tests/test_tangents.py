from pathlib import Path

import numpy as np

from fake_bevel_baker.mesh_prep import prepare_mesh
from fake_bevel_baker.tangents import bitangent_from_tangent, compute_tangents

FIXTURES = Path(__file__).parent / "fixtures"


def _per_corner_normals(normals, faces):
    return normals[faces]  # (F, 3, 3)


def test_tangent_shape_and_finite():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    T = compute_tangents(
        prep.tangent_positions,
        prep.tangent_normals,
        prep.tangent_uvs,
        prep.tangent_faces,
    )
    assert T.shape == (prep.tangent_faces.shape[0], 3, 4)
    assert np.isfinite(T).all()


def test_tangents_are_unit_length():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    T = compute_tangents(
        prep.tangent_positions, prep.tangent_normals, prep.tangent_uvs, prep.tangent_faces,
    )
    lens = np.linalg.norm(T[..., :3], axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-4), f"max diff {np.abs(lens - 1.0).max()}"


def test_tangent_orthogonal_to_normal():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    T = compute_tangents(
        prep.tangent_positions, prep.tangent_normals, prep.tangent_uvs, prep.tangent_faces,
    )
    N = _per_corner_normals(prep.tangent_normals, prep.tangent_faces)  # (F, 3, 3)
    d = np.einsum("fkc,fkc->fk", T[..., :3], N)
    assert np.allclose(d, 0.0, atol=1e-4), f"max dot {np.abs(d).max()}"


def test_handedness_sign_is_pm_one():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    T = compute_tangents(
        prep.tangent_positions, prep.tangent_normals, prep.tangent_uvs, prep.tangent_faces,
    )
    s = T[..., 3]
    assert np.all((np.abs(s - 1.0) < 1e-4) | (np.abs(s + 1.0) < 1e-4)), np.unique(s)


def test_bitangent_right_handed_with_normal_and_tangent():
    prep = prepare_mesh(FIXTURES / "cube.obj", angle_deg=30.0)
    T = compute_tangents(
        prep.tangent_positions, prep.tangent_normals, prep.tangent_uvs, prep.tangent_faces,
    )
    N = _per_corner_normals(prep.tangent_normals, prep.tangent_faces)
    B = bitangent_from_tangent(N, T)
    # B should be unit length too
    assert np.allclose(np.linalg.norm(B, axis=-1), 1.0, atol=1e-4)
    # B orthogonal to N
    assert np.allclose(np.einsum("fkc,fkc->fk", B, N), 0.0, atol=1e-4)


def test_sphere_tangents_stable():
    prep = prepare_mesh(FIXTURES / "sphere.obj", angle_deg=180.0)
    T = compute_tangents(
        prep.tangent_positions, prep.tangent_normals, prep.tangent_uvs, prep.tangent_faces,
    )
    lens = np.linalg.norm(T[..., :3], axis=-1)
    assert np.allclose(lens, 1.0, atol=1e-4)
    assert np.isfinite(T).all()
