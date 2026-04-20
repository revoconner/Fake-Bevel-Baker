"""
per-texel bevel bake.

For each valid texel we shoot
`num_samples` probe rays on a BSSRDF-weighted disk around the shading
point, gather multi-hit normals through Embree, MIS-combine across the
3-axis disk frame, and average. World-space result is then rotated into
MikkTSpace tangent space via the tangent_mesh face corners.

The inner loop (ray prep + hit accumulation) is numba-jitted, parallel
across texels. Embree remains the one step on the Python side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numba
import numpy as np
from numba import njit, prange

from .bvh import BVH, LOCAL_MAX_HITS
from .mesh_prep import PreparedMesh
from .sampling import (
    make_orthonormals_batch,
    per_texel_offsets,
    sobol_sequence_2d,
)
from .tangents import compute_tangents
from .uv_raster import RasterResult


_PI = math.pi


@dataclass
class BakeResult:
    world_normal: np.ndarray   # (H, W, 3) float32, world-space, unit
    tangent_normal: np.ndarray # (H, W, 3) float32, MikkT tangent-space, unit
    valid: np.ndarray          # (H, W) bool


@njit(cache=True, parallel=True, fastmath=False)
def _prepare_sample(
    P, frame, offsets,
    h1, h2, radius,
    ray_origins, ray_dirs, ray_tmax,
    disk_N, disk_T, disk_B, pick_pdf, disk_pdf_self,
):
    N = P.shape[0]
    Rm5 = radius * radius * radius * radius * radius
    two_pi = 2.0 * _PI
    for i in prange(N):
        u1 = (h1 + offsets[i, 0]) % 1.0
        u2 = (h2 + offsets[i, 1]) % 1.0

        if u1 < 0.5:
            dN_idx = 0; dT_idx = 1; dB_idx = 2
            pick_pdf[i, 0] = 0.5; pick_pdf[i, 1] = 0.25; pick_pdf[i, 2] = 0.25
            rand_x = u1 * 2.0
        elif u1 < 0.75:
            dN_idx = 1; dT_idx = 0; dB_idx = 2
            pick_pdf[i, 0] = 0.25; pick_pdf[i, 1] = 0.5; pick_pdf[i, 2] = 0.25
            rand_x = (u1 - 0.5) * 4.0
        else:
            dN_idx = 2; dT_idx = 1; dB_idx = 0
            pick_pdf[i, 0] = 0.25; pick_pdf[i, 1] = 0.25; pick_pdf[i, 2] = 0.5
            rand_x = (u1 - 0.75) * 4.0

        for k in range(3):
            disk_N[i, k] = frame[i, dN_idx, k]
            disk_T[i, k] = frame[i, dT_idx, k]
            disk_B[i, k] = frame[i, dB_idx, k]

        # Quintic root-find, Newton-Raphson (Cycles svm_bevel_cubic_quintic_root_find)
        x = 0.25
        for _ in range(10):
            x2 = x * x
            x3 = x2 * x
            nx = 1.0 - x
            f = 10.0 * x2 - 20.0 * x3 + 15.0 * x2 * x2 - 4.0 * x2 * x3 - u2
            fp = 20.0 * (x * nx) * (nx * nx)
            if abs(fp) < 1e-20 or abs(f) < 1e-6:
                break
            xn = x - f / fp
            if xn < 0.0:
                xn = 0.0
            elif xn > 1.0:
                xn = 1.0
            x = xn

        disk_r = x * radius
        dh2 = radius * radius - disk_r * disk_r
        disk_height = math.sqrt(dh2) if dh2 > 0.0 else 0.0

        phi = two_pi * rand_x
        cp = math.cos(phi)
        sp = math.sin(phi)

        for k in range(3):
            ray_origins[i, k] = (
                P[i, k]
                + disk_N[i, k] * disk_height
                + disk_T[i, k] * (disk_r * cp)
                + disk_B[i, k] * (disk_r * sp)
            )
            ray_dirs[i, k] = -disk_N[i, k]

        ray_tmax[i] = 2.0 * disk_height

        if disk_r >= radius:
            disk_pdf_self[i] = 0.0
        else:
            fr = radius - disk_r
            disk_pdf_self[i] = (10.0 * fr * fr * fr) / (Rm5 * _PI)


@njit(cache=True, parallel=True, fastmath=False)
def _accumulate_hits(
    P, disk_N, disk_T, disk_B, pick_pdf, disk_pdf_self,
    hit_pid, hit_u, hit_v, hit_Ng,
    split_pos, split_nrm, split_faces,
    radius, max_hits,
    sum_N,
):
    N = P.shape[0]
    Rm5 = radius * radius * radius * radius * radius
    for i in prange(N):
        dsp = disk_pdf_self[i]
        if dsp < 1e-20:
            continue
        Px = P[i, 0]; Py = P[i, 1]; Pz = P[i, 2]
        pdf_n_pk = pick_pdf[i, 0]
        pdf_t_pk = pick_pdf[i, 1]
        pdf_b_pk = pick_pdf[i, 2]
        dN0 = disk_N[i, 0]; dN1 = disk_N[i, 1]; dN2 = disk_N[i, 2]
        dT0 = disk_T[i, 0]; dT1 = disk_T[i, 1]; dT2 = disk_T[i, 2]
        dB0 = disk_B[i, 0]; dB1 = disk_B[i, 1]; dB2 = disk_B[i, 2]

        acc0 = 0.0; acc1 = 0.0; acc2 = 0.0

        for h in range(max_hits):
            pid = hit_pid[i, h]
            if pid < 0:
                continue

            gx = hit_Ng[i, h, 0]; gy = hit_Ng[i, h, 1]; gz = hit_Ng[i, h, 2]
            gl = math.sqrt(gx * gx + gy * gy + gz * gz)
            if gl < 1e-20:
                continue
            gx /= gl; gy /= gl; gz /= gl

            us = hit_u[i, h]
            vs = hit_v[i, h]
            w0 = 1.0 - us - vs
            w1 = us
            w2 = vs
            v0 = split_faces[pid, 0]
            v1 = split_faces[pid, 1]
            v2 = split_faces[pid, 2]

            nx = split_nrm[v0, 0] * w0 + split_nrm[v1, 0] * w1 + split_nrm[v2, 0] * w2
            ny = split_nrm[v0, 1] * w0 + split_nrm[v1, 1] * w1 + split_nrm[v2, 1] * w2
            nz = split_nrm[v0, 2] * w0 + split_nrm[v1, 2] * w1 + split_nrm[v2, 2] * w2
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nl < 1e-20:
                continue
            nx /= nl; ny /= nl; nz /= nl

            hpx = split_pos[v0, 0] * w0 + split_pos[v1, 0] * w1 + split_pos[v2, 0] * w2
            hpy = split_pos[v0, 1] * w0 + split_pos[v1, 1] * w1 + split_pos[v2, 1] * w2
            hpz = split_pos[v0, 2] * w0 + split_pos[v1, 2] * w1 + split_pos[v2, 2] * w2

            dx = hpx - Px
            dy = hpy - Py
            dz = hpz - Pz
            rr = math.sqrt(dx * dx + dy * dy + dz * dz)
            if rr >= radius:
                continue
            fr = radius - rr
            pdf_real = (10.0 * fr * fr * fr) / (Rm5 * _PI)

            pdf_N = pdf_n_pk * abs(dN0 * gx + dN1 * gy + dN2 * gz)
            pdf_T = pdf_t_pk * abs(dT0 * gx + dT1 * gy + dT2 * gz)
            pdf_B = pdf_b_pk * abs(dB0 * gx + dB1 * gy + dB2 * gz)
            denom = pdf_N * pdf_N + pdf_T * pdf_T + pdf_B * pdf_B
            if denom < 1e-20:
                continue
            w = (pdf_N / denom) * (pdf_real / dsp)

            acc0 += w * nx
            acc1 += w * ny
            acc2 += w * nz

        sum_N[i, 0] += acc0
        sum_N[i, 1] += acc1
        sum_N[i, 2] += acc2


def bake(
    prep: PreparedMesh,
    raster: RasterResult,
    bvh: BVH,
    radius: float,
    num_samples: int,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> BakeResult:
    """Run the bevel bake."""
    if radius <= 0.0 or num_samples < 1:
        raise ValueError("radius must be > 0 and num_samples >= 1")

    H, W = raster.valid.shape
    world_out = np.zeros((H, W, 3), dtype=np.float32)
    tan_out = np.zeros((H, W, 3), dtype=np.float32)

    tangents_corner = compute_tangents(
        prep.tangent_positions, prep.tangent_normals,
        prep.tangent_uvs, prep.tangent_faces,
    )

    ys, xs = np.where(raster.valid)
    tri_ids = raster.tri_idx[ys, xs].astype(np.int64)
    barys = raster.bary[ys, xs].astype(np.float32)
    N = int(ys.shape[0])
    if N == 0:
        return BakeResult(world_out, tan_out, raster.valid)

    sp = np.ascontiguousarray(prep.split_positions, dtype=np.float32)
    sn = np.ascontiguousarray(prep.split_normals, dtype=np.float32)
    sf = np.ascontiguousarray(prep.split_faces, dtype=np.int32)
    fn = prep.face_normals

    tri_sp = sp[sf[tri_ids]]
    P = np.ascontiguousarray((tri_sp * barys[..., None]).sum(axis=1), dtype=np.float32)
    Ng = np.ascontiguousarray(fn[tri_ids], dtype=np.float32)

    # Tangent frame at texel (for output transform later)
    tf = prep.tangent_faces
    T_corner = tangents_corner[tri_ids]
    N_corner = prep.tangent_normals[tf[tri_ids]]
    T_interp = (T_corner[..., :3] * barys[..., None]).sum(axis=1)
    N_tan_interp = (N_corner * barys[..., None]).sum(axis=1)
    N_tan_interp /= np.maximum(np.linalg.norm(N_tan_interp, axis=-1, keepdims=True), 1e-20)
    T_interp -= (T_interp * N_tan_interp).sum(axis=-1, keepdims=True) * N_tan_interp
    T_interp /= np.maximum(np.linalg.norm(T_interp, axis=-1, keepdims=True), 1e-20)
    T_sign = T_corner[:, 0, 3]
    B_interp = T_sign[:, None] * np.cross(N_tan_interp, T_interp)

    # Base Ng frame
    T_base, B_base = make_orthonormals_batch(Ng)
    frame = np.ascontiguousarray(
        np.stack([Ng, T_base, B_base], axis=1), dtype=np.float32
    )

    # Sampler state
    offsets = per_texel_offsets(N, seed=seed)
    qmc = sobol_sequence_2d(num_samples, seed=seed)

    # Scratch buffers reused across samples
    ray_origins = np.empty((N, 3), dtype=np.float32)
    ray_dirs = np.empty((N, 3), dtype=np.float32)
    ray_tmax = np.empty(N, dtype=np.float32)
    ray_tmin = np.zeros(N, dtype=np.float32)
    disk_N = np.empty((N, 3), dtype=np.float32)
    disk_T = np.empty((N, 3), dtype=np.float32)
    disk_B = np.empty((N, 3), dtype=np.float32)
    pick_pdf = np.empty((N, 3), dtype=np.float32)
    disk_pdf_self = np.empty(N, dtype=np.float32)

    sum_N = np.zeros((N, 3), dtype=np.float64)

    for s in range(num_samples):
        if progress is not None:
            progress(s, num_samples)
        h1 = float(qmc[s, 0])
        h2 = float(qmc[s, 1])

        _prepare_sample(
            P, frame, offsets, h1, h2, float(radius),
            ray_origins, ray_dirs, ray_tmax,
            disk_N, disk_T, disk_B, pick_pdf, disk_pdf_self,
        )

        hits = bvh.multi_hit(
            ray_origins, ray_dirs,
            tmin=ray_tmin, tmax=ray_tmax,
            max_hits=LOCAL_MAX_HITS,
        )

        _accumulate_hits(
            P, disk_N, disk_T, disk_B, pick_pdf, disk_pdf_self,
            hits.primID, hits.u, hits.v, hits.Ng,
            sp, sn, sf,
            float(radius), int(LOCAL_MAX_HITS),
            sum_N,
        )

    if progress is not None:
        progress(num_samples, num_samples)

    lens = np.linalg.norm(sum_N, axis=-1, keepdims=True)
    nonzero = lens.squeeze(-1) > 1e-20
    N_world = np.where(
        nonzero[:, None],
        sum_N / np.maximum(lens, 1e-20),
        Ng,
    ).astype(np.float32)

    t_dot = (N_world * T_interp).sum(axis=-1)
    b_dot = (N_world * B_interp).sum(axis=-1)
    n_dot = (N_world * N_tan_interp).sum(axis=-1)
    N_tan = np.stack([t_dot, b_dot, n_dot], axis=-1)
    N_tan /= np.maximum(np.linalg.norm(N_tan, axis=-1, keepdims=True), 1e-20)

    world_out[ys, xs] = N_world
    tan_out[ys, xs] = N_tan

    return BakeResult(world_normal=world_out, tangent_normal=tan_out, valid=raster.valid)


def encode_normal_to_uint16(img: np.ndarray) -> np.ndarray:
    """Map unit-vector image [-1, 1] -> uint16 [0, 65535] with 0.5-offset encoding."""
    v = np.clip(img * 0.5 + 0.5, 0.0, 1.0)
    return (v * 65535.0 + 0.5).astype(np.uint16)
