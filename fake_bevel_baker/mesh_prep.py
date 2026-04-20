"""
mesh ingestion + split-by-angle.

Loads a mesh, preserves per-corner (position / uv / normal) data so UV
seams and hard-edge normal seams are not silently merged, then produces:

- split_mesh: positions re-split by dihedral threshold, with angle-weighted
  smooth normals per smoothing group. Used for Embree sampling.
- tangent_mesh: what the engine sees. Pre-split vertex layout from the
  source file (unique per-corner triples). Used later for UV raster +
  MikkTSpace tangents.

Face arrays are index-parallel between the two meshes: tangent_faces[i]
and split_faces[i] refer to the same physical triangle, so the
split-to-tangent triangle map is identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass
class PreparedMesh:
    split_positions: np.ndarray    # (Ns, 3) float32
    split_normals: np.ndarray      # (Ns, 3) float32
    split_faces: np.ndarray        # (F, 3) int32

    tangent_positions: np.ndarray  # (Nt, 3) float32
    tangent_uvs: np.ndarray        # (Nt, 2) float32
    tangent_normals: np.ndarray    # (Nt, 3) float32
    tangent_faces: np.ndarray      # (F, 3) int32

    face_normals: np.ndarray       # (F, 3) float32


def _parse_obj_corners(path: Path):
    """Minimal OBJ parser that keeps per-corner v/vt/vn triples intact.

    Returns: positions (V,3), uvs (T,2)|None, normals (N,3)|None,
    face_corners (F,3,3) with columns [v, vt, vn], 0-based; -1 for absent.
    Only triangles are supported (fixtures are pre-triangulated).
    """
    positions, uvs, normals, faces = [], [], [], []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                positions.append([float(x) for x in parts[1:4]])
            elif tag == "vt":
                uvs.append([float(x) for x in parts[1:3]])
            elif tag == "vn":
                normals.append([float(x) for x in parts[1:4]])
            elif tag == "f":
                corners = []
                for tok in parts[1:]:
                    a = tok.split("/")
                    vi = int(a[0]) - 1 if a[0] else -1
                    ti = int(a[1]) - 1 if len(a) > 1 and a[1] else -1
                    ni = int(a[2]) - 1 if len(a) > 2 and a[2] else -1
                    corners.append((vi, ti, ni))
                if len(corners) == 3:
                    faces.append(corners)
                elif len(corners) > 3:
                    for i in range(1, len(corners) - 1):
                        faces.append([corners[0], corners[i], corners[i + 1]])
    positions = np.asarray(positions, dtype=np.float32)
    uvs_arr = np.asarray(uvs, dtype=np.float32) if uvs else None
    norms_arr = np.asarray(normals, dtype=np.float32) if normals else None
    face_corners = np.asarray(faces, dtype=np.int64)
    return positions, uvs_arr, norms_arr, face_corners


def _load_via_trimesh(path: Path):
    """Fallback for non-OBJ: emulates (positions, uvs, normals, face_corners).

    Since trimesh merges by position, per-corner data is lost, so each
    corner just indexes into the merged vertex arrays. Acceptable for
    formats that carry per-vertex (not per-corner) attributes natively.
    """
    mesh = trimesh.load(path, process=False, maintain_order=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Loaded object is not a Trimesh: {type(mesh)}")
    uvs = None
    if getattr(mesh.visual, "uv", None) is not None:
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    positions = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    F = faces.shape[0]
    face_corners = np.empty((F, 3, 3), dtype=np.int64)
    face_corners[..., 0] = faces
    face_corners[..., 1] = faces if uvs is not None else -1
    face_corners[..., 2] = faces
    return positions, uvs, normals, face_corners


def _load_corners(path: Path):
    p = Path(path)
    if p.suffix.lower() == ".obj":
        return _parse_obj_corners(p)
    return _load_via_trimesh(p)


def _face_normals(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = positions[faces[:, 0]]
    v1 = positions[faces[:, 1]]
    v2 = positions[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0).astype(np.float64)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    lens = np.where(lens < 1e-30, 1.0, lens)
    return (n / lens).astype(np.float32)


def _corner_angles(positions: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Interior angle at each corner. Shape (F, 3)."""
    out = np.zeros((faces.shape[0], 3), dtype=np.float64)
    for k in range(3):
        i0 = faces[:, k]
        i1 = faces[:, (k + 1) % 3]
        i2 = faces[:, (k + 2) % 3]
        e1 = (positions[i1] - positions[i0]).astype(np.float64)
        e2 = (positions[i2] - positions[i0]).astype(np.float64)
        e1 /= np.maximum(np.linalg.norm(e1, axis=1, keepdims=True), 1e-30)
        e2 /= np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-30)
        d = np.clip((e1 * e2).sum(axis=1), -1.0, 1.0)
        out[:, k] = np.arccos(d)
    return out.astype(np.float32)


def _unique_positions(raw_positions: np.ndarray, tol: float = 1e-6):
    """Collapse duplicate positions (same xyz) to unique indices.

    Returns (unique_positions, raw_to_unique).
    """
    q = np.round(raw_positions.astype(np.float64) / tol).astype(np.int64)
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    sq = q[order]
    starts = np.ones(len(sq), dtype=bool)
    starts[1:] = np.any(sq[1:] != sq[:-1], axis=1)
    unique_idx_per_sorted = np.cumsum(starts) - 1
    raw_to_unique = np.empty(len(raw_positions), dtype=np.int32)
    raw_to_unique[order] = unique_idx_per_sorted
    unique_positions = raw_positions[order[starts]].astype(np.float32)
    return unique_positions, raw_to_unique


def split_by_angle(positions: np.ndarray, faces: np.ndarray, angle_deg: float):
    """Split vertices where incident faces meet at dihedrals > angle_deg.

    positions: (Np, 3) unique positions
    faces:     (F, 3) int, indices into positions
    returns:   split_positions (Ns,3), split_normals (Ns,3), split_faces (F,3)
    """
    F = int(faces.shape[0])
    Np = int(positions.shape[0])
    cos_thresh = float(np.cos(np.radians(angle_deg)))

    face_n = _face_normals(positions, faces)
    corner_ang = _corner_angles(positions, faces)

    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for f in range(F):
        tri = faces[f]
        for k in range(3):
            a = int(tri[k]); b = int(tri[(k + 1) % 3])
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(f)

    vert_faces: list[list[int]] = [[] for _ in range(Np)]
    vert_corner: list[list[int]] = [[] for _ in range(Np)]
    for f in range(F):
        tri = faces[f]
        for k in range(3):
            v = int(tri[k])
            vert_faces[v].append(f)
            vert_corner[v].append(k)

    split_positions: list[np.ndarray] = []
    split_normals: list[np.ndarray] = []
    split_faces = np.full_like(faces, -1, dtype=np.int32)

    for v in range(Np):
        faces_here = vert_faces[v]
        if not faces_here:
            continue

        parent = {f: f for f in faces_here}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, f in enumerate(faces_here):
            k = vert_corner[v][i]
            tri = faces[f]
            for other_k in ((k + 1) % 3, (k + 2) % 3):
                ov = int(tri[other_k])
                key = (v, ov) if v < ov else (ov, v)
                nbrs = edge_to_faces[key]
                if len(nbrs) != 2:
                    continue
                g = nbrs[1] if nbrs[0] == f else nbrs[0]
                if float(np.dot(face_n[f], face_n[g])) >= cos_thresh:
                    union(f, g)

        groups: dict[int, list[int]] = {}
        for i, f in enumerate(faces_here):
            r = find(f)
            groups.setdefault(r, []).append(i)

        for _, idxs in groups.items():
            n_sum = np.zeros(3, dtype=np.float64)
            for i in idxs:
                f = faces_here[i]
                k = vert_corner[v][i]
                n_sum += face_n[f] * float(corner_ang[f, k])
            ln = float(np.linalg.norm(n_sum))
            if ln < 1e-30:
                n_sum = face_n[faces_here[idxs[0]]].astype(np.float64)
                ln = 1.0
            n_sum /= ln

            new_idx = len(split_positions)
            split_positions.append(positions[v])
            split_normals.append(n_sum.astype(np.float32))
            for i in idxs:
                f = faces_here[i]
                k = vert_corner[v][i]
                split_faces[f, k] = new_idx

    return (
        np.asarray(split_positions, dtype=np.float32),
        np.asarray(split_normals, dtype=np.float32),
        split_faces.astype(np.int32),
    )


def _build_tangent_mesh(positions_file, uvs_file, normals_file, face_corners):
    """Engine-authored mesh: unique (pos, uv, normal) corner triples.

    Each unique triple becomes one tangent vertex. Face indices map into
    this compact array in the same triangle order as the input file.
    """
    F = face_corners.shape[0]
    keys = []
    for f in range(F):
        for k in range(3):
            vi, ti, ni = face_corners[f, k]
            keys.append((int(vi), int(ti), int(ni)))

    key_to_idx: dict[tuple[int, int, int], int] = {}
    tangent_faces = np.empty((F, 3), dtype=np.int32)
    for f in range(F):
        for k in range(3):
            key = keys[f * 3 + k]
            idx = key_to_idx.get(key)
            if idx is None:
                idx = len(key_to_idx)
                key_to_idx[key] = idx
            tangent_faces[f, k] = idx

    Nt = len(key_to_idx)
    t_pos = np.zeros((Nt, 3), dtype=np.float32)
    t_uv = np.zeros((Nt, 2), dtype=np.float32)
    t_nrm = np.zeros((Nt, 3), dtype=np.float32)
    has_uv = uvs_file is not None
    has_nrm = normals_file is not None

    for (vi, ti, ni), idx in key_to_idx.items():
        if vi >= 0:
            t_pos[idx] = positions_file[vi]
        if has_uv and ti >= 0:
            t_uv[idx] = uvs_file[ti]
        if has_nrm and ni >= 0:
            t_nrm[idx] = normals_file[ni]

    return t_pos, t_uv, t_nrm, tangent_faces, has_uv, has_nrm


def prepare_mesh(path: str | Path, angle_deg: float = 30.0) -> PreparedMesh:
    """Load mesh and build split + tangent representations."""
    positions_file, uvs_file, normals_file, face_corners = _load_corners(Path(path))
    if positions_file.size == 0 or face_corners.size == 0:
        raise ValueError("Mesh is empty")
    if uvs_file is None:
        raise ValueError("Mesh has no UVs; UVs are required for baking")

    t_pos, t_uv, t_nrm, tangent_faces, _has_uv, has_nrm = _build_tangent_mesh(
        positions_file, uvs_file, normals_file, face_corners
    )

    unique_positions, raw_to_unique = _unique_positions(positions_file)
    faces_pos = raw_to_unique[face_corners[..., 0]].astype(np.int32)

    face_n = _face_normals(unique_positions, faces_pos)

    split_positions, split_normals, split_faces = split_by_angle(
        unique_positions, faces_pos, angle_deg
    )

    if not has_nrm:
        Nt = t_nrm.shape[0]
        acc = np.zeros((Nt, 3), dtype=np.float64)
        cnt = np.zeros(Nt, dtype=np.int32)
        for f in range(tangent_faces.shape[0]):
            for k in range(3):
                tv = int(tangent_faces[f, k])
                sv = int(split_faces[f, k])
                acc[tv] += split_normals[sv]
                cnt[tv] += 1
        cnt_safe = np.maximum(cnt[:, None], 1)
        acc /= cnt_safe
        lens = np.linalg.norm(acc, axis=1, keepdims=True)
        lens = np.where(lens < 1e-20, 1.0, lens)
        t_nrm = (acc / lens).astype(np.float32)

    return PreparedMesh(
        split_positions=split_positions,
        split_normals=split_normals,
        split_faces=split_faces,
        tangent_positions=t_pos,
        tangent_uvs=t_uv,
        tangent_normals=t_nrm,
        tangent_faces=tangent_faces,
        face_normals=face_n,
    )
