"""Stage 2: Embree BVH wrapper with multi-hit support.

`embreex` 0.1.6 binds Embree 2 and only exposes `rtcIntersect` /
`rtcOccluded`; the intersection filter callback API is not reachable
from Python. To obtain multi-hit we march the ray origin forward by
`t_prev + epsilon` and re-shoot up to `max_hits` times. At most
`LOCAL_MAX_HITS = 4` calls per ray, matching Cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from embreex.mesh_construction import TriangleMesh
from embreex.rtcore_scene import EmbreeScene


LOCAL_MAX_HITS = 4

_EPS_REL = 1e-5
_EPS_ABS = 1e-5


@dataclass
class MultiHitResult:
    """Per-ray hit records, padded to max_hits columns.

    All arrays are shape (N, max_hits) except `Ng` which is (N, max_hits, 3)
    and `hit_count` which is (N,). Slots past `hit_count[i]` are invalid
    (primID == -1, tfar == -1.0).
    """

    tfar: np.ndarray      # float32, cumulative distance from original origin
    primID: np.ndarray    # int32
    u: np.ndarray         # float32, barycentric
    v: np.ndarray         # float32, barycentric
    Ng: np.ndarray        # float32, unnormalized geometric normal from Embree
    hit_count: np.ndarray # int32


class BVH:
    """Thin wrapper holding one committed Embree scene + one triangle mesh."""

    def __init__(self, positions: np.ndarray, faces: np.ndarray) -> None:
        self.positions = np.ascontiguousarray(positions, dtype=np.float32)
        self.faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self.scene = EmbreeScene()
        self._mesh = TriangleMesh(self.scene, self.positions, self.faces)

    def multi_hit(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        tmin: float | np.ndarray = 0.0,
        tmax: float | np.ndarray = np.inf,
        max_hits: int = LOCAL_MAX_HITS,
    ) -> MultiHitResult:
        """Batched multi-hit query.

        origins: (N, 3), directions: (N, 3) unit vectors.
        tmin / tmax: scalar or (N,). Distances are along the (unit) direction.
        Returned `tfar` is the cumulative distance from the original origin.
        """
        if max_hits < 1:
            raise ValueError("max_hits must be >= 1")
        N = int(origins.shape[0])
        if directions.shape != origins.shape:
            raise ValueError("origins and directions must have the same shape")

        tmin_arr = np.broadcast_to(np.asarray(tmin, dtype=np.float32), (N,)).copy()
        tmax_arr = np.broadcast_to(np.asarray(tmax, dtype=np.float32), (N,)).copy()

        cur_origin = (origins.astype(np.float32) + directions.astype(np.float32) * tmin_arr[:, None]).copy()
        cur_dir = np.ascontiguousarray(directions, dtype=np.float32)
        cumulative = tmin_arr.copy()
        remaining = (tmax_arr - tmin_arr).astype(np.float32)
        alive = np.ones(N, dtype=bool) & (remaining > 0.0)

        out_t = np.full((N, max_hits), -1.0, dtype=np.float32)
        out_pid = np.full((N, max_hits), -1, dtype=np.int32)
        out_u = np.zeros((N, max_hits), dtype=np.float32)
        out_v = np.zeros((N, max_hits), dtype=np.float32)
        out_Ng = np.zeros((N, max_hits, 3), dtype=np.float32)
        hit_count = np.zeros(N, dtype=np.int32)

        # Compact alive rays between iterations: scene.run() is the hot call
        # and we should only feed it rays still in flight. `active_idx` maps
        # positions in the compact batch back to the original (N-sized) slots.
        active_idx = np.nonzero(alive)[0]
        comp_origin = cur_origin[active_idx]
        comp_dir = cur_dir[active_idx]

        for hit_i in range(max_hits):
            if active_idx.size == 0:
                break

            r = self.scene.run(comp_origin, comp_dir, query="INTERSECT", output=True)
            pid = r["primID"]
            tfar = r["tfar"]

            rem_active = remaining[active_idx]
            budget_slack = np.maximum(rem_active * _EPS_REL, _EPS_ABS)
            comp_hit_mask = (pid != -1) & (tfar <= rem_active + budget_slack)

            comp_idx = np.nonzero(comp_hit_mask)[0]
            if comp_idx.size:
                orig_idx = active_idx[comp_idx]
                t_hits = tfar[comp_idx]
                out_t[orig_idx, hit_i] = cumulative[orig_idx] + t_hits
                out_pid[orig_idx, hit_i] = pid[comp_idx]
                out_u[orig_idx, hit_i] = r["u"][comp_idx]
                out_v[orig_idx, hit_i] = r["v"][comp_idx]
                out_Ng[orig_idx, hit_i] = r["Ng"][comp_idx]
                hit_count[orig_idx] += 1

                step = t_hits + np.maximum(t_hits * _EPS_REL, _EPS_ABS)
                # Update accounting only for rays that hit and still have budget;
                # build the next compact batch from those survivors.
                new_remaining = remaining[orig_idx] - step
                remaining[orig_idx] = new_remaining
                cumulative[orig_idx] += step

                survive_mask = new_remaining > 0.0
                if hit_i + 1 < max_hits and survive_mask.any():
                    surv_comp = comp_idx[survive_mask]
                    survivors = orig_idx[survive_mask]
                    new_origin = comp_origin[surv_comp] + comp_dir[surv_comp] * step[survive_mask][:, None]
                    comp_origin = new_origin
                    comp_dir = comp_dir[surv_comp]
                    active_idx = survivors
                else:
                    active_idx = np.empty(0, dtype=np.int64)
            else:
                break

        return MultiHitResult(
            tfar=out_t,
            primID=out_pid,
            u=out_u,
            v=out_v,
            Ng=out_Ng,
            hit_count=hit_count,
        )

    def multi_hit_single(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        tmin: float = 0.0,
        tmax: float = np.inf,
        max_hits: int = LOCAL_MAX_HITS,
    ) -> list[dict]:
        """Convenience scalar form used in tests. Returns a list sorted by distance."""
        o = np.asarray(origin, dtype=np.float32).reshape(1, 3)
        d = np.asarray(direction, dtype=np.float32).reshape(1, 3)
        r = self.multi_hit(o, d, tmin, tmax, max_hits)
        n = int(r.hit_count[0])
        return [
            {
                "tfar": float(r.tfar[0, i]),
                "primID": int(r.primID[0, i]),
                "u": float(r.u[0, i]),
                "v": float(r.v[0, i]),
                "Ng": r.Ng[0, i].copy(),
            }
            for i in range(n)
        ]
