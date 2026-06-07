import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from typing import Tuple, Optional, List
import config


class VelocityModel:
    def __init__(
        self,
        nx: int = None,
        ny: int = None,
        nz: int = None,
        x_km: float = None,
        y_km: float = None,
        z_km: float = None,
        vp: float = 6.0,
        vs: float = 3.46,
    ):
        self.nx = nx or config.GRID_NX
        self.ny = ny or config.GRID_NY
        self.nz = nz or config.GRID_NZ
        self.x_km = x_km or config.GRID_X_KM
        self.y_km = y_km or config.GRID_Y_KM
        self.z_km = z_km or config.GRID_Z_KM
        self.vp = vp
        self.vs = vs

        self.dx = self.x_km / self.nx
        self.dy = self.y_km / self.ny
        self.dz = self.z_km / self.nz

        self.n_cells = self.nx * self.ny * self.nz
        self.velocity = np.ones(self.n_cells) * vp
        self.dv = np.zeros(self.n_cells)

    def _cell_index(self, ix: int, iy: int, iz: int) -> int:
        return iz * self.nx * self.ny + iy * self.nx + ix

    def _cell_coords(self, idx: int) -> Tuple[int, int, int]:
        iz = idx // (self.nx * self.ny)
        remainder = idx % (self.nx * self.ny)
        iy = remainder // self.nx
        ix = remainder % self.nx
        return ix, iy, iz

    def _xyz_to_cell(self, x: float, y: float, z: float) -> Optional[int]:
        ix = int(x / self.dx)
        iy = int(y / self.dy)
        iz = int(z / self.dz)
        if 0 <= ix < self.nx and 0 <= iy < self.ny and 0 <= iz < self.nz:
            return self._cell_index(ix, iy, iz)
        return None

    def get_cell_centers(self) -> np.ndarray:
        centers = np.zeros((self.n_cells, 3))
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    idx = self._cell_index(ix, iy, iz)
                    centers[idx, 0] = (ix + 0.5) * self.dx
                    centers[idx, 1] = (iy + 0.5) * self.dy
                    centers[idx, 2] = (iz + 0.5) * self.dz
        return centers

    def get_3d_velocity(self) -> np.ndarray:
        return self.velocity.reshape(self.nz, self.ny, self.nx)

    def get_3d_anomaly(self) -> np.ndarray:
        return self.dv.reshape(self.nz, self.ny, self.nx)


class Station:
    def __init__(self, name: str, network: str, x_km: float, y_km: float, z_km: float = 0.0):
        self.name = name
        self.network = network
        self.x = x_km
        self.y = y_km
        self.z = z_km


class Event:
    def __init__(self, x_km: float, y_km: float, z_km: float, origin_time: float = 0.0):
        self.x = x_km
        self.y = y_km
        self.z = z_km
        self.origin_time = origin_time


def _trace_ray_straight(
    src: Tuple[float, float, float],
    rcv: Tuple[float, float, float],
    model: VelocityModel,
) -> dict:
    sx, sy, sz = src
    rx, ry, rz = rcv
    distance = np.sqrt((rx - sx) ** 2 + (ry - sy) ** 2 + (rz - sz) ** 2)
    if distance < 1e-6:
        return {"path_length": 0.0, "cells": {}, "distance": 0.0}

    min_cell = min(model.dx, model.dy, model.dz)
    n_steps = max(int(distance / min_cell * 2), 100)
    n_steps = min(n_steps, 5000)

    dx_step = (rx - sx) / n_steps
    dy_step = (ry - sy) / n_steps
    dz_step = (rz - sz) / n_steps
    step_len = distance / n_steps

    px_all = sx + np.arange(n_steps + 1) * dx_step
    py_all = sy + np.arange(n_steps + 1) * dy_step
    pz_all = sz + np.arange(n_steps + 1) * dz_step

    ix_all = (px_all / model.dx).astype(np.int32)
    iy_all = (py_all / model.dy).astype(np.int32)
    iz_all = (pz_all / model.dz).astype(np.int32)

    valid = (
        (ix_all >= 0) & (ix_all < model.nx) &
        (iy_all >= 0) & (iy_all < model.ny) &
        (iz_all >= 0) & (iz_all < model.nz)
    )

    ix_valid = ix_all[valid]
    iy_valid = iy_all[valid]
    iz_valid = iz_all[valid]

    cell_indices = iz_valid * (model.nx * model.ny) + iy_valid * model.nx + ix_valid

    unique_cells, counts = np.unique(cell_indices, return_counts=True)
    path_lengths = counts.astype(np.float64) * step_len

    cells = dict(zip(unique_cells.tolist(), path_lengths.tolist()))

    return {"path_length": distance, "cells": cells, "distance": distance}


def build_sensitivity_matrix(
    model: VelocityModel,
    stations: List[Station],
    events: List[Event],
    phase: str = "P",
) -> Tuple[csr_matrix, np.ndarray]:
    n_rays = len(events) * len(stations)
    n_params = model.n_cells
    v0 = model.vp if phase == "P" else model.vs
    coeff = -1.0 / (v0 ** 2)

    row_list = []
    col_list = []
    val_list = []
    dt = np.zeros(n_rays)

    ray_idx = 0
    for ev in events:
        for sta in stations:
            src = (ev.x, ev.y, ev.z)
            rcv = (sta.x, sta.y, sta.z)

            ray = _trace_ray_straight(src, rcv, model)
            if ray["path_length"] > 0 and ray["cells"]:
                cells = ray["cells"]
                cell_ids = np.array(list(cells.keys()), dtype=np.int64)
                path_lens = np.array(list(cells.values()), dtype=np.float64)

                row_list.append(np.full(len(cell_ids), ray_idx, dtype=np.int64))
                col_list.append(cell_ids)
                val_list.append(coeff * path_lens)

                dt[ray_idx] = ray["distance"] / v0

            ray_idx += 1

    if row_list:
        rows = np.concatenate(row_list)
        cols = np.concatenate(col_list)
        vals = np.concatenate(val_list)
        G = coo_matrix((vals, (rows, cols)), shape=(n_rays, n_params)).tocsr()
    else:
        G = csr_matrix((n_rays, n_params))

    return G, dt


def build_travel_time_residuals(
    model: VelocityModel,
    stations: List[Station],
    events: List[Event],
    observed_arrivals: np.ndarray,
    phase: str = "P",
) -> Tuple[csr_matrix, np.ndarray]:
    G, t_calc = build_sensitivity_matrix(model, stations, events, phase)
    residuals = observed_arrivals - t_calc
    return G, residuals
