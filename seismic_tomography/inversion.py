import numpy as np
from scipy.sparse import csc_matrix, diags, vstack as sparse_vstack
from scipy.sparse.linalg import lsqr
from typing import Tuple, Optional
import config


def _build_damping_matrix(n_params: int, damping: float) -> csc_matrix:
    return damping * diags(np.ones(n_params), 0, shape=(n_params, n_params), format="csc")


def _build_smoothing_matrix(
    nx: int, ny: int, nz: int, smoothing: float
) -> csc_matrix:
    n_cells = nx * ny * nz
    rows = []
    cols = []
    vals = []

    def idx(ix, iy, iz):
        return iz * nx * ny + iy * nx + ix

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                c = idx(ix, iy, iz)
                if ix + 1 < nx:
                    r = len(rows) // 2
                    rows.extend([r * 2, r * 2])
                    cols.extend([c, idx(ix + 1, iy, iz)])
                    vals.extend([-smoothing, smoothing])
                if iy + 1 < ny:
                    r = len(rows) // 2
                    rows.extend([r * 2, r * 2])
                    cols.extend([c, idx(ix, iy + 1, iz)])
                    vals.extend([-smoothing, smoothing])
                if iz + 1 < nz:
                    r = len(rows) // 2
                    rows.extend([r * 2, r * 2])
                    cols.extend([c, idx(ix, iy, iz + 1)])
                    vals.extend([-smoothing, smoothing])

    if not rows:
        return csc_matrix((0, n_cells))

    n_smooth = max(rows) + 1
    L = csc_matrix((vals, (rows, cols)), shape=(n_smooth, n_cells))
    return L


def lsqr_inversion(
    G: csc_matrix,
    dt: np.ndarray,
    nx: int = None,
    ny: int = None,
    nz: int = None,
    damping: float = None,
    smoothing: float = None,
    max_iterations: int = None,
) -> Tuple[np.ndarray, dict]:
    if nx is None:
        nx = config.GRID_NX
    if ny is None:
        ny = config.GRID_NY
    if nz is None:
        nz = config.GRID_NZ
    if damping is None:
        damping = config.DAMPING
    if smoothing is None:
        smoothing = config.SMOOTHING
    if max_iterations is None:
        max_iterations = config.MAX_ITERATIONS

    n_params = G.shape[1]
    n_data = G.shape[0]

    D = _build_damping_matrix(n_params, damping)
    dt_extended = np.concatenate([dt, np.zeros(n_params)])

    G_aug = sparse_vstack([G, D], format="csc")

    L = _build_smoothing_matrix(nx, ny, nz, smoothing)
    if L.shape[0] > 0:
        dt_extended = np.concatenate([dt_extended, np.zeros(L.shape[0])])
        G_aug = sparse_vstack([G_aug, L], format="csc")

    result = lsqr(G_aug, dt_extended, iter_lim=max_iterations, show=False)

    dm = result[0]
    info = {
        "iterations": result[2],
        "residual_norm": result[3],
        "solution_norm": result[6],
        "condition_number": result[7] if len(result) > 7 else None,
    }

    return dm, info


def svd_inversion(
    G: csc_matrix,
    dt: np.ndarray,
    nx: int = None,
    ny: int = None,
    nz: int = None,
    damping: float = None,
    n_singular: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    if damping is None:
        damping = config.DAMPING
    if nx is None:
        nx = config.GRID_NX
    if ny is None:
        ny = config.GRID_NY
    if nz is None:
        nz = config.GRID_NZ

    n_params = G.shape[1]

    D = _build_damping_matrix(n_params, damping)
    dt_extended = np.concatenate([dt, np.zeros(n_params)])
    G_aug = sparse_vstack([G, D], format="csc")

    G_dense = G_aug.toarray()

    U, s, Vt = np.linalg.svd(G_dense, full_matrices=False)

    if n_singular is not None:
        n_keep = min(n_singular, len(s))
    else:
        variance_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
        n_keep = np.searchsorted(variance_explained, 0.99) + 1

    s_inv = np.zeros_like(s)
    s_inv[:n_keep] = 1.0 / s[:n_keep]

    dm = Vt[:n_keep].T @ np.diag(s_inv[:n_keep]) @ U[:, :n_keep].T @ dt_extended

    info = {
        "n_singular_used": n_keep,
        "total_singular": len(s),
        "singular_values": s[:min(20, len(s))],
        "variance_explained": float(np.sum(s[:n_keep] ** 2) / np.sum(s ** 2)),
    }

    return dm, info


def iterative_inversion(
    G: csc_matrix,
    dt: np.ndarray,
    nx: int = None,
    ny: int = None,
    nz: int = None,
    damping: float = None,
    smoothing: float = None,
    max_iterations: int = None,
    n_outer: int = 3,
) -> Tuple[np.ndarray, list]:
    if nx is None:
        nx = config.GRID_NX
    if ny is None:
        ny = config.GRID_NY
    if nz is None:
        nz = config.GRID_NZ
    if damping is None:
        damping = config.DAMPING
    if smoothing is None:
        smoothing = config.SMOOTHING
    if max_iterations is None:
        max_iterations = config.MAX_ITERATIONS

    dm_total = np.zeros(G.shape[1])
    dt_current = dt.copy()
    history = []

    for i in range(n_outer):
        dm, info = lsqr_inversion(
            G, dt_current, nx, ny, nz, damping, smoothing, max_iterations
        )
        dm_total += dm
        dt_current = dt - G @ dm_total
        rms = np.sqrt(np.mean(dt_current ** 2))
        history.append({
            "iteration": i + 1,
            "rms_residual": rms,
            "max_update": np.max(np.abs(dm)),
            "info": info,
        })

    return dm_total, history
