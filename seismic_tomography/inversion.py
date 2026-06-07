import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import lsqr, svds
from typing import Tuple, Optional
import config


def build_3d_laplacian_csr(
    nx: int, ny: int, nz: int, alpha: float
) -> csr_matrix:
    n = nx * ny * nz
    if n == 0 or alpha == 0:
        return csr_matrix((0, n))

    ix = np.arange(nx, dtype=np.int64)
    iy = np.arange(ny, dtype=np.int64)
    iz = np.arange(nz, dtype=np.int64)
    IX, IY, IZ = np.meshgrid(ix, iy, iz, indexing="ij")
    IX = IX.ravel()
    IY = IY.ravel()
    IZ = IZ.ravel()

    cell = IZ * (nx * ny) + IY * nx + IX

    rows_off = []
    cols_off = []

    for d_ix, d_iy, d_iz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        nix = IX + d_ix
        niy = IY + d_iy
        niz = IZ + d_iz

        valid = (
            (nix >= 0) & (nix < nx) &
            (niy >= 0) & (niy < ny) &
            (niz >= 0) & (niz < nz)
        )

        valid_cell = cell[valid]
        neighbor_cell = niz[valid] * (nx * ny) + niy[valid] * nx + nix[valid]

        rows_off.append(valid_cell)
        cols_off.append(neighbor_cell)

    all_rows_off = np.concatenate(rows_off)
    all_cols_off = np.concatenate(cols_off)
    n_off = len(all_rows_off)

    off_val = np.full(n_off, -alpha, dtype=np.float64)

    ones_off = np.ones(n_off, dtype=np.float64)
    diag_contrib = np.zeros(n, dtype=np.float64)
    np.add.at(diag_contrib, all_rows_off, ones_off)
    diag_contrib *= alpha

    all_rows = np.concatenate([all_rows_off, np.arange(n, dtype=np.int64)])
    all_cols = np.concatenate([all_cols_off, np.arange(n, dtype=np.int64)])
    all_vals = np.concatenate([off_val, diag_contrib])

    L = coo_matrix(
        (all_vals, (all_rows, all_cols)), shape=(n, n)
    ).tocsr()

    return L


def _row_scale_normalize(
    G: csr_matrix, dt: np.ndarray
) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    row_norms = np.sqrt(np.asarray(G.power(2).sum(axis=1)).flatten())
    row_norms[row_norms < 1e-12] = 1.0
    inv_norms = 1.0 / row_norms

    D = diags(inv_norms, format="csr")
    G_scaled = D @ G
    dt_scaled = inv_norms * dt

    return G_scaled, dt_scaled, row_norms


def _build_augmented_system_coo(
    G: csr_matrix,
    dt: np.ndarray,
    nx: int, ny: int, nz: int,
    damping: float,
    smoothing: float,
) -> Tuple[csr_matrix, np.ndarray]:
    n_data = G.shape[0]
    n_params = G.shape[1]

    G_coo = G.tocoo()
    data_rows = G_coo.row.copy()
    data_cols = G_coo.col.copy()
    data_vals = G_coo.data.copy()

    all_rows = [data_rows]
    all_cols = [data_cols]
    all_vals = [data_vals]
    rhs_parts = [dt.copy()]

    current_row = n_data

    if damping > 0:
        damp_n = n_params
        damp_rows = np.arange(damp_n, dtype=np.int64) + current_row
        damp_cols = np.arange(n_params, dtype=np.int64)
        damp_vals = np.full(damp_n, damping, dtype=np.float64)

        all_rows.append(damp_rows)
        all_cols.append(damp_cols)
        all_vals.append(damp_vals)
        rhs_parts.append(np.zeros(damp_n))

        current_row += damp_n

    if smoothing > 0:
        L = build_3d_laplacian_csr(nx, ny, nz, smoothing)
        if L.shape[0] > 0 and L.nnz > 0:
            L_coo = L.tocoo()
            smooth_rows = L_coo.row.copy().astype(np.int64) + current_row
            smooth_cols = L_coo.col.copy().astype(np.int64)
            smooth_vals = L_coo.data.copy()

            all_rows.append(smooth_rows)
            all_cols.append(smooth_cols)
            all_vals.append(smooth_vals)
            rhs_parts.append(np.zeros(L.shape[0]))

            current_row += L.shape[0]

    total_rows = current_row
    merged_rows = np.concatenate(all_rows)
    merged_cols = np.concatenate(all_cols)
    merged_vals = np.concatenate(all_vals)
    merged_rhs = np.concatenate(rhs_parts)

    G_aug = coo_matrix(
        (merged_vals, (merged_rows, merged_cols)),
        shape=(total_rows, n_params),
    ).tocsr()

    return G_aug, merged_rhs


def _col_norm_precondition(
    G_aug: csr_matrix, rhs: np.ndarray
) -> Tuple[csr_matrix, np.ndarray, csr_matrix]:
    col_sq_sum = np.zeros(G_aug.shape[1], dtype=np.float64)
    G_csc = G_aug.tocsc()

    for j in range(G_aug.shape[1]):
        start = G_csc.indptr[j]
        end = G_csc.indptr[j + 1]
        if start < end:
            col_sq_sum[j] = np.sum(G_csc.data[start:end] ** 2)

    col_diag = np.sqrt(col_sq_sum)
    col_diag[col_diag < 1e-12] = 1.0
    D_inv = diags(1.0 / col_diag, format="csr")
    G_precond = G_aug @ D_inv

    return G_precond, rhs, D_inv


def lsqr_inversion(
    G: csr_matrix,
    dt: np.ndarray,
    nx: int = None,
    ny: int = None,
    nz: int = None,
    damping: float = None,
    smoothing: float = None,
    max_iterations: int = None,
    normalize: bool = True,
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

    if normalize:
        G_work, dt_work, row_norms = _row_scale_normalize(G, dt)
    else:
        G_work = G
        dt_work = dt.copy()
        row_norms = np.ones(G.shape[0])

    G_aug, rhs = _build_augmented_system_coo(
        G_work, dt_work, nx, ny, nz, damping, smoothing
    )

    G_precond, rhs_precond, D_inv = _col_norm_precondition(G_aug, rhs)

    result = lsqr(G_precond, rhs_precond, iter_lim=max_iterations, show=False)

    dm_scaled = result[0]
    dm = D_inv @ dm_scaled

    residual = dt - G @ dm
    rms_data = np.sqrt(np.mean(residual ** 2))

    info = {
        "iterations": result[2],
        "residual_norm": result[3],
        "solution_norm": float(np.linalg.norm(dm)),
        "rms_data_residual": float(rms_data),
        "augmented_rows": G_aug.shape[0],
        "augmented_cols": G_aug.shape[1],
        "augmented_nnz": G_aug.nnz,
        "memory_mb": float(G_aug.nnz * 16 / 1e6),
        "normalized": normalize,
    }

    return dm, info


def svd_inversion(
    G: csr_matrix,
    dt: np.ndarray,
    nx: int = None,
    ny: int = None,
    nz: int = None,
    damping: float = None,
    n_singular: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, dict]:
    if damping is None:
        damping = config.DAMPING
    if nx is None:
        nx = config.GRID_NX
    if ny is None:
        ny = config.GRID_NY
    if nz is None:
        nz = config.GRID_NZ

    if normalize:
        G_work, dt_work, row_norms = _row_scale_normalize(G, dt)
    else:
        G_work = G
        dt_work = dt.copy()

    G_aug, rhs = _build_augmented_system_coo(
        G_work, dt_work, nx, ny, nz, damping, 0.0
    )

    m, n = G_aug.shape
    k = n_singular or min(m, n) - 2
    k = max(2, min(k, min(m, n) - 2))

    try:
        u, s, vt = svds(G_aug, k=k, which="LM")
    except Exception:
        k_safe = max(2, min(50, min(m, n) - 2))
        u, s, vt = svds(G_aug, k=k_safe, which="LM")

    sort_idx = np.argsort(s)[::-1]
    s = s[sort_idx]
    u = u[:, sort_idx]
    vt = vt[sort_idx, :]

    variance_explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    n_keep = np.searchsorted(variance_explained, 0.99) + 1
    n_keep = max(1, n_keep)

    s_inv = np.zeros_like(s)
    s_inv[:n_keep] = 1.0 / s[:n_keep]

    proj = (u[:, :n_keep].T @ rhs) * s_inv[:n_keep]
    dm = vt[:n_keep].T @ proj

    residual = dt - G @ dm
    rms_data = np.sqrt(np.mean(residual ** 2))

    info = {
        "n_singular_used": int(n_keep),
        "total_singular": int(k),
        "singular_values": s[:min(20, len(s))].tolist(),
        "variance_explained": float(variance_explained[n_keep - 1]),
        "rms_data_residual": float(rms_data),
        "augmented_rows": G_aug.shape[0],
        "augmented_nnz": G_aug.nnz,
        "memory_mb": float(G_aug.nnz * 16 / 1e6),
    }

    return dm, info


def iterative_inversion(
    G: csr_matrix,
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
            "max_update": float(np.max(np.abs(dm))),
            "info": info,
        })

    return dm_total, history
