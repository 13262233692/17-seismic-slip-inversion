import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List
import config


def render_isosurface(
    velocity_anomaly: np.ndarray,
    dx: float, dy: float, dz: float,
    isovalue_low: float = None,
    isovalue_high: float = None,
    title: str = "3D Velocity Anomaly",
) -> go.Figure:
    if isovalue_low is None:
        isovalue_low = config.ISOVALUE_LOW
    if isovalue_high is None:
        isovalue_high = config.ISOVALUE_HIGH

    nz, ny, nx = velocity_anomaly.shape

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    V = velocity_anomaly.flatten()

    fig = go.Figure()

    fig.add_trace(go.Isosurface(
        x=X, y=Y, z=Z, value=V,
        isomin=isovalue_low,
        isomax=isovalue_high,
        surface_count=4,
        colorscale="RdBu_r",
        colorbar=dict(title="δV/V (%)"),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6,
        name="Low-Velocity Zone",
    ))

    fig.add_trace(go.Isosurface(
        x=X, y=Y, z=Z, value=V,
        isomin=isovalue_high,
        isomax=velocity_anomaly.max() + 0.01,
        surface_count=2,
        colorscale="Reds",
        colorbar=dict(title="δV/V (%)", x=1.1),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.3,
        name="High-Velocity Zone",
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Depth (km)",
            zaxis=dict(autorange="reversed"),
        ),
        width=900,
        height=700,
    )

    return fig


def render_cross_sections(
    velocity_anomaly: np.ndarray,
    dx: float, dy: float, dz: float,
    x_slice: Optional[int] = None,
    y_slice: Optional[int] = None,
    z_slice: Optional[int] = None,
    title: str = "Velocity Anomaly Cross-Sections",
) -> go.Figure:
    nz, ny, nx = velocity_anomaly.shape

    if x_slice is None:
        x_slice = nx // 2
    if y_slice is None:
        y_slice = ny // 2
    if z_slice is None:
        z_slice = nz // 2

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"X-Z Section (Y={y_slice * dy:.1f}km)",
            f"Y-Z Section (X={x_slice * dx:.1f}km)",
            f"X-Y Section (Z={z_slice * dz:.1f}km)",
        ),
    )

    x_coords = np.arange(nx) * dx
    y_coords = np.arange(ny) * dy
    z_coords = np.arange(nz) * dz

    xz_slice = velocity_anomaly[:, y_slice, :]
    fig.add_trace(
        go.Heatmap(
            z=xz_slice, x=x_coords, y=z_coords,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="δV/V", x=0.3, len=0.7),
        ),
        row=1, col=1,
    )

    yz_slice = velocity_anomaly[:, :, x_slice]
    fig.add_trace(
        go.Heatmap(
            z=yz_slice, x=y_coords, y=z_coords,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="δV/V", x=0.65, len=0.7),
        ),
        row=1, col=2,
    )

    xy_slice = velocity_anomaly[z_slice, :, :]
    fig.add_trace(
        go.Heatmap(
            z=xy_slice, x=x_coords, y=y_coords,
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="δV/V", x=1.0, len=0.7),
        ),
        row=1, col=3,
    )

    fig.update_layout(title=title, width=1200, height=400)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    return fig


def render_fault_zones(
    velocity_anomaly: np.ndarray,
    dx: float, dy: float, dz: float,
    fault_threshold: float = None,
    title: str = "Potential Fault Zones (Low-Velocity Anomalies)",
) -> go.Figure:
    if fault_threshold is None:
        fault_threshold = config.ISOVALUE_LOW

    nz, ny, nx = velocity_anomaly.shape

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    V = velocity_anomaly.flatten()

    fig = go.Figure()

    fault_mask = V < fault_threshold
    if np.any(fault_mask):
        fig.add_trace(go.Isosurface(
            x=X[fault_mask], y=Y[fault_mask], z=Z[fault_mask], value=V[fault_mask],
            isomin=V[fault_mask].min(),
            isomax=fault_threshold,
            surface_count=3,
            colorscale="Blues_r",
            colorbar=dict(title="δV/V"),
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.8,
            name="Fault Zone",
        ))

    background_mask = V >= fault_threshold
    if np.any(background_mask):
        fig.add_trace(go.Isosurface(
            x=X, y=Y, z=Z, value=V,
            isomin=fault_threshold,
            isomax=0,
            surface_count=2,
            colorscale="Greys",
            caps=dict(x_show=False, y_show=False, z_show=False),
            opacity=0.15,
            name="Background",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Depth (km)",
            zaxis=dict(autorange="reversed"),
        ),
        width=900,
        height=700,
    )

    return fig
