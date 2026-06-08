import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Dict
import config
from seismic_tomography.focal_mechanism import (
    generate_beachball_contours,
    generate_beachball_svg,
    generate_beachball_plotly_3d,
    _beachball_projection,
    classify_fault_type,
)


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


def render_beachball_2d(
    strike: float, dip: float, rake: float,
    observations: Optional[List[dict]] = None,
    title: str = None,
) -> go.Figure:
    if title is None:
        fault_type = classify_fault_type(strike, dip, rake)
        title = f"Focal Mechanism: Strike={strike:.0f}° Dip={dip:.0f}° Rake={rake:.0f}° ({fault_type})"

    contours = generate_beachball_contours(strike, dip, rake)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=contours["circle_x"], y=contours["circle_y"],
        mode="lines", line=dict(color="black", width=2),
        showlegend=False,
    ))

    if contours["compressive_x"]:
        fig.add_trace(go.Scatter(
            x=contours["compressive_x"], y=contours["compressive_y"],
            mode="markers", marker=dict(size=3, color="#E03030", opacity=0.8),
            name="Compressive (+)",
        ))

    if contours["dilatational_x"]:
        fig.add_trace(go.Scatter(
            x=contours["dilatational_x"], y=contours["dilatational_y"],
            mode="markers", marker=dict(size=3, color="#3030E0", opacity=0.8),
            name="Dilatational (-)",
        ))

    if observations:
        for obs in observations:
            x, y = _beachball_projection(obs["azimuth"], obs["takeoff"])
            color = "black" if obs["polarity"] > 0 else "white"
            symbol = "circle" if obs["polarity"] > 0 else "circle-open"
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers",
                marker=dict(size=10, color=color, symbol=symbol, line=dict(color="black", width=1.5)),
                showlegend=False,
            ))

    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", scaleratio=1, range=[-1.3, 1.3], title="E-W"),
        yaxis=dict(range=[-1.3, 1.3], title="N-S"),
        width=600,
        height=600,
        plot_bgcolor="white",
    )

    return fig


def render_fault_zones_with_beachballs(
    velocity_anomaly: np.ndarray,
    dx: float, dy: float, dz: float,
    focal_mechanisms: List[Dict],
    fault_threshold: float = None,
    beachball_scale: float = 5.0,
    title: str = "Fault Zones with Focal Mechanisms",
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
            opacity=0.6,
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
            opacity=0.1,
            name="Background",
        ))

    for i, fm in enumerate(focal_mechanisms):
        bb_data = generate_beachball_plotly_3d(
            strike=fm["strike"],
            dip=fm["dip"],
            rake=fm["rake"],
            observations=fm.get("observations"),
            center_x=fm["x"],
            center_y=fm["y"],
            center_z=fm["z"],
            scale=beachball_scale,
        )

        vals = np.array(bb_data["value"])
        fig.add_trace(go.Scatter3d(
            x=bb_data["x"], y=bb_data["y"], z=bb_data["z"],
            mode="markers",
            marker=dict(
                size=3,
                color=vals,
                colorscale=[[0, "#3030E0"], [0.5, "#808080"], [1, "#E03030"]],
                cmin=vals.min(), cmax=vals.max(),
                opacity=0.9,
            ),
            name=f"Beachball {i + 1}: S={fm['strike']:.0f}° D={fm['dip']:.0f}° R={fm['rake']:.0f}°",
            showlegend=True,
        ))

        if bb_data["obs_x"]:
            obs_pol = np.array(bb_data["obs_polarity"])
            obs_colors = ["black" if p > 0 else "white" for p in obs_pol]
            fig.add_trace(go.Scatter3d(
                x=bb_data["obs_x"], y=bb_data["obs_y"], z=bb_data["obs_z"],
                mode="markers",
                marker=dict(size=6, color=obs_colors, line=dict(color="black", width=1)),
                showlegend=False,
            ))

        fig.add_trace(go.Scatter3d(
            x=[fm["x"]], y=[fm["y"]], z=[fm["z"]],
            mode="markers",
            marker=dict(size=8, color="yellow", symbol="diamond", line=dict(color="black", width=2)),
            name=f"Epicenter {i + 1}",
            showlegend=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Depth (km)",
            zaxis=dict(autorange="reversed"),
        ),
        width=1000,
        height=800,
    )

    return fig
