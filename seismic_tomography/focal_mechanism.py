import numpy as np
from typing import Tuple, List, Optional, Dict
import config


def compute_p_polarity(
    st,
    picks: List[dict],
    window_sec: float = 0.2,
) -> List[dict]:
    results = []
    for pick in picks:
        if pick["phase"] != "P":
            continue

        net = pick["network"]
        sta = pick["station"]
        pick_time = pick["arrival_time"]

        matching = [
            tr for tr in st.select(network=net, station=sta)
            if tr.stats.channel and tr.stats.channel[-1].upper() in ("Z", "1")
        ]

        if not matching:
            results.append({**pick, "polarity": 0})
            continue

        tr = matching[0]
        sr = tr.stats.sampling_rate
        pick_sample = int((pick_time - tr.stats.starttime) * sr)
        win_samples = int(window_sec * sr)

        start = max(0, pick_sample - 1)
        end = min(len(tr.data), pick_sample + win_samples)

        if end - start < 3:
            results.append({**pick, "polarity": 0})
            continue

        segment = tr.data[start:end].astype(np.float64)

        baseline = np.mean(tr.data[max(0, pick_sample - int(0.5 * sr)):pick_sample])
        segment -= baseline

        max_idx = np.argmax(np.abs(segment))
        polarity = int(np.sign(segment[max_idx]))

        results.append({**pick, "polarity": polarity})

    return results


def compute_azimuth_takeoff(
    event_x: float, event_y: float, event_z: float,
    station_x: float, station_y: float, station_z: float = 0.0,
    vp: float = 6.0, vs: float = 3.46,
) -> Tuple[float, float]:
    dx = station_x - event_x
    dy = station_y - event_y

    azimuth = np.degrees(np.arctan2(dx, dy)) % 360

    distance = np.sqrt(dx ** 2 + dy ** 2)
    depth = event_z

    if distance < 1e-6 and depth < 1e-6:
        takeoff = 0.0
    elif distance < 1e-6:
        takeoff = 0.0 if depth > 0 else 180.0
    else:
        ray_param = 0.0
        if depth > 1e-6:
            takeoff = np.degrees(np.arctan2(distance, depth))
        else:
            takeoff = 90.0

    takeoff = min(takeoff, 90.0)

    return azimuth, takeoff


def _radians(deg: float) -> float:
    return np.radians(deg)


def _compute_moment_tensor(strike: float, dip: float, rake: float) -> np.ndarray:
    sd = _radians(strike)
    dd = _radians(dip)
    lam = _radians(rake)

    M = np.zeros((3, 3))

    M[0, 0] = -(np.sin(dd) * np.cos(lam) * np.sin(2 * sd)
                 + np.sin(2 * dd) * np.sin(lam) * np.sin(sd) ** 2)
    M[1, 1] = (np.sin(dd) * np.cos(lam) * np.sin(2 * sd)
                - np.sin(2 * dd) * np.sin(lam) * np.cos(sd) ** 2)
    M[2, 2] = np.sin(2 * dd) * np.sin(lam)

    M[0, 1] = np.sin(dd) * np.cos(lam) * np.cos(2 * sd) + 0.5 * np.sin(2 * dd) * np.sin(lam) * np.sin(2 * sd)
    M[1, 0] = M[0, 1]

    M[0, 2] = -(np.cos(dd) * np.cos(lam) * np.cos(sd) - np.cos(2 * dd) * np.sin(lam) * np.sin(sd))
    M[2, 0] = M[0, 2]

    M[1, 2] = -(np.cos(dd) * np.cos(lam) * np.sin(sd) + np.cos(2 * dd) * np.sin(lam) * np.cos(sd))
    M[2, 1] = M[1, 2]

    return M


def _predict_polarity(
    azimuth: float, takeoff: float, strike: float, dip: float, rake: float
) -> int:
    M = _compute_moment_tensor(strike, dip, rake)

    az = _radians(azimuth)
    take = _radians(takeoff)

    theta = np.pi / 2 - take

    r = np.array([
        np.sin(theta) * np.sin(az),
        np.sin(theta) * np.cos(az),
        np.cos(theta),
    ])

    amplitude = r @ M @ r
    return 1 if amplitude >= 0 else -1


def grid_search_focal_mechanism(
    polarities: List[dict],
    event_x: float, event_y: float, event_z: float,
    stations: list,
    strike_step: float = 5.0,
    dip_step: float = 5.0,
    rake_step: float = 5.0,
) -> Dict:
    obs = []
    for p in polarities:
        if p.get("polarity", 0) == 0:
            continue
        sta_matches = [s for s in stations if s.name == p["station"]]
        if not sta_matches:
            continue
        sta = sta_matches[0]
        az, takeoff = compute_azimuth_takeoff(event_x, event_y, event_z, sta.x, sta.y, sta.z)
        obs.append({
            "azimuth": az,
            "takeoff": takeoff,
            "polarity": p["polarity"],
        })

    if len(obs) < 4:
        return {
            "strike": 0.0, "dip": 0.0, "rake": 0.0,
            "score": 0.0, "n_observations": len(obs),
            "fault_type": "Insufficient data",
            "observations": obs,
        }

    best_score = -1
    best_strike = 0.0
    best_dip = 45.0
    best_rake = 0.0

    strikes = np.arange(0, 360, strike_step)
    dips = np.arange(5, 91, dip_step)
    rakes = np.arange(-180, 180, rake_step)

    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                score = 0
                for o in obs:
                    pred = _predict_polarity(o["azimuth"], o["takeoff"], strike, dip, rake)
                    if pred == o["polarity"]:
                        score += 1

                if score > best_score:
                    best_score = score
                    best_strike = strike
                    best_dip = dip
                    best_rake = rake

    n_total = len(obs)
    agreement = best_score / n_total if n_total > 0 else 0.0

    fault_type = classify_fault_type(best_strike, best_dip, best_rake)

    return {
        "strike": float(best_strike),
        "dip": float(best_dip),
        "rake": float(best_rake),
        "score": agreement,
        "n_observations": n_total,
        "n_correct": best_score,
        "fault_type": fault_type,
        "observations": obs,
    }


def classify_fault_type(strike: float, dip: float, rake: float) -> str:
    if abs(rake) < 15 or abs(rake) > 165:
        return "Strike-Slip"
    elif 15 <= rake <= 75 or -165 <= rake <= -105:
        return "Thrust (Reverse)"
    elif -75 <= rake <= -15 or 105 <= rake <= 165:
        return "Normal"
    elif 75 < rake < 105:
        return "Reverse"
    elif -105 < rake < -75:
        return "Normal (Steep)"
    else:
        return "Oblique-Slip"


def _beachball_projection(azimuth: float, takeoff: float) -> Tuple[float, float]:
    theta = np.radians(takeoff)
    if theta <= np.pi / 2:
        r = np.sqrt(2.0) * np.sin(theta / 2.0)
    else:
        r = np.sqrt(2.0) * np.cos((np.pi - theta) / 2.0)

    az_rad = np.radians(azimuth)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)

    return x, y


def generate_beachball_paths(
    strike: float, dip: float, rake: float,
    n_points: int = 360,
) -> Dict:
    M = _compute_moment_tensor(strike, dip, rake)

    compressive_x = []
    compressive_y = []
    dilatational_x = []
    dilatational_y = []

    for i in range(n_points):
        azimuth = 360.0 * i / n_points
        for j in range(n_points // 4 + 1):
            takeoff = 90.0 * j / (n_points // 4)

            x, y = _beachball_projection(azimuth, takeoff)

            az_rad = np.radians(azimuth)
            theta = np.radians(takeoff)

            r_vec = np.array([
                np.sin(theta) * np.sin(az_rad),
                np.sin(theta) * np.cos(az_rad),
                np.cos(theta),
            ])

            amplitude = r_vec @ M @ r_vec

            if amplitude >= 0:
                compressive_x.append(x)
                compressive_y.append(y)
            else:
                dilatational_x.append(x)
                dilatational_y.append(y)

    return {
        "compressive_x": compressive_x,
        "compressive_y": compressive_y,
        "dilatational_x": dilatational_x,
        "dilatational_y": dilatational_y,
    }


def generate_beachball_contours(
    strike: float, dip: float, rake: float,
    n_az: int = 72,
    n_takeoff: int = 45,
) -> Dict:
    M = _compute_moment_tensor(strike, dip, rake)

    azimuths = np.linspace(0, 360, n_az, endpoint=False)
    takeoffs = np.linspace(0, 90, n_takeoff + 1)

    proj_x = np.zeros((n_az, n_takeoff + 1))
    proj_y = np.zeros((n_az, n_takeoff + 1))
    amplitude = np.zeros((n_az, n_takeoff + 1))

    for i, az in enumerate(azimuths):
        for j, take in enumerate(takeoffs):
            x, y = _beachball_projection(az, take)
            proj_x[i, j] = x
            proj_y[i, j] = y

            az_rad = np.radians(az)
            theta = np.radians(take)

            r_vec = np.array([
                np.sin(theta) * np.sin(az_rad),
                np.sin(theta) * np.cos(az_rad),
                np.cos(theta),
            ])

            amplitude[i, j] = r_vec @ M @ r_vec

    nodal_line_1_x = []
    nodal_line_1_y = []
    nodal_line_2_x = []
    nodal_line_2_y = []

    for i in range(n_az):
        for j in range(n_takeoff):
            if (amplitude[i, j] * amplitude[i, (j + 1) % (n_takeoff + 1)] < 0):
                if len(nodal_line_1_x) < n_az:
                    nodal_line_1_x.append(proj_x[i, j])
                    nodal_line_1_y.append(proj_y[i, j])
                elif len(nodal_line_2_x) < n_az:
                    nodal_line_2_x.append(proj_x[i, j])
                    nodal_line_2_y.append(proj_y[i, j])

    comp_x_all = []
    comp_y_all = []
    dila_x_all = []
    dila_y_all = []

    for i in range(n_az):
        for j in range(n_takeoff + 1):
            if amplitude[i, j] >= 0:
                comp_x_all.append(proj_x[i, j])
                comp_y_all.append(proj_y[i, j])
            else:
                dila_x_all.append(proj_x[i, j])
                dila_y_all.append(proj_y[i, j])

    circle_t = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(circle_t)
    circle_y = np.sin(circle_t)

    return {
        "compressive_x": comp_x_all,
        "compressive_y": comp_y_all,
        "dilatational_x": dila_x_all,
        "dilatational_y": dila_y_all,
        "circle_x": circle_x.tolist(),
        "circle_y": circle_y.tolist(),
        "nodal1_x": nodal_line_1_x,
        "nodal1_y": nodal_line_1_y,
        "nodal2_x": nodal_line_2_x,
        "nodal2_y": nodal_line_2_y,
    }


def generate_beachball_svg(
    strike: float, dip: float, rake: float,
    observations: Optional[List[dict]] = None,
    size: float = 200.0,
) -> str:
    contours = generate_beachball_contours(strike, dip, rake)

    cx = size / 2
    cy = size / 2
    r = size / 2 - 10

    svg_parts = []
    svg_parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">')

    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="black" stroke-width="2"/>')

    if contours["compressive_x"]:
        points = " ".join(
            f"{cx + x * r:.1f},{cy - y * r:.1f}"
            for x, y in zip(contours["compressive_x"], contours["compressive_y"])
        )
        svg_parts.append(f'<polygon points="{points}" fill="#E03030" fill-opacity="0.7" stroke="none"/>')

    if contours["dilatational_x"]:
        points = " ".join(
            f"{cx + x * r:.1f},{cy - y * r:.1f}"
            for x, y in zip(contours["dilatational_x"], contours["dilatational_y"])
        )
        svg_parts.append(f'<polygon points="{points}" fill="#3030E0" fill-opacity="0.7" stroke="none"/>')

    svg_parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="black" stroke-width="2"/>')

    if observations:
        for obs in observations:
            x_proj, y_proj = _beachball_projection(obs["azimuth"], obs["takeoff"])
            sx = cx + x_proj * r
            sy = cy - y_proj * r

            if obs["polarity"] > 0:
                svg_parts.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="black" stroke="white" stroke-width="1"/>'
                )
            else:
                svg_parts.append(
                    f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="white" stroke="black" stroke-width="1.5"/>'
                )

    fault_type = classify_fault_type(strike, dip, rake)
    svg_parts.append(f'<text x="{cx}" y="{size - 2}" text-anchor="middle" font-size="11" font-family="sans-serif">{fault_type}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)


def generate_beachball_plotly_3d(
    strike: float, dip: float, rake: float,
    observations: Optional[List[dict]] = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    center_z: float = 0.0,
    scale: float = 5.0,
    n_az: int = 36,
    n_takeoff: int = 18,
) -> Dict:
    M = _compute_moment_tensor(strike, dip, rake)

    azimuths = np.linspace(0, 2 * np.pi, n_az, endpoint=False)
    takeoffs = np.linspace(0, np.pi / 2, n_takeoff + 1)

    all_x = []
    all_y = []
    all_z = []
    all_val = []

    for az in azimuths:
        for take in takeoffs:
            x_sphere = np.sin(take) * np.sin(az) * scale + center_x
            y_sphere = np.sin(take) * np.cos(az) * scale + center_y
            z_sphere = np.cos(take) * scale + center_z

            r_vec = np.array([np.sin(take) * np.sin(az), np.sin(take) * np.cos(az), np.cos(take)])
            amp = r_vec @ M @ r_vec

            all_x.append(x_sphere)
            all_y.append(y_sphere)
            all_z.append(z_sphere)
            all_val.append(amp)

    obs_x, obs_y, obs_z, obs_pol = [], [], [], []
    if observations:
        for obs in observations:
            az_r = np.radians(obs["azimuth"])
            take_r = np.radians(obs["takeoff"])
            ox = np.sin(take_r) * np.sin(az_r) * scale + center_x
            oy = np.sin(take_r) * np.cos(az_r) * scale + center_y
            oz = np.cos(take_r) * scale + center_z
            obs_x.append(ox)
            obs_y.append(oy)
            obs_z.append(oz)
            obs_pol.append(obs["polarity"])

    return {
        "x": all_x,
        "y": all_y,
        "z": all_z,
        "value": all_val,
        "obs_x": obs_x,
        "obs_y": obs_y,
        "obs_z": obs_z,
        "obs_polarity": obs_pol,
        "strike": strike,
        "dip": dip,
        "rake": rake,
    }
