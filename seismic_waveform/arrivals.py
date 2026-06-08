import pandas as pd
from obspy import Stream
from typing import List, Optional
from seismic_waveform.picker import sta_lta_pick, recursive_sta_lta_pick
from seismic_tomography.focal_mechanism import compute_p_polarity


def build_arrivals_dataframe(
    st: Stream,
    method: str = "classic",
    sta_sec: float = None,
    lta_sec: float = None,
    p_threshold: float = None,
    s_threshold: float = None,
    extract_polarity: bool = False,
) -> pd.DataFrame:
    if method == "recursive":
        picks = recursive_sta_lta_pick(
            st, sta_sec=sta_sec, lta_sec=lta_sec,
            p_threshold=p_threshold, s_threshold=s_threshold,
        )
    else:
        picks = sta_lta_pick(
            st, sta_sec=sta_sec, lta_sec=lta_sec,
            p_threshold=p_threshold, s_threshold=s_threshold,
        )

    if extract_polarity and picks:
        picks = compute_p_polarity(st, picks)

    if not picks:
        cols = ["network", "station", "phase", "arrival_time", "arrival_time_str", "relative_delay_ms"]
        if extract_polarity:
            cols.append("polarity")
        return pd.DataFrame(columns=cols)

    p_times = [p["arrival_time"] for p in picks if p["phase"] == "P"]
    if p_times:
        reference_time = min(p_times)
    else:
        reference_time = min(p["arrival_time"] for p in picks)

    records = []
    for p in picks:
        relative_delay = (p["arrival_time"] - reference_time) * 1000.0
        rec = {
            "network": p["network"],
            "station": p["station"],
            "phase": p["phase"],
            "arrival_time": p["arrival_time"],
            "arrival_time_str": str(p["arrival_time"]),
            "relative_delay_ms": round(relative_delay, 2),
        }
        if extract_polarity:
            rec["polarity"] = p.get("polarity", 0)
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values(["phase", "relative_delay_ms"]).reset_index(drop=True)
    return df


def export_arrivals_csv(df: pd.DataFrame, filepath: str) -> None:
    df_export = df.copy()
    df_export["arrival_time_str"] = df_export["arrival_time"].astype(str)
    df_export = df_export.drop(columns=["arrival_time"])
    df_export.to_csv(filepath, index=False)


def compute_residuals(df: pd.DataFrame) -> pd.DataFrame:
    p_delays = df[df["phase"] == "P"]["relative_delay_ms"]
    if len(p_delays) == 0:
        return df

    mean_p_delay = p_delays.mean()
    df = df.copy()
    df["residual_ms"] = df.apply(
        lambda row: row["relative_delay_ms"] - mean_p_delay if row["phase"] == "P" else None,
        axis=1,
    )
    s_delays = df[df["phase"] == "S"]["relative_delay_ms"]
    if len(s_delays) > 0:
        mean_s_delay = s_delays.mean()
        df.loc[df["phase"] == "S", "residual_ms"] = (
            df.loc[df["phase"] == "S", "relative_delay_ms"] - mean_s_delay
        )
    return df
