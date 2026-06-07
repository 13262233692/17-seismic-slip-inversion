import numpy as np
from obspy import Stream, UTCDateTime
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta, trigger_onset
import config
from typing import List, Optional, Tuple


def _pick_on_trace(
    data: np.ndarray,
    sampling_rate: float,
    sta_sec: float,
    lta_sec: float,
    threshold: float,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Optional[int]:
    if end_idx is None:
        end_idx = len(data)

    segment = data[start_idx:end_idx]
    if len(segment) < int(lta_sec * sampling_rate):
        return None

    nsta = int(sta_sec * sampling_rate)
    nlta = int(lta_sec * sampling_rate)

    cft = classic_sta_lta(segment, nsta, nlta)
    triggers = trigger_onset(cft, threshold, threshold - 1.0)

    if len(triggers) > 0:
        pick_local = triggers[0][0]
        return start_idx + pick_local
    return None


def _recursive_pick_on_trace(
    data: np.ndarray,
    sampling_rate: float,
    sta_sec: float,
    lta_sec: float,
    threshold: float,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
) -> Optional[int]:
    if end_idx is None:
        end_idx = len(data)

    segment = data[start_idx:end_idx]
    if len(segment) < int(lta_sec * sampling_rate):
        return None

    nsta = int(sta_sec * sampling_rate)
    nlta = int(lta_sec * sampling_rate)

    cft = recursive_sta_lta(segment, nsta, nlta)
    triggers = trigger_onset(cft, threshold, threshold - 1.0)

    if len(triggers) > 0:
        pick_local = triggers[0][0]
        return start_idx + pick_local
    return None


def sta_lta_pick(
    st: Stream,
    sta_sec: float = None,
    lta_sec: float = None,
    p_threshold: float = None,
    s_threshold: float = None,
    s_search_offset_sec: float = None,
    s_search_window_sec: float = None,
) -> List[dict]:
    if sta_sec is None:
        sta_sec = config.STA_WINDOW_SEC
    if lta_sec is None:
        lta_sec = config.LTA_WINDOW_SEC
    if p_threshold is None:
        p_threshold = config.P_THRESHOLD
    if s_threshold is None:
        s_threshold = config.S_THRESHOLD
    if s_search_offset_sec is None:
        s_search_offset_sec = config.S_SEARCH_OFFSET_SEC
    if s_search_window_sec is None:
        s_search_window_sec = config.S_SEARCH_WINDOW_SEC

    from seismic_waveform.reader import split_components
    components = split_components(st)
    picks = []

    stations = set()
    for tr in st:
        stations.add((tr.stats.network, tr.stats.station))

    for net, sta in stations:
        z_traces = [tr for tr in st.select(network=net, station=sta)
                    if tr.stats.channel and tr.stats.channel[-1].upper() in ("Z", "1")]
        h_traces = [tr for tr in st.select(network=net, station=sta)
                    if tr.stats.channel and tr.stats.channel[-1].upper() in ("N", "E", "2", "3")]

        p_pick_sample = None
        p_pick_time = None
        sampling_rate = None
        trace_starttime = None

        for tr in z_traces:
            data = tr.data.astype(np.float64)
            sampling_rate = tr.stats.sampling_rate
            trace_starttime = tr.stats.starttime
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                data = (data - mean_val) / std_val

            pick_idx = _pick_on_trace(data, sampling_rate, sta_sec, lta_sec, p_threshold)
            if pick_idx is not None:
                p_pick_sample = pick_idx
                p_pick_time = trace_starttime + pick_idx / sampling_rate
                picks.append({
                    "network": net,
                    "station": sta,
                    "phase": "P",
                    "arrival_time": p_pick_time,
                    "sample_index": pick_idx,
                })
                break

        if p_pick_time is not None and len(h_traces) > 0:
            p_offset_samples = int(s_search_offset_sec * sampling_rate)
            s_window_samples = int(s_search_window_sec * sampling_rate)
            s_sta = sta_sec
            s_lta = lta_sec * 0.5

            for tr in h_traces:
                data = tr.data.astype(np.float64)
                sampling_rate_h = tr.stats.sampling_rate
                trace_starttime_h = tr.stats.starttime
                mean_val = np.mean(data)
                std_val = np.std(data)
                if std_val > 0:
                    data = (data - mean_val) / std_val

                p_idx_in_h = int((p_pick_time - trace_starttime_h) * sampling_rate_h)
                search_start = max(0, p_idx_in_h + p_offset_samples)
                search_end = min(len(data), p_idx_in_h + s_window_samples)

                if search_start >= search_end:
                    continue

                pick_idx = _pick_on_trace(
                    data, sampling_rate_h, s_sta, s_lta, s_threshold,
                    start_idx=search_start, end_idx=search_end,
                )
                if pick_idx is not None:
                    s_pick_time = trace_starttime_h + pick_idx / sampling_rate_h
                    picks.append({
                        "network": net,
                        "station": sta,
                        "phase": "S",
                        "arrival_time": s_pick_time,
                        "sample_index": pick_idx,
                    })
                    break

    return picks


def recursive_sta_lta_pick(
    st: Stream,
    sta_sec: float = None,
    lta_sec: float = None,
    p_threshold: float = None,
    s_threshold: float = None,
    s_search_offset_sec: float = None,
    s_search_window_sec: float = None,
) -> List[dict]:
    if sta_sec is None:
        sta_sec = config.STA_WINDOW_SEC
    if lta_sec is None:
        lta_sec = config.LTA_WINDOW_SEC
    if p_threshold is None:
        p_threshold = config.P_THRESHOLD
    if s_threshold is None:
        s_threshold = config.S_THRESHOLD
    if s_search_offset_sec is None:
        s_search_offset_sec = config.S_SEARCH_OFFSET_SEC
    if s_search_window_sec is None:
        s_search_window_sec = config.S_SEARCH_WINDOW_SEC

    picks = []
    stations = set()
    for tr in st:
        stations.add((tr.stats.network, tr.stats.station))

    for net, sta in stations:
        z_traces = [tr for tr in st.select(network=net, station=sta)
                    if tr.stats.channel and tr.stats.channel[-1].upper() in ("Z", "1")]
        h_traces = [tr for tr in st.select(network=net, station=sta)
                    if tr.stats.channel and tr.stats.channel[-1].upper() in ("N", "E", "2", "3")]

        p_pick_time = None
        sampling_rate = None

        for tr in z_traces:
            data = tr.data.astype(np.float64)
            sampling_rate = tr.stats.sampling_rate
            trace_starttime = tr.stats.starttime
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val > 0:
                data = (data - mean_val) / std_val

            pick_idx = _recursive_pick_on_trace(data, sampling_rate, sta_sec, lta_sec, p_threshold)
            if pick_idx is not None:
                p_pick_time = trace_starttime + pick_idx / sampling_rate
                picks.append({
                    "network": net,
                    "station": sta,
                    "phase": "P",
                    "arrival_time": p_pick_time,
                    "sample_index": pick_idx,
                })
                break

        if p_pick_time is not None and len(h_traces) > 0:
            p_offset_samples = int(s_search_offset_sec * sampling_rate)
            s_window_samples = int(s_search_window_sec * sampling_rate)
            s_sta = sta_sec
            s_lta = lta_sec * 0.5

            for tr in h_traces:
                data = tr.data.astype(np.float64)
                sampling_rate_h = tr.stats.sampling_rate
                trace_starttime_h = tr.stats.starttime
                mean_val = np.mean(data)
                std_val = np.std(data)
                if std_val > 0:
                    data = (data - mean_val) / std_val

                p_idx_in_h = int((p_pick_time - trace_starttime_h) * sampling_rate_h)
                search_start = max(0, p_idx_in_h + p_offset_samples)
                search_end = min(len(data), p_idx_in_h + s_window_samples)

                if search_start >= search_end:
                    continue

                pick_idx = _recursive_pick_on_trace(
                    data, sampling_rate_h, s_sta, s_lta, s_threshold,
                    start_idx=search_start, end_idx=search_end,
                )
                if pick_idx is not None:
                    s_pick_time = trace_starttime_h + pick_idx / sampling_rate_h
                    picks.append({
                        "network": net,
                        "station": sta,
                        "phase": "S",
                        "arrival_time": s_pick_time,
                        "sample_index": pick_idx,
                    })
                    break

    return picks
