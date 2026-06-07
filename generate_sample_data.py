import numpy as np
from obspy import Stream, Trace, UTCDateTime
import os
import config


def _ricker_wavelet(npts, sampling_rate, f0, t0):
    t = np.arange(npts) / sampling_rate
    tau = t - t0
    w = (1 - 2 * (np.pi * f0 * tau) ** 2) * np.exp(-(np.pi * f0 * tau) ** 2)
    return w


def _earthquake_signal(npts, sampling_rate, arrival_sec, f0, duration_sec, amplitude):
    t = np.arange(npts) / sampling_rate
    signal = np.zeros(npts)
    onset_idx = int(arrival_sec * sampling_rate)
    if onset_idx >= npts:
        return signal

    n_onset = npts - onset_idx
    t_onset = np.arange(n_onset) / sampling_rate

    onset = _ricker_wavelet(n_onset, sampling_rate, f0, 0.03)
    coda_decay = np.exp(-t_onset / duration_sec)
    coda_osc = np.sin(2 * np.pi * f0 * 0.8 * t_onset + np.random.rand() * 2 * np.pi)
    coda = coda_decay * coda_osc * 0.4

    signal[onset_idx:] = (onset + coda) * amplitude
    return signal


def _generate_station_trace(
    sampling_rate: float,
    duration: float,
    p_arrival_sec: float,
    s_arrival_sec: float,
    snr: float = 8.0,
    f0: float = 5.0,
    channel: str = "BHZ",
) -> np.ndarray:
    npts = int(duration * sampling_rate)
    noise = np.random.randn(npts) * 0.1

    if channel[-1].upper() in ("Z", "1"):
        p_amp = snr * 0.15
        s_amp = snr * 0.06
    else:
        p_amp = snr * 0.06
        s_amp = snr * 0.15

    p_signal = _earthquake_signal(npts, sampling_rate, p_arrival_sec, f0, 3.0, p_amp)
    s_signal = _earthquake_signal(npts, sampling_rate, s_arrival_sec, f0 * 0.6, 5.0, s_amp)

    return noise + p_signal + s_signal


def generate_synthetic_station(
    network: str,
    station: str,
    sampling_rate: float,
    duration: float,
    p_arrival_sec: float,
    s_arrival_sec: float,
    starttime: UTCDateTime,
    snr: float = 5.0,
) -> Stream:
    st = Stream()
    for ch in ["BHZ", "BHN", "BHE"]:
        data = _generate_station_trace(
            sampling_rate, duration, p_arrival_sec, s_arrival_sec,
            snr=snr, channel=ch,
        )
        tr = Trace(data=data)
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.channel = ch
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = starttime
        st.append(tr)
    return st


def generate_synthetic_network(
    n_stations: int = 6,
    sampling_rate: float = 100.0,
    duration: float = 60.0,
    snr: float = 5.0,
) -> Stream:
    np.random.seed(12345)

    st = Stream()
    origin_time = UTCDateTime("2024-01-15T10:30:00.0")
    epicenter_dist_base = 20.0
    lta_buffer_sec = 12.0

    for i in range(n_stations):
        station_name = f"S{i + 1:02d}"
        dist_km = epicenter_dist_base + i * 10 + np.random.randn() * 3
        dist_km = max(10, dist_km)

        vp = 6.0
        vs = 3.46
        p_travel_time = dist_km / vp
        s_travel_time = dist_km / vs

        p_arrival = lta_buffer_sec + p_travel_time + np.random.randn() * 0.05
        s_arrival = lta_buffer_sec + s_travel_time + np.random.randn() * 0.05

        s_arrival = max(p_arrival + 2.0, s_arrival)

        station_st = generate_synthetic_station(
            network="SY",
            station=station_name,
            sampling_rate=sampling_rate,
            duration=duration,
            p_arrival_sec=p_arrival,
            s_arrival_sec=s_arrival,
            starttime=origin_time,
            snr=snr,
        )
        st += station_st

    return st


def generate_and_save_mseed(output_dir: str = None) -> str:
    if output_dir is None:
        output_dir = config.DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    st = generate_synthetic_network()
    filepath = os.path.join(output_dir, "synthetic_network.mseed")
    st.write(filepath, format="MSEED")

    return filepath


if __name__ == "__main__":
    filepath = generate_and_save_mseed()
    print(f"Synthetic MiniSEED data saved to: {filepath}")
