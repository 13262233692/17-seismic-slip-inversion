import numpy as np
from obspy import Stream
from scipy.signal import butter, sosfiltfilt
import config


def butterworth_bandpass(
    st: Stream,
    freqmin: float = None,
    freqmax: float = None,
    order: int = None,
) -> Stream:
    if freqmin is None:
        freqmin = config.BANDPASS_FREQMIN
    if freqmax is None:
        freqmax = config.BANDPASS_FREQMAX
    if order is None:
        order = config.BANDPASS_ORDER

    st_filtered = st.copy()
    for tr in st_filtered:
        fs = tr.stats.sampling_rate
        nyq = fs / 2.0
        low = freqmin / nyq
        high = freqmax / nyq
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        if low >= high:
            continue
        sos = butter(order, [low, high], btype="band", output="sos")
        tr.data = sosfiltfilt(sos, tr.data)
    return st_filtered


def butterworth_lowpass(st: Stream, freq: float, order: int = 4) -> Stream:
    st_filtered = st.copy()
    for tr in st_filtered:
        fs = tr.stats.sampling_rate
        nyq = fs / 2.0
        sos = butter(order, freq / nyq, btype="low", output="sos")
        tr.data = sosfiltfilt(sos, tr.data)
    return st_filtered


def butterworth_highpass(st: Stream, freq: float, order: int = 4) -> Stream:
    st_filtered = st.copy()
    for tr in st_filtered:
        fs = tr.stats.sampling_rate
        nyq = fs / 2.0
        sos = butter(order, freq / nyq, btype="high", output="sos")
        tr.data = sosfiltfilt(sos, tr.data)
    return st_filtered
