import os
from obspy import read, Stream, UTCDateTime
from typing import Optional, List


def read_mseed(filepath: str) -> Stream:
    st = read(filepath)
    return st


def scan_directory(directory: str, pattern: str = "*.mseed") -> Stream:
    st = Stream()
    for f in sorted(os.listdir(directory)):
        if f.endswith(".mseed") or f.endswith(".ms"):
            filepath = os.path.join(directory, f)
            st += read(filepath)
    return st


def get_station_metadata(st: Stream) -> List[dict]:
    metadata = []
    seen = set()
    for tr in st:
        key = (tr.stats.network, tr.stats.station, tr.stats.channel)
        if key not in seen:
            seen.add(key)
            metadata.append({
                "network": tr.stats.network,
                "station": tr.stats.station,
                "channel": tr.stats.channel,
                "sampling_rate": tr.stats.sampling_rate,
                "starttime": tr.stats.starttime,
                "endtime": tr.stats.endtime,
                "npts": tr.stats.npts,
            })
    return metadata


def split_components(st: Stream) -> dict:
    components = {"Z": Stream(), "N": Stream(), "E": Stream(), "1": Stream(), "2": Stream()}
    for tr in st:
        ch = tr.stats.channel
        if ch and ch[-1].upper() in ("Z", "1"):
            components["Z"].append(tr)
        elif ch and ch[-1].upper() in ("N", "2"):
            components["N"].append(tr)
        elif ch and ch[-1].upper() in ("E", "3"):
            components["E"].append(tr)
    for key in list(components.keys()):
        if key not in ("Z", "N", "E") and len(components[key]) == 0:
            del components[key]
    return {k: v for k, v in components.items() if len(v) > 0}
