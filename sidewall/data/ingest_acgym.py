"""Download MoTeC .ld laps from the Assetto Corsa Gym dataset (CC-BY-4.0) and convert them to parquet.

Only .ld files are fetched. The dataset also ships .pkl files, which we deliberately never load
(unpickling third-party files can execute arbitrary code).

    python -m sidewall.data.ingest_acgym --car dallara_f317
    python -m sidewall.data.ingest_acgym --car dallara_f317 --include-policy
"""
import argparse
import logging
import re
import urllib.parse

import pandas as pd
import requests

from sidewall import config
from sidewall.data.motec_ld import to_frame

log = logging.getLogger("ingest_acgym")

API = "https://huggingface.co/api/datasets/dasgringuen/assettoCorsaGym/tree/main/"
FILES = "https://huggingface.co/datasets/dasgringuen/assettoCorsaGym/resolve/main/"
OUT = config.PROCESSED / "acgym"
RAW = config.RAW / "acgym"

# Channels we keep, renamed to a compact schema.
KEEP = {
    "Ground Speed": "speed_kmh",
    "Engine RPM": "rpm",
    "Gear": "gear",
    "Throttle Pos": "throttle",
    "Brake Pos": "brake",
    "Steering Angle": "steer",
    "CG Accel Lateral": "g_lat",
    "CG Accel Longitudinal": "g_lon",
    "Car Coord X": "x",
    "Car Coord Y": "y",
    "Car Coord Z": "z",
    "Session Lap Count": "lap",
    "Car Pos Norm": "lap_progress",
    "Lap Invalidated": "lap_invalid",
    "Air Temp": "air_temp",
    "Road Temp": "road_temp",
}
for w in ("FL", "FR", "RL", "RR"):
    wl = w.lower()
    KEEP.update({
        f"Tire Slip Ratio {w}": f"kappa_{wl}",
        f"Tire Slip Angle {w}": f"alpha_{wl}",
        f"Wheel Angular Speed {w}": f"omega_{wl}",
        f"Tire Load {w}": f"load_{wl}",
        f"Tire Pressure {w}": f"psi_{wl}",
        f"Tire Temp Core {w}": f"t_core_{wl}",
        f"Tire Temp Inner {w}": f"t_in_{wl}",
        f"Tire Temp Middle {w}": f"t_mid_{wl}",
        f"Tire Temp Outer {w}": f"t_out_{wl}",
        f"Tire Rubber Grip {w}": f"grip_{wl}",
        f"Tire Loaded Radius {w}": f"radius_{wl}",
    })


def _list(path: str) -> list[dict]:
    r = requests.get(API + urllib.parse.quote(path), timeout=60)
    r.raise_for_status()
    return r.json()


def find_ld_files(car: str, include_policy: bool) -> list[str]:
    files = []
    for track in _list("data_sets"):
        if track["type"] != "directory":
            continue
        cars = {c["path"].rsplit("/", 1)[-1]: c["path"] for c in _list(track["path"]) if c["type"] == "directory"}
        if car not in cars:
            continue
        for sess in _list(cars[car]):
            if sess["type"] != "directory":
                continue
            is_policy = bool(re.search(r"SAC|policy", sess["path"], re.I))
            if is_policy and not include_policy:
                continue
            try:
                laps = _list(sess["path"] + "/laps")
            except requests.HTTPError:
                continue
            files += [f["path"] for f in laps if f["path"].endswith(".ld")]
    return files


def convert(ld_path, src: str) -> pd.DataFrame:
    df = to_frame(ld_path, hz=100)
    df = df[["t"] + [c for c in KEEP if c in df.columns]].rename(columns=KEEP)
    parts = src.split("/")
    df["track"], df["car"], df["session"] = parts[1], parts[2], parts[3]
    df["driver_id"] = parts[3].split("_", 1)[-1]
    # Slip ratio is logged in percent.
    for w in ("fl", "fr", "rl", "rr"):
        if f"kappa_{w}" in df:
            df[f"kappa_{w}"] = df[f"kappa_{w}"] / 100.0
    return df


def main(car: str, include_policy: bool, limit: int | None):
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    files = find_ld_files(car, include_policy)
    log.info("found %d .ld files for %s", len(files), car)
    for src in files[:limit]:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", src.replace("data_sets/", ""))
        out = OUT / (name[:-3] + ".parquet")
        if out.exists():
            continue
        local = RAW / name
        if not local.exists():
            with requests.get(FILES + urllib.parse.quote(src), stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(local, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        try:
            convert(local, src).to_parquet(out, compression="zstd")
            log.info("converted %s", name)
        except Exception as e:
            log.warning("failed %s: %s", name, e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--car", default="dallara_f317")
    ap.add_argument("--include-policy", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    main(a.car, a.include_policy, a.limit)
