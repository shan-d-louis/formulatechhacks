"""Download F1 race data with FastF1 and store it as per-race parquet files.

Resumable: races whose output already exists are skipped, so it can be re-run after
rate-limit errors or interruptions.

    python -m sidewall.data.ingest_fastf1                 # laps, race control, results (fast)
    python -m sidewall.data.ingest_fastf1 --telemetry     # also car + position data (slow, large)
    python -m sidewall.data.ingest_fastf1 --years 2020 2021
"""
import argparse
import logging
import time

import fastf1
import numpy as np
import pandas as pd
from fastf1.req import RateLimitExceededError

from sidewall import config

log = logging.getLogger("ingest_fastf1")

# FastF1 issues requests without a timeout, so one stalled connection can hang the whole download.
# Give every request a timeout; a stalled race then fails and is retried instead.
import requests  # noqa: E402

_orig_request = requests.Session.request


def _request_with_timeout(self, method, url, **kwargs):
    kwargs.setdefault("timeout", 60)
    return _orig_request(self, method, url, **kwargs)


requests.Session.request = _request_with_timeout


def race_key(year: int, rnd: int) -> str:
    return f"{year}_{rnd:02d}"


def _td_seconds(s: pd.Series) -> pd.Series:
    return s.dt.total_seconds()


def laps_frame(session, year: int, rnd: int) -> pd.DataFrame:
    laps = session.laps.copy()
    out = pd.DataFrame({
        "year": year,
        "round": rnd,
        "event": session.event["EventName"],
        "circuit": session.event["Location"],
        "driver": laps["Driver"],
        "driver_number": laps["DriverNumber"],
        "team": laps["Team"],
        "lap": laps["LapNumber"],
        "lap_time_s": _td_seconds(laps["LapTime"]),
        "s1_s": _td_seconds(laps["Sector1Time"]),
        "s2_s": _td_seconds(laps["Sector2Time"]),
        "s3_s": _td_seconds(laps["Sector3Time"]),
        "lap_start_s": _td_seconds(laps["LapStartTime"]),
        "lap_end_s": _td_seconds(laps["Time"]),
        "stint": laps["Stint"],
        "compound": laps["Compound"],
        "tyre_life": laps["TyreLife"],
        "fresh_tyre": laps["FreshTyre"],
        "pit_in": laps["PitInTime"].notna(),
        "pit_out": laps["PitOutTime"].notna(),
        "track_status": laps["TrackStatus"],
        "position": laps["Position"],
        "is_accurate": laps["IsAccurate"],
        "speed_i1": laps["SpeedI1"],
        "speed_i2": laps["SpeedI2"],
        "speed_fl": laps["SpeedFL"],
        "speed_st": laps["SpeedST"],
    })
    total_laps = out["lap"].max()
    out["total_laps"] = total_laps

    # Attach weather at each lap's end (nearest earlier sample).
    w = session.weather_data
    if w is not None and len(w):
        w = pd.DataFrame({
            "t": _td_seconds(w["Time"]),
            "air_temp": w["AirTemp"],
            "track_temp": w["TrackTemp"],
            "humidity": w["Humidity"],
            "rainfall": w["Rainfall"].astype(bool),
            "wind_speed": w["WindSpeed"],
        }).sort_values("t")
        out = out.sort_values("lap_end_s")
        has_t = out["lap_end_s"].notna()
        merged = pd.merge_asof(out[has_t], w, left_on="lap_end_s", right_on="t", direction="backward")
        out = pd.concat([merged.drop(columns="t"), out[~has_t]], ignore_index=True)
    return out.sort_values(["driver", "lap"]).reset_index(drop=True)


def race_control_frame(session, year: int, rnd: int) -> pd.DataFrame:
    rc = session.race_control_messages.copy()
    if rc is None or rc.empty:
        return pd.DataFrame()
    rc["year"], rc["round"] = year, rnd
    rc["Time"] = pd.to_datetime(rc["Time"])
    return rc


def results_frame(session, year: int, rnd: int) -> pd.DataFrame:
    r = session.results[["DriverNumber", "Abbreviation", "TeamName", "Position",
                         "GridPosition", "Status", "Points"]].copy()
    r["year"], r["round"] = year, rnd
    r["event"] = session.event["EventName"]
    r["circuit"] = session.event["Location"]
    return r


def telemetry_frame(session, year: int, rnd: int) -> pd.DataFrame:
    """Car data merged with position data for every driver, at the car-data rate."""
    frames = []
    for drv in session.drivers:
        try:
            car = session.car_data[drv]
            pos = session.pos_data[drv]
        except KeyError:
            continue
        car = car[["SessionTime", "Speed", "RPM", "nGear", "Throttle", "Brake", "DRS"]].copy()
        pos = pos[["SessionTime", "X", "Y", "Z", "Status"]].copy()
        car["t"] = _td_seconds(car["SessionTime"])
        pos["t"] = _td_seconds(pos["SessionTime"])
        car = car.drop(columns="SessionTime").sort_values("t")
        pos = pos.drop(columns="SessionTime").sort_values("t")
        # Interpolate position onto car-data timestamps.
        for c in ("X", "Y", "Z"):
            car[c] = np.interp(car["t"].values, pos["t"].values, pos[c].values.astype(float))
        car["off_track"] = pd.merge_asof(car[["t"]], pos[["t", "Status"]], on="t")["Status"].eq("OffTrack").values
        car["driver_number"] = drv
        frames.append(car)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["year"], out["round"] = year, rnd
    out["Brake"] = out["Brake"].astype(bool)
    return out


def _with_rate_limit(fn, *args, **kwargs):
    """FastF1 enforces 500 API calls/hour. Wait for the window to roll over instead of failing."""
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitExceededError:
            log.info("FastF1 rate limit reached; sleeping 10 min")
            time.sleep(600)


def ingest(years, telemetry: bool):
    fastf1.Cache.enable_cache(str(config.FASTF1_CACHE))
    for year in years:
        schedule = _with_rate_limit(fastf1.get_event_schedule, year, include_testing=False)
        for _, ev in schedule.iterrows():
            rnd = int(ev["RoundNumber"])
            key = race_key(year, rnd)
            laps_path = config.LAPS_DIR / f"{key}.parquet"
            tel_path = config.TELEM_DIR / f"{key}.parquet"
            need_laps = not laps_path.exists()
            need_tel = telemetry and not tel_path.exists()
            if not (need_laps or need_tel):
                continue
            if ev["EventDate"] > pd.Timestamp.now():
                continue
            for attempt in range(3):
                try:
                    session = _with_rate_limit(fastf1.get_session, year, rnd, "R")
                    _with_rate_limit(session.load, laps=True, telemetry=need_tel, weather=True, messages=True)
                    if need_laps:
                        laps_frame(session, year, rnd).to_parquet(laps_path)
                        rc = race_control_frame(session, year, rnd)
                        if not rc.empty:
                            rc.to_parquet(config.RC_DIR / f"{key}.parquet")
                        results_frame(session, year, rnd).to_parquet(config.RESULTS_DIR / f"{key}.parquet")
                    if need_tel:
                        tel = telemetry_frame(session, year, rnd)
                        if not tel.empty:
                            tel.to_parquet(tel_path, compression="zstd")
                    log.info("done %s %s", key, ev["EventName"])
                    break
                except Exception as e:  # missing data, network
                    log.warning("failed %s (%s) attempt %d: %s", key, ev["EventName"], attempt + 1, str(e)[:200])
                    time.sleep(5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=config.SEASONS)
    ap.add_argument("--telemetry", action="store_true")
    args = ap.parse_args()
    ingest(args.years, args.telemetry)
