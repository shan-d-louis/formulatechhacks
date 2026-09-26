"""Replay a real race stint through the monitor ("Ghost of Silverstone 2020").

Everything is computed once, causally, and cached; the server then plays the frames back at any speed.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sidewall import config
from sidewall.data.build_stints import OUT as STINTS
from sidewall.engine.monitor import TyreMonitor
from sidewall.features import stream_from_fastf1
from sidewall.models.tyre_life import lap_predictions, prepare_laps

REPLAY_CACHE = config.PROCESSED / "replays"
REPLAY_CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class Scenario:
    key: str
    title: str
    year: int
    round: int
    driver: str
    from_lap: int
    to_lap: int
    what_happened: str
    # What-if only: inject a TPMS leak {wheel: (seconds after window start, gas fraction lost per minute)}.
    # Public F1 data has no pressure channel, so real replays never contain a leak unless one is injected.
    leaks: dict | None = None


SCENARIOS = {
    "silverstone2020": Scenario(
        "silverstone2020", "British GP 2020: Hamilton's front-left", 2020, 4, "HAM", 20, 52,
        "Front-left tyre failed on the final lap after a ~40-lap stint on hards; won on three wheels."),
    "baku2021": Scenario(
        "baku2021", "Azerbaijan GP 2021: Verstappen's left-rear", 2021, 6, "VER", 14, 46,
        "Left-rear failed at ~300 km/h on lap 46 while leading; Pirelli blamed low running pressures."),
    "silverstone2020_whatif": Scenario(
        "silverstone2020_whatif", "What-if: slow puncture injected (Silverstone 2020)", 2020, 4, "HAM", 20, 40,
        "Synthetic: a 0.8 %/min leak is injected on the front-left TPMS channel at lap ~24 to show the "
        "temperature-independent leak detector. Not real data.",
        leaks={"fl": (400.0, 0.008)}),
}


def _session_frames(year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"{year}_{rnd:02d}"
    tel = pd.read_parquet(config.TELEM_DIR / f"{key}.parquet")
    laps = pd.read_parquet(config.LAPS_DIR / f"{key}.parquet")
    return tel, laps


def build_replay(sc: Scenario, bundles: dict) -> dict:
    tel, laps = _session_frames(sc.year, sc.round)
    L = laps[laps["driver"] == sc.driver].sort_values("lap")
    num = str(L["driver_number"].iloc[0])
    t0 = L.loc[L["lap"] == sc.from_lap, "lap_start_s"].iloc[0]
    t1 = L.loc[L["lap"] == sc.to_lap, "lap_end_s"].iloc[0]
    # Include a few minutes before the window so the thermal-memory features are warmed up.
    tel = tel[(tel["driver_number"].astype(str) == num) & (tel["t"] >= t0 - 300) & (tel["t"] <= t1)]
    stream = stream_from_fastf1(tel.sort_values("t"))
    # Seconds since these tyres were fitted.
    stint = L.loc[L["lap"] == sc.from_lap, "stint"].iloc[0]
    fitted = L[(L["stint"] == stint)]["lap_start_s"].min()
    stream["tyre_age_s"] = (stream["t"] - fitted).clip(lower=0)

    # Tier B per-lap predictions for this driver's whole race (causal features), then keep the window.
    life = bundles.get("life")
    stints = pd.read_parquet(STINTS)
    S = stints[(stints["year"] == sc.year) & (stints["round"] == sc.round) & (stints["driver"] == sc.driver)]
    if life is not None and len(S):
        S = prepare_laps(S, life["priors"])
        S = S.join(lap_predictions(life, S))
    S = S.rename(columns={"lap_start_s": "lap_start_t", "lap_end_s": "lap_end_t"})
    S = S[(S["lap"] >= sc.from_lap) & (S["lap"] <= sc.to_lap)].sort_values("lap")

    leaks = {w: (t0 + start, rate) for w, (start, rate) in sc.leaks.items()} if sc.leaks else None
    mon = TyreMonitor(bundles)
    frames = mon.frame_outputs(stream, leaks=leaks)
    frames = frames[frames["t"] >= t0].reset_index(drop=True)
    fused = mon.fuse(frames, S)
    track = _track_outline(tel)
    return {"scenario": sc.__dict__, "track": track, "frames": fused,
            "laps": S[["lap", "tyre_life", "compound", "lap_time_s"] +
                      [c for c in ("safe_laps", "median_laps", "p_cliff_3", "failure_hazard", "cliff_hazard")
                       if c in S]].to_dict(orient="records")}


def _track_outline(tel: pd.DataFrame, n: int = 600) -> list[list[float]]:
    """A single clean lap of positions from the fastest-looking stretch, downsampled for the map."""
    one = tel[tel["driver_number"] == tel["driver_number"].iloc[0]].sort_values("t")
    one = one[(one["Speed"] > 60)]
    xy = one[["X", "Y"]].values / 10.0
    if len(xy) < 50:
        return []
    # Take roughly one lap: stop when we come back near the first point after travelling a while.
    d = np.hypot(*(xy - xy[0]).T)
    travelled = np.cumsum(np.r_[0, np.hypot(*np.diff(xy, axis=0).T)])
    back = np.flatnonzero((travelled > 2000) & (d < 30))
    end = back[0] if len(back) else len(xy)
    lap = xy[:end]
    idx = np.linspace(0, len(lap) - 1, min(n, len(lap))).astype(int)
    return lap[idx].round(1).tolist()


def get_replay(key: str, bundles: dict, rebuild: bool = False) -> dict:
    path = REPLAY_CACHE / f"{key}.json"
    if path.exists() and not rebuild:
        return json.loads(path.read_text())
    data = _sanitize(build_replay(SCENARIOS[key], bundles))
    path.write_text(json.dumps(data, default=_json_default, allow_nan=False))
    return data


def _sanitize(o):
    """Make a structure strictly JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(o, dict):
        return {str(k): _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
