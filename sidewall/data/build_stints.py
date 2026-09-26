"""Build the lap-level stint table used by the Tier B models (laps-to-cliff and failure hazard).

For every race, driver and stint:
- flag clean laps (green flag, not in/out laps, dry, valid time)
- fuel-correct lap times and fit the stint's degradation trend
- label the performance cliff (first sustained break above the trend); stints without one are censored
- label tyre failures: official retirements (Puncture/Tyre/Wheel) and suspected failures (lap-time spike
  followed by a pit stop or the end of the race)

    python -m sidewall.data.build_stints
"""
import glob
import logging

import numpy as np
import pandas as pd

from sidewall import config

log = logging.getLogger("build_stints")

FUEL_START_KG = 105.0
FUEL_S_PER_KG = 0.03
CLIFF_JUMP_S = 0.7         # sustained loss vs the stint's own trend
DRY = ["HYPERSOFT", "ULTRASOFT", "SUPERSOFT", "SOFT", "MEDIUM", "HARD", "SUPERHARD"]
FAIL_STATUS = {"Puncture", "Tyre", "Wheel", "Wheel nut"}
OUT = config.PROCESSED / "stint_laps.parquet"


def _load(dir_):
    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(dir_ / "*.parquet")))]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def _clean_mask(L: pd.DataFrame) -> pd.Series:
    ts = L["track_status"].fillna("").astype(str)
    neutral = ts.str.contains("[4567]")
    return (
        L["lap_time_s"].notna()
        & ~L["pit_in"] & ~L["pit_out"]
        & (L["lap"] > 1)
        & ~neutral
        & ~L["rainfall"].fillna(False).astype(bool)
        & L["compound"].isin(DRY)
    )


def _compound_rank(L: pd.DataFrame) -> pd.Series:
    """0 = softest dry compound used at this race, increasing with hardness."""
    order = {c: i for i, c in enumerate(DRY)}
    hard = L["compound"].map(order)
    rank = hard.groupby([L["year"], L["round"]]).rank(method="dense") - 1
    return rank


def _stint_trend(g: pd.DataFrame) -> pd.DataFrame:
    """Degradation trend, residuals and cliff label for one stint (rows sorted by lap)."""
    g = g.copy()
    clean = g["clean"].values
    life = g["tyre_life"].values.astype(float)
    t = g["t_corr"].values
    g["deg_slope"] = np.nan
    g["resid"] = np.nan
    g["cliff"] = False
    idx = np.flatnonzero(clean & ~np.isnan(t) & ~np.isnan(life))
    if len(idx) < 6:
        g["stint_usable"] = False
        return g
    g["stint_usable"] = True
    # Robust reference pace: median of the first clean laps of the stint.
    first = idx[:3]
    base = np.median(t[first])
    # Trend from the first ~half of clean laps (at least 4).
    k = max(4, len(idx) // 2)
    fit_idx = idx[:k]
    slope, icpt = np.polyfit(life[fit_idx], t[fit_idx], 1)
    # Reported slope is net of fuel but includes track evolution (rubbering-in can make it negative).
    g["deg_slope"] = slope
    # For cliff detection never extrapolate an improving trend.
    trend = icpt + max(slope, 0.0) * life
    resid = t - trend
    g["resid"] = resid
    g["deg_delta"] = t - base
    # Cliff: first clean lap after the fit window where this and the next clean lap both exceed the trend.
    for a, b in zip(idx[k - 1:-1], idx[k:]):
        if resid[a] > CLIFF_JUMP_S and resid[b] > CLIFF_JUMP_S:
            g.iloc[a, g.columns.get_loc("cliff")] = True
            break
    return g


def build() -> pd.DataFrame:
    L = _load(config.LAPS_DIR)
    R = _load(config.RESULTS_DIR)
    L = L.sort_values(["year", "round", "driver", "lap"]).reset_index(drop=True)
    L["clean"] = _clean_mask(L)
    L["compound_rank"] = _compound_rank(L)
    fuel = FUEL_START_KG * (1 - (L["lap"] - 1) / L["total_laps"].clip(lower=1))
    L["fuel_kg"] = fuel.clip(lower=0)
    L["t_corr"] = L["lap_time_s"] - FUEL_S_PER_KG * L["fuel_kg"]
    L["race_frac"] = L["lap"] / L["total_laps"]

    # Suspected tyre failure: a big lap-time spike under green flag, then pit or end of race.
    med = L[L["clean"]].groupby(["year", "round", "driver"])["lap_time_s"].median().rename("drv_med")
    L = L.join(med, on=["year", "round", "driver"])
    nxt_pit = L.groupby(["year", "round", "driver"])["pit_in"].shift(-1).fillna(False).astype(bool)
    last_lap = L["lap"] == L.groupby(["year", "round", "driver"])["lap"].transform("max")
    ts = L["track_status"].fillna("").astype(str)
    green = ~ts.str.contains("[4567]") & (L["lap"] > 2) & L["compound"].isin(DRY) & (L["tyre_life"] >= 5)
    # (a) Slow non-pit lap followed by a pit stop or the end of the race (e.g. Hamilton, Silverstone 2020 lap 52).
    spike = (L["lap_time_s"] > 1.15 * L["drv_med"]) & ~L["pit_in"] & (nxt_pit | last_lap)
    # (b) In-laps already include the pit-lane loss, so compare against other drivers' in-laps at this race.
    inlap_med = L[L["pit_in"] & green].groupby(["year", "round"])["lap_time_s"].median().rename("inlap_med")
    L = L.join(inlap_med, on=["year", "round"])
    limp = L["pit_in"] & (L["lap_time_s"] > 1.20 * L["inlap_med"])
    sus = green & (spike | limp)
    # Three or more drivers slow on the same lap is weather or a race-wide event, not a tyre.
    same_lap = sus.groupby([L["year"], L["round"], L["lap"]]).transform("sum")
    L["suspected_failure"] = sus & (same_lap < 3)

    # Official tyre-related retirements: mark the driver's last lap.
    fail = R[R["Status"].isin(FAIL_STATUS)][["year", "round", "Abbreviation"]].rename(columns={"Abbreviation": "driver"})
    fail["official_failure"] = True
    L = L.merge(fail, on=["year", "round", "driver"], how="left")
    L["official_failure"] = L["official_failure"].fillna(False).astype(bool) & last_lap
    L["failure"] = L["official_failure"] | L["suspected_failure"]

    L = L.groupby(["year", "round", "driver", "stint"], group_keys=False).apply(_stint_trend)
    log.info("laps %d, usable stints %d, cliffs %d, failures %d (official %d)",
             len(L), L.groupby(["year", "round", "driver", "stint"])["stint_usable"].first().sum(),
             int(L["cliff"].sum()), int(L["failure"].sum()), int(L["official_failure"].sum()))
    return L


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = build()
    df.to_parquet(OUT)
    print(df[df["failure"]][["year", "event", "driver", "lap", "compound", "tyre_life", "lap_time_s",
                             "drv_med", "official_failure"]].to_string())
