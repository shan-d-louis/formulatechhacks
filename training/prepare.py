"""Clean the fetched laps, fuel-correct, and build the lap-time delta target.

Reads training/data/*_laps.parquet + *_weather.parquet (from fetch.py) and writes
training/data/train.parquet: one row per clean racing lap with its delta vs. the
stint baseline (median of stint laps 2-4) and the track temperature.

We model degradation as a lap-time delta rather than stint length, because stints
end for strategy reasons and their lengths are censored labels.

Usage:
    python prepare.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
import config as cfg  # noqa: E402  - thresholds shared with the live backend

DATA_DIR = HERE / "data"
OUT_PATH = DATA_DIR / "train.parquet"

OUT_COLUMNS = [
    "RaceId", "Year", "Round", "EventName", "Driver", "Team", "Stint", "Compound",
    "LapNumber", "TotalLaps", "StintLap", "TyreLife", "FreshTyre",
    "LapTime_s", "FuelCorrLapTime_s", "Baseline_s", "Delta_s", "TrackTemp", "AirTemp",
]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Concatenate every saved race's laps and weather tables."""
    lap_files = sorted(DATA_DIR.glob("*_laps.parquet"))
    weather_files = sorted(DATA_DIR.glob("*_weather.parquet"))
    if not lap_files:
        sys.exit(f"No *_laps.parquet files in {DATA_DIR}. Run fetch.py first.")
    laps = pd.concat([pd.read_parquet(f) for f in lap_files], ignore_index=True)
    weather = pd.concat([pd.read_parquet(f) for f in weather_files], ignore_index=True)
    for df in (laps, weather):
        df["RaceId"] = df["Year"].astype(str) + "_" + df["Round"].astype(int).map("{:02d}".format)
    print(f"Loaded {len(laps)} laps from {len(lap_files)} races")
    return laps, weather


def add_lap_context(laps: pd.DataFrame) -> pd.DataFrame:
    """Add race length and lap-in-stint, computed before any laps are dropped."""
    laps = laps.copy()
    laps["TotalLaps"] = laps.groupby("RaceId")["LapNumber"].transform("max")
    stint_start = laps.groupby(["RaceId", "Driver", "Stint"])["LapNumber"].transform("min")
    laps["StintLap"] = laps["LapNumber"] - stint_start + 1
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    return laps


def filter_laps(laps: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Drop non-representative laps. Returns kept laps and (reason, count) in the order applied."""
    status = laps["TrackStatus"].fillna("").astype(str)
    rules = [
        ("no lap time / stint / tyre age", laps["LapTime_s"].isna() | laps["Stint"].isna() | laps["TyreLife"].isna()),
        ("not a dry compound", ~laps["Compound"].isin(cfg.DRY_COMPOUNDS)),
        ("race lap 1 (standing start)", laps["LapNumber"] == 1),
        ("out-lap", laps["PitOutTime"].notna()),
        ("in-lap", laps["PitInTime"].notna()),
        ("safety car / VSC / red flag", status.apply(lambda s: any(c in s for c in cfg.NEUTRALISED_TRACK_STATUS))),
        ("IsAccurate false", ~laps["IsAccurate"].astype(bool)),
    ]
    keep = pd.Series(True, index=laps.index)
    report = []
    for reason, drop in rules:
        newly = keep & drop
        report.append((reason, int(newly.sum())))
        keep &= ~drop
    return laps[keep].copy(), report


def fuel_correct(laps: pd.DataFrame) -> pd.DataFrame:
    """Remove the fuel-load effect: subtract the time cost of the fuel still on board."""
    laps = laps.copy()
    fuel_left_kg = (laps["TotalLaps"] - laps["LapNumber"]) * cfg.FUEL_KG_PER_LAP
    laps["FuelCorrLapTime_s"] = laps["LapTime_s"] - cfg.FUEL_S_PER_KG * fuel_left_kg
    return laps


def add_delta(laps: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Delta vs. stint baseline (median of clean stint laps 2-4). Drops stints without a baseline."""
    lo, hi = cfg.BASELINE_LAPS
    keys = ["RaceId", "Driver", "Stint"]
    window = laps[laps["StintLap"].between(lo, hi)]
    base = window.groupby(keys)["FuelCorrLapTime_s"].agg(["median", "count"])
    base = base[base["count"] >= cfg.BASELINE_MIN_LAPS]["median"].rename("Baseline_s")
    out = laps.join(base, on=keys, how="inner")
    out["Delta_s"] = out["FuelCorrLapTime_s"] - out["Baseline_s"]
    return out, len(laps) - len(out)


def join_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach the nearest-in-time TrackTemp/AirTemp reading within the same race."""
    parts = []
    for race_id, race_laps in laps.groupby("RaceId"):
        w = weather.loc[weather["RaceId"] == race_id, ["Time", "TrackTemp", "AirTemp"]].sort_values("Time")
        parts.append(pd.merge_asof(race_laps.sort_values("Time"), w, on="Time", direction="nearest"))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    laps, weather = load_tables()
    total = len(laps)

    laps = add_lap_context(laps)
    laps, report = filter_laps(laps)
    laps = fuel_correct(laps)
    laps, no_baseline = add_delta(laps)
    report.append((f"stint has < {cfg.BASELINE_MIN_LAPS} clean baseline laps", no_baseline))
    outliers = laps["Delta_s"].abs() > cfg.DELTA_OUTLIER_S
    report.append((f"|delta| > {cfg.DELTA_OUTLIER_S:g} s (spin/traffic/yellow)", int(outliers.sum())))
    laps = laps[~outliers]
    stint_median = laps.groupby(["RaceId", "Driver", "Stint"])["Delta_s"].transform("median")
    bad_baseline = stint_median < cfg.STINT_MEDIAN_DELTA_FLOOR_S
    report.append((f"stint median delta < {cfg.STINT_MEDIAN_DELTA_FLOOR_S:g} s (baseline laps compromised)",
                   int(bad_baseline.sum())))
    laps = laps[~bad_baseline]

    train = join_weather(laps, weather)
    train["Stint"] = train["Stint"].astype(int)
    train["LapNumber"] = train["LapNumber"].astype(int)
    train["StintLap"] = train["StintLap"].astype(int)
    train = train[OUT_COLUMNS].sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True)
    train.to_parquet(OUT_PATH, index=False)

    kept = len(train)
    print(f"\nKept {kept} of {total} laps ({kept / total:.0%}), dropped {total - kept}:")
    for reason, n in report:
        print(f"  {n:>6}  {reason}")
    print(f"\nRaces: {train['RaceId'].nunique()}   stints: {train.groupby(['RaceId', 'Driver', 'Stint']).ngroups}")
    print("Laps per compound:", train["Compound"].value_counts().to_dict())
    print(f"Delta_s: mean {train['Delta_s'].mean():.2f}, median {train['Delta_s'].median():.2f}, "
          f"p90 {train['Delta_s'].quantile(0.9):.2f}")

    show = ["RaceId", "Driver", "Stint", "Compound", "LapNumber", "StintLap", "TyreLife",
            "LapTime_s", "FuelCorrLapTime_s", "Baseline_s", "Delta_s", "TrackTemp"]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print("\nExample rows (one stint):")
        first = train.groupby(["RaceId", "Driver", "Stint"]).ngroup() == 0
        print(train.loc[first, show].head(8).to_string(index=False))
        print("\nRandom sample:")
        print(train[show].sample(min(6, kept), random_state=0).to_string(index=False))
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
