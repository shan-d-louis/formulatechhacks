"""Download FastF1 race laps + weather for every dry race and save as parquet.

Offline step, runs before the demo. Cache lives in training/cache so reruns are fast,
and races already saved in training/data are skipped.

Usage:
    python fetch.py                  # 2023 and 2024
    python fetch.py --years 2024     # one season
"""

import argparse
import logging
import re
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import RateLimitExceededError

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
DATA_DIR = HERE / "data"

DEFAULT_YEARS = [2023, 2024]
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}


def slugify(name: str) -> str:
    """'São Paulo Grand Prix' -> 'sao_paulo_grand_prix' (ASCII-ish, filesystem safe)."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower().encode("ascii", "ignore").decode()).strip("_")


def race_paths(year: int, rnd: int, event_name: str) -> tuple[Path, Path]:
    """Output parquet paths for one race's laps and weather tables."""
    stem = f"{year}_{rnd:02d}_{slugify(event_name)}"
    return DATA_DIR / f"{stem}_laps.parquet", DATA_DIR / f"{stem}_weather.parquet"


def wet_reason(laps: pd.DataFrame, weather: pd.DataFrame) -> str | None:
    """Return why a race counts as wet, or None if it was dry."""
    if "Rainfall" in weather and weather["Rainfall"].fillna(False).astype(bool).any():
        return "rainfall recorded"
    used = set(laps["Compound"].dropna().str.upper())
    if used & WET_COMPOUNDS:
        return f"wet compounds used ({', '.join(sorted(used & WET_COMPOUNDS))})"
    return None


def load_race(year: int, rnd: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one race's laps and weather. Raises if either table is missing."""
    session = fastf1.get_session(year, rnd, "R")
    # Telemetry is the slow part and we don't need it; track status comes with laps.
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = pd.DataFrame(session.laps)  # raises DataNotLoadedError if timing failed
    weather = pd.DataFrame(session.weather_data)
    if laps.empty:
        raise ValueError("laps table is empty")
    if weather.empty:
        raise ValueError("weather table is empty")
    return laps, weather


def with_context(df: pd.DataFrame, year: int, rnd: int, event_name: str) -> pd.DataFrame:
    """Tag a table with the race it came from so prepare.py can concat and group."""
    df = df.copy()
    df.insert(0, "Year", year)
    df.insert(1, "Round", rnd)
    df.insert(2, "EventName", event_name)
    return df


def fetch(years: list[int]) -> None:
    """Download and save every dry race for the given seasons."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level(logging.WARNING)  # FastF1's INFO logs drown out our progress lines

    races = []
    for year in years:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in schedule.iterrows():
            races.append((year, int(ev["RoundNumber"]), str(ev["EventName"])))

    saved, skipped_existing, wet, failed = [], [], [], []
    start = time.time()
    total = len(races)
    print(f"Fetching {total} races for {years} into {DATA_DIR}")

    for i, (year, rnd, name) in enumerate(races, 1):
        label = f"{year} R{rnd:02d} {name}"
        prefix = f"[{i:>2}/{total}] {label}"
        laps_path, weather_path = race_paths(year, rnd, name)

        if laps_path.exists() and weather_path.exists():
            print(f"{prefix}: already saved, skipping")
            skipped_existing.append(label)
            continue

        t0 = time.time()
        print(f"{prefix}: loading...", flush=True)
        try:
            laps, weather = load_race(year, rnd)
        except RateLimitExceededError as exc:
            # Not the race's fault: stop now, rerun later and it resumes where it left off.
            print(f"{prefix}: RATE LIMITED by FastF1 API ({exc}). Stopping; rerun later to resume.")
            failed.append(f"{label} (rate limited, run stopped)")
            break
        except Exception as exc:  # noqa: BLE001 - any load failure means skip this race
            print(f"{prefix}: FAILED ({type(exc).__name__}: {exc})")
            failed.append(f"{label} ({type(exc).__name__}: {exc})")
            continue

        reason = wet_reason(laps, weather)
        if reason:
            print(f"{prefix}: wet race, not saved ({reason}) [{time.time() - t0:.0f}s]")
            wet.append(f"{label} ({reason})")
            continue

        try:
            with_context(laps, year, rnd, name).to_parquet(laps_path, index=False)
            with_context(weather, year, rnd, name).to_parquet(weather_path, index=False)
        except Exception as exc:  # noqa: BLE001
            laps_path.unlink(missing_ok=True)
            weather_path.unlink(missing_ok=True)
            print(f"{prefix}: FAILED to save parquet ({type(exc).__name__}: {exc})")
            failed.append(f"{label} (save error: {exc})")
            continue

        elapsed = time.time() - start
        print(f"{prefix}: saved {len(laps)} laps, {len(weather)} weather rows "
              f"[{time.time() - t0:.0f}s, {elapsed / 60:.1f} min total]")
        saved.append(label)

    print()
    print(f"Done in {(time.time() - start) / 60:.1f} min.")
    print(f"  saved:           {len(saved)}")
    print(f"  already present: {len(skipped_existing)}")
    print(f"  wet (skipped):   {len(wet)}")
    for w in wet:
        print(f"    - {w}")
    print(f"  failed:          {len(failed)}")
    for f in failed:
        print(f"    - {f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()
    fetch(args.years)


if __name__ == "__main__":
    main()
