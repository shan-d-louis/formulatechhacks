"""
extract_openf1.py

Extracts high-frequency car telemetry and driver interval/timing data from
the OpenF1 REST API (https://openf1.org) for the same "possible lock-up /
braking anomaly" review use case described in Tires_win.docx.

Why OpenF1 alongside FastF1:
    FastF1 (extract_fastf1.py) gives per-lap telemetry synced to lap/tyre
    context. OpenF1 complements it with:
      - car_data: raw high-frequency samples (~3.7 Hz) of Speed, Throttle,
        RPM, Brake, Gear, DRS, timestamped to the second.
      - intervals: gap-to-leader / interval-to-driver-ahead, sampled every
        ~4s, which can be diffed to spot a driver suddenly losing time
        (e.g. an acceleration drop out of a slow corner).

No API key/auth is required. Historical data is available from the 2023
season onward. Be mindful of the free-tier rate limit: 3 requests/sec,
30 requests/min (see openf1.org).

Install:
    pip install requests pandas
"""

import argparse
import time

import pandas as pd
import requests

try:
    from .utils import append_or_write
except ImportError:
    from utils import append_or_write

BASE_URL = "https://api.openf1.org/v1"

# Stay comfortably under the free tier's 3 req/s limit.
REQUEST_DELAY_SECONDS = 0.4


def parse_openf1_datetime(values: pd.Series) -> pd.Series:
    """
    Parse OpenF1 ISO timestamps with or without fractional seconds.

    Args:
        values: Series containing OpenF1 timestamp strings.

    Returns:
        UTC-normalized pandas datetime Series.

    Raises:
        ValueError: If pandas cannot parse a timestamp value.
    """
    return pd.to_datetime(values, utc=True, format="mixed")


def api_get(endpoint: str, **params) -> list[dict]:
    """GET an OpenF1 endpoint and return the parsed JSON list, with basic retry."""
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(3):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:
            time.sleep(2 * (attempt + 1))
            continue
        response.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return response.json()
    response.raise_for_status()
    return []


def find_session_key(year: int, event: str, session_type: str) -> int:
    """
    Resolve a session_key from year + event name + session type.

    session_type accepts either OpenF1's own session_name values
    ("Race", "Qualifying", "Sprint", "Practice 1", etc.) or FastF1-style
    shorthand ("R", "Q", "S", "FP1"/"FP2"/"FP3"), for consistency with
    extract_fastf1.py's --session flag.
    """
    shorthand_map = {
        "R": "Race",
        "Q": "Qualifying",
        "S": "Sprint",
        "FP1": "Practice 1",
        "FP2": "Practice 2",
        "FP3": "Practice 3",
    }
    session_name = shorthand_map.get(session_type.upper(), session_type)

    sessions = api_get(
        "sessions",
        year=year,
        session_name=session_name,
    )
    if not sessions:
        raise ValueError(
            f"No session found for year={year}, session_name='{session_name}'."
        )

    event_lower = event.lower()
    matches = [
        s for s in sessions
        if event_lower in str(s.get("location", "")).lower()
        or event_lower in str(s.get("country_name", "")).lower()
        or event_lower in str(s.get("circuit_short_name", "")).lower()
    ]
    if not matches:
        available = sorted({s.get("location") for s in sessions})
        raise ValueError(
            f"No '{session_name}' session matching event '{event}' in {year}. "
            f"Available locations: {available}"
        )

    return matches[0]["session_key"]


def get_driver_number(session_key: int, driver_code: str) -> int:
    """Resolve a 3-letter driver code (e.g. 'VER') to OpenF1's driver_number."""
    drivers = api_get("drivers", session_key=session_key, name_acronym=driver_code)
    if not drivers:
        raise ValueError(f"No driver '{driver_code}' found for session {session_key}.")
    return drivers[0]["driver_number"]


def extract_car_data(session_key: int, driver_number: int) -> pd.DataFrame:
    """
    High-frequency car telemetry: Speed, Throttle, RPM, Brake, Gear, DRS.
    Sampled at ~3.7 Hz, far denser than FastF1's per-lap-derived channel.
    """
    records = api_get("car_data", session_key=session_key, driver_number=driver_number)
    if not records:
        return pd.DataFrame(
            columns=["date", "driver_number", "speed", "throttle", "brake", "rpm", "n_gear", "drs"]
        )

    df = pd.DataFrame(records)
    df["date"] = parse_openf1_datetime(df["date"])
    df = df.rename(columns={"n_gear": "gear"})
    keep = ["date", "driver_number", "speed", "throttle", "brake", "rpm", "gear", "drs"]
    return df[[c for c in keep if c in df.columns]].sort_values("date")


def extract_intervals(session_key: int, driver_number: int) -> pd.DataFrame:
    """
    Driver-specific gap/interval timing (~4s cadence). A sudden jump in
    interval_to_leader or interval_to_driver_ahead relative to recent
    history can flag a micro-sector speed/acceleration loss, e.g. exiting
    a slow corner poorly.
    """
    records = api_get("intervals", session_key=session_key, driver_number=driver_number)
    if not records:
        return pd.DataFrame(
            columns=["date", "driver_number", "gap_to_leader", "interval"]
        )

    df = pd.DataFrame(records)
    df["date"] = parse_openf1_datetime(df["date"])
    keep = ["date", "driver_number", "gap_to_leader", "interval"]
    df = df[[c for c in keep if c in df.columns]].sort_values("date")

    # Delta vs. the previous sample highlights sudden losses of time
    # (a proxy for the "sudden acceleration drop out of a slow corner"
    # signal called for in the brief). Positive spikes = losing time fast.
    df["interval_delta"] = pd.to_numeric(df["interval"], errors="coerce").diff()
    return df


def merge_nearest(car_df: pd.DataFrame, intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Align the dense car_data stream with the sparser intervals stream by nearest time."""
    if car_df.empty or intervals_df.empty:
        merged = car_df.copy()
        for col in ("gap_to_leader", "interval", "interval_delta"):
            merged[col] = pd.NA
        return merged

    return pd.merge_asof(
        car_df.sort_values("date"),
        intervals_df.drop(columns=["driver_number"]).sort_values("date"),
        on="date",
        direction="nearest",
        tolerance=pd.Timedelta(3, unit="s"),
    )


BASE_COLUMNS = [
    "Driver",
    "driver_number",
    "date",
    "speed",
    "throttle",
    "brake",
    "rpm",
    "gear",
    "drs",
    "gap_to_leader",
    "interval",
    "interval_delta",
]

TELEMETRY_COLUMNS = ["Year", "Event", "Session", *BASE_COLUMNS]


def extract_driver(session_key: int, driver_code: str) -> pd.DataFrame:
    driver_number = get_driver_number(session_key, driver_code)

    car_df = extract_car_data(session_key, driver_number)
    intervals_df = extract_intervals(session_key, driver_number)

    merged = merge_nearest(car_df, intervals_df)
    merged["Driver"] = driver_code
    merged = merged.rename(columns={"driver_number": "driver_number"})
    return merged[[c for c in BASE_COLUMNS if c in merged.columns]]


def with_session_context(df: pd.DataFrame, year: int, event: str, session_type: str) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = year
    df["Event"] = event
    df["Session"] = session_type
    for col in TELEMETRY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[TELEMETRY_COLUMNS]


DEFAULT_SETS = [
    {
        "year": 2024,
        "event": "Monza",
        "session": "R",
        "drivers": ["LEC", "PIA"],
    },
    {
        "year": 2024,
        "event": "Las Vegas",
        "session": "R",
        "drivers": ["VER", "RUS"],
    },
]


def extract_sets(sets: list[dict], out: str) -> pd.DataFrame:
    frames = []
    for set_config in sets:
        year = set_config["year"]
        event = set_config["event"]
        session_type = set_config.get("session", "R")
        session_key = find_session_key(year, event, session_type)

        for driver in set_config["drivers"]:
            print(f"Extracting {year} {event} {session_type} OpenF1 data for {driver}")
            df = extract_driver(session_key, driver)
            frames.append(with_session_context(df, year, event, session_type))

    if not frames:
        raise ValueError("No telemetry sets were configured for extraction.")

    df = pd.concat(frames, ignore_index=True)
    append_or_write(df, out)
    return df


def main(sets: list[dict] | None = None):
    parser = argparse.ArgumentParser(description="Extract OpenF1 car_data + intervals fields.")
    parser.add_argument("--year", type=int, help="Season year, e.g. 2024")
    parser.add_argument("--event", type=str, help="Event name/location, e.g. 'Monza'")
    parser.add_argument(
        "--session", type=str, default="R",
        help="Session type: FP1/FP2/FP3/Q/S/R, or an OpenF1 session_name (default: R)",
    )
    parser.add_argument("--driver", type=str, help="Driver code, e.g. 'VER'")
    parser.add_argument(
        "--out", type=str, default="outputs/openf1_output.csv",
        help="Output CSV path (default: openf1_output.csv)",
    )
    args = parser.parse_args()

    single_args = [args.year, args.event, args.driver]
    if sets is not None and all(value is None for value in single_args):
        return extract_sets(sets, args.out)

    missing_args = [
        arg_name for arg_name in ("year", "event", "driver")
        if getattr(args, arg_name) is None
    ]
    if missing_args:
        parser.error(
            "the following arguments are required for single extraction: "
            + ", ".join(f"--{arg_name}" for arg_name in missing_args)
        )

    session_key = find_session_key(args.year, args.event, args.session)
    df = with_session_context(
        extract_driver(session_key, args.driver),
        args.year,
        args.event,
        args.session,
    )
    append_or_write(df, args.out)
    return df


# Example usage:
# python extract_openf1.py --year 2024 --event Monza --session R --driver VER
if __name__ == "__main__":
    main(DEFAULT_SETS)
