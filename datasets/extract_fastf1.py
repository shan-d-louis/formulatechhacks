"""
extract_fastf1.py

Extracts core telemetry + weather fields from a FastF1 session for use in
the "possible lock-up / braking anomaly" review dashboard described in
Tires_win.docx (Safety Diagnosis track).

Fields pulled:
    Speed                Instantaneous car velocity (km/h)
    Throttle             Throttle pedal % (0-100)
    Brake                Brake applied (bool)
    RPM                  Engine RPM
    Gear                 Gear number
    TrackTemp, AirTemp   Ambient/track conditions (from weather data)

Notes on data limits (see Tires_win.docx "Data limits"):
    FastF1's public telemetry does NOT include individual wheel speeds,
    fuel mass, tyre pressure, or tyre temperature. Anything derived from
    this script (e.g. a flagged braking event) should be labeled
    "possible lock-up" / "braking anomaly", never a confirmed failure.

Install:
    pip install fastf1 pandas
"""

import argparse
from pathlib import Path

import fastf1
from fastf1.exceptions import DataNotLoadedError
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def root_relative_path(path: str) -> Path:
    out_path = Path(path)
    if out_path.is_absolute():
        return out_path
    return PROJECT_ROOT / out_path


def load_session(year: int, event: str, session_type: str) -> fastf1.core.Session:
    """Load and process a FastF1 session (enables local caching)."""
    cache_dir = Path(".fastf1_cache")
    cache_dir.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(year, event, session_type)
    session.load(telemetry=True, weather=True, laps=True)
    return session


def extract_driver_telemetry(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    """
    Build a merged telemetry + weather dataframe for one driver across
    the whole session, at car-telemetry sample resolution.
    """
    try:
        laps = session.laps.pick_drivers(driver)
    except DataNotLoadedError:
        return extract_driver_car_data(session, driver)

    if laps.empty:
        raise ValueError(f"No laps found for driver '{driver}' in this session.")

    frames = []
    for _, lap in laps.iterlaps():
        tel = lap.get_car_data().add_distance()
        tel = tel.rename(columns={"nGear": "Gear"})

        tel["Driver"] = driver
        tel["DriverNumber"] = str(lap["DriverNumber"])
        tel["LapNumber"] = lap["LapNumber"]
        tel["Compound"] = lap["Compound"]
        tel["TyreLife"] = lap["TyreLife"]  # laps completed on current tyre set

        frames.append(tel)

    telemetry = pd.concat(frames, ignore_index=True)
    return merge_weather_data(telemetry, session)[BASE_TELEMETRY_COLUMNS]


def extract_driver_car_data(session: fastf1.core.Session, driver: str) -> pd.DataFrame:
    """
    Build telemetry directly from car_data when timing/lap data is unavailable.

    Some FastF1 sessions can load car/weather data while failing to process timing
    data. In that state, session.laps raises DataNotLoadedError, but the core
    telemetry channels are still useful for diagnosis.
    """
    driver_result = session.get_driver(driver)
    driver_number = str(driver_result["DriverNumber"])

    try:
        telemetry = session.car_data[driver_number].copy()
    except KeyError as exc:
        raise ValueError(
            f"No car telemetry found for driver '{driver}' "
            f"(racing number {driver_number})."
        ) from exc

    telemetry = telemetry.rename(columns={"nGear": "Gear"})
    telemetry["Driver"] = driver
    telemetry["DriverNumber"] = driver_number
    telemetry["LapNumber"] = pd.NA
    telemetry["Compound"] = pd.NA
    telemetry["TyreLife"] = pd.NA

    if "Distance" not in telemetry.columns:
        telemetry["Distance"] = pd.NA

    return merge_weather_data(telemetry, session)[BASE_TELEMETRY_COLUMNS]


def merge_weather_data(telemetry: pd.DataFrame, session: fastf1.core.Session) -> pd.DataFrame:

    # Weather channel is sampled independently (~every few sec) -> merge nearest-in-time
    telemetry = telemetry.copy()
    weather = session.weather_data[["Time", "AirTemp", "TrackTemp", "Rainfall"]].copy()
    telemetry["Time"] = pd.to_timedelta(telemetry["Time"]).astype("timedelta64[ns]")
    weather["Time"] = pd.to_timedelta(weather["Time"]).astype("timedelta64[ns]")

    return pd.merge_asof(
        telemetry.sort_values("Time"),
        weather.sort_values("Time"),
        on="Time",
        direction="nearest",
    )


BASE_TELEMETRY_COLUMNS = [
    "Driver",
    "DriverNumber",
    "Time",
    "Distance",
    "LapNumber",
    "Speed",
    "Throttle",
    "Brake",
    "RPM",
    "Gear",
    "Compound",
    "TyreLife",
    "TrackTemp",
    "AirTemp",
    "Rainfall",
]


TELEMETRY_COLUMNS = [
    "Year",
    "Event",
    "Session",
    *BASE_TELEMETRY_COLUMNS,
]


DEFAULT_SETS = [
    {
        "year": 2024,
        "event": "Monza",
        "session": "R",
        "drivers": ["LEC", "PIA"],  # LEC (Charles Leclerc) or PIA (Oscar Piastri)
    },
    {
        "year": 2024,
        "event": "Las Vegas",
        "session": "R",
        "drivers": ["VER", "RUS"],  # VER (Max Verstappen) or RUS (George Russell)
    },
]


def with_session_context(
    df: pd.DataFrame,
    year: int,
    event: str,
    session_type: str,
) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = year
    df["Event"] = event
    df["Session"] = session_type
    return df[TELEMETRY_COLUMNS]


def append_or_write(df: pd.DataFrame, out: str) -> None:
    out_path = root_relative_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Appending to an existing file: {out_path}")
        existing_df = pd.read_csv(out_path)
        df = pd.concat([existing_df, df], ignore_index=True).drop_duplicates(
            subset=["Year", "Event", "Session", "Driver", "LapNumber", "Time"],
            keep="last",
        )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(
        df.groupby(["Year", "Event", "Session", "Driver", "DriverNumber"])
        .size()
        .rename("Rows")
        .reset_index()
        .to_string(index=False)
    )


def extract_sets(sets: list[dict], out: str) -> pd.DataFrame:
    frames = []
    for set_config in sets:
        year = set_config["year"]
        event = set_config["event"]
        session_type = set_config.get("session", "R")
        session = load_session(year, event, session_type)

        for driver in set_config["drivers"]:
            print(f"Extracting {year} {event} {session_type} telemetry for {driver}")
            df = extract_driver_telemetry(session, driver)
            frames.append(with_session_context(df, year, event, session_type))

    if not frames:
        raise ValueError("No telemetry sets were configured for extraction.")

    df = pd.concat(frames, ignore_index=True)
    append_or_write(df, out)
    return df


def main(sets: list[dict] | None = None):
    parser = argparse.ArgumentParser(description="Extract FastF1 telemetry + weather fields.")
    parser.add_argument("--year", type=int, help="Season year, e.g. 2024")
    parser.add_argument("--event", type=str, help="Event name or round, e.g. 'Monza'")
    parser.add_argument(
        "--session", type=str, default="R",
        help="Session type: FP1/FP2/FP3/Q/S/R (default: R = race)",
    )
    parser.add_argument("--driver", type=str, help="Driver code, e.g. 'VER'")
    parser.add_argument(
        "--out", type=str, default="outputs/telemetry_output.csv",
        help="Output CSV path (default: telemetry_output.csv)",
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

    session = load_session(args.year, args.event, args.session)
    df = with_session_context(
        extract_driver_telemetry(session, args.driver),
        args.year,
        args.event,
        args.session,
    )
    append_or_write(df, args.out)
    return df

# Example usage:
# python extract_telemetry.py --year 2024 --event Monza --session R --driver VER
if __name__ == "__main__":
    main(DEFAULT_SETS)
