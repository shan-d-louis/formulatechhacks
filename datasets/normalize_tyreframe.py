"""Normalize SIDEWALL source exports into hierarchical TyreFrame tables.

The normalized outputs keep public F1 macro telemetry, simulator event windows, and
stint summaries at their natural sampling levels. This avoids leaking 64 Hz simulator
samples into lower-frequency public telemetry rows.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

try:
    from .utils import root_relative_path
except ImportError:
    from utils import root_relative_path


DEFAULT_FASTF1 = "outputs/telemetry_output.csv"
DEFAULT_OPENF1 = "outputs/openf1_output.csv"
DEFAULT_STINTS = "outputs/kaggle_tyre_strategy_output.csv"
DEFAULT_EVENT_WINDOWS = "datasets/spa/event_windows.csv"
DEFAULT_CANDIDATE_EVENTS = "datasets/spa/candidate_events.csv"
DEFAULT_OUT_DIR = "outputs/tyreframe"

WHEEL_POSITIONS = ("fl", "fr", "rl", "rr")
TYRE_SIGNAL_COLUMNS = [
    *(f"psi_{wheel}" for wheel in WHEEL_POSITIONS),
    *(f"tyre_core_{wheel}" for wheel in WHEEL_POSITIONS),
    *(f"tyre_surface_{wheel}" for wheel in WHEEL_POSITIONS),
    *(f"wheel_speed_{wheel}" for wheel in WHEEL_POSITIONS),
    *(f"slip_ratio_{wheel}" for wheel in WHEEL_POSITIONS),
    *(f"slip_proxy_{wheel}" for wheel in WHEEL_POSITIONS),
]

def snake_case(name: object) -> str:
    """Convert a source column label into lowercase snake_case."""
    text = str(name).strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    return text.strip("_").lower()


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe whose columns follow SIDEWALL lowercase snake_case style."""
    renamed = df.rename(columns={column: snake_case(column) for column in df.columns})
    return renamed.rename(
        columns={
            "speed": "speed_kmh",
            "driver_number": "driver_number",
            "lap_number": "lap_number",
            "tyre_life": "tyre_life",
            "track_temp": "track_temp_c",
            "air_temp": "air_temp_c",
            "rpm": "rpm",
        }
    )


def add_elapsed_seconds(
    df: pd.DataFrame,
    time_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add cumulative elapsed seconds from the start of each session-like group."""
    if time_col not in df:
        return df

    df = df.copy()
    group_cols = [col for col in (group_cols or []) if col in df.columns]
    parsed = parse_elapsed_series(df[time_col])
    if group_cols:
        starts = parsed.groupby([df[col] for col in group_cols], dropna=False).transform("min")
    else:
        starts = pd.Series(parsed.min(), index=df.index)
    df["elapsed_s"] = (parsed - starts).dt.total_seconds()
    return df


def parse_elapsed_series(values: pd.Series) -> pd.Series:
    """Parse datetime, timedelta, or numeric timestamp-like values into timedeltas."""
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.dropna().gt(1_000_000_000).any():
            numeric = numeric - numeric.min()
        return pd.to_timedelta(numeric, unit="s")

    text = values.astype("string")
    timedeltas = pd.to_timedelta(text, errors="coerce")
    missing = timedeltas.isna()
    if missing.any():
        datetimes = pd.to_datetime(text[missing], utc=True, format="mixed", errors="coerce")
        if datetimes.notna().any():
            timedeltas.loc[missing] = datetimes - datetimes.min()
    return timedeltas


def prepare_fastf1(path: str) -> pd.DataFrame:
    """Normalize public FastF1 macro telemetry without inventing tyre-only channels."""
    df = standardize_columns(pd.read_csv(root_relative_path(path)))
    df = add_elapsed_seconds(df, "time", ["year", "event", "session", "driver"])
    df["source"] = "fastf1"
    df["sample_level"] = "macro_telemetry"
    return add_unavailable_tyre_fields(df)


def prepare_openf1(path: str) -> pd.DataFrame:
    """Normalize public OpenF1 macro telemetry timestamps and signal names."""
    df = standardize_columns(pd.read_csv(root_relative_path(path)))
    df = add_elapsed_seconds(df, "date", ["year", "event", "session", "driver"])
    df["source"] = "openf1"
    df["sample_level"] = "macro_telemetry"
    return add_unavailable_tyre_fields(df)


def add_unavailable_tyre_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Flag public F1 tyre channels as unavailable inputs for twin estimation."""
    df = df.copy()
    for column in TYRE_SIGNAL_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    df["estimated_via_twin"] = True
    df["measurement_status"] = "public_f1_tyre_channels_unavailable"
    return df


def prepare_stints(path: str) -> pd.DataFrame:
    """Normalize lap/stint-level tyre strategy summaries."""
    df = standardize_columns(pd.read_csv(root_relative_path(path)))
    df = df.rename(columns={"stint_length": "stint_length_laps"})
    df["source"] = "kaggle_tyre_strategy"
    df["sample_level"] = "stint"
    df["estimated_via_twin"] = False
    df["measurement_status"] = "stint_summary"
    return df


def prepare_candidate_events(path: str) -> pd.DataFrame:
    """Normalize candidate event metadata and attach stable event identifiers."""
    df = standardize_columns(pd.read_csv(root_relative_path(path)))
    if "event_id" not in df:
        df.insert(0, "event_id", range(1, len(df) + 1))
    df["source"] = "simulator"
    df["sample_level"] = "event"
    df["estimated_via_twin"] = False
    df["measurement_status"] = "simulated_candidate_event"
    return df


def prepare_event_windows(path: str) -> pd.DataFrame:
    """Normalize simulator windows and deduplicate timestamps within each event."""
    df = standardize_columns(pd.read_csv(root_relative_path(path)))
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")
    df = df.sort_values(["event_id", "elapsed_s"])
    df = df.drop_duplicates(subset=["event_id", "elapsed_s"], keep="last")
    df = df[df["speed_kmh"].fillna(0) >= 10].copy()
    for column in [col for col in df.columns if col.startswith("slip_proxy_")]:
        df[column] = pd.to_numeric(df[column], errors="coerce").clip(-3.0, 3.0)
    df["source"] = "simulator"
    df["sample_level"] = "event_window"
    df["estimated_via_twin"] = False
    df["measurement_status"] = "simulated_measured"
    return df


def prepare_macro_laps(fastf1: pd.DataFrame, openf1: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Build lap-level macro features from public telemetry plus stint/weather context."""
    macro = pd.concat([fastf1, openf1], ignore_index=True, sort=False)
    group_cols = [
        col for col in ["source", "year", "event", "session", "driver", "lap_number"] if col in macro
    ]
    aggregations = {
        "elapsed_s": "min",
        "speed_kmh": "mean",
        "throttle": "mean",
        "brake": "mean",
        "rpm": "mean",
        "gear": "median",
        "air_temp_c": "mean",
        "track_temp_c": "mean",
        "tyre_life": "max",
        "compound": "last",
    }
    available_aggs = {col: agg for col, agg in aggregations.items() if col in macro}
    macro_laps = macro.groupby(group_cols, dropna=False).agg(available_aggs).reset_index()

    stint_cols = [
        col
        for col in ["year", "event", "session", "driver", "stint", "compound", "stint_length_laps"]
        if col in stints
    ]
    if stint_cols:
        context = stints[stint_cols].drop_duplicates()
        join_cols = [col for col in ["year", "event", "session", "driver", "compound"] if col in context and col in macro_laps]
        if join_cols:
            for col in join_cols:
                macro_laps[col] = macro_laps[col].astype("string")
                context[col] = context[col].astype("string")
            macro_laps = macro_laps.merge(context, on=join_cols, how="left", suffixes=("", "_stint"))

    macro_laps["sample_level"] = "macro_lap"
    macro_laps["estimated_via_twin"] = True
    macro_laps["measurement_status"] = "public_f1_macro_context"
    return macro_laps


def write_tyreframe_tables(
    fastf1_path: str = DEFAULT_FASTF1,
    openf1_path: str = DEFAULT_OPENF1,
    stints_path: str = DEFAULT_STINTS,
    event_windows_path: str = DEFAULT_EVENT_WINDOWS,
    candidate_events_path: str = DEFAULT_CANDIDATE_EVENTS,
    out_dir: str = DEFAULT_OUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Normalize source files and write hierarchical TyreFrame CSV outputs."""
    source_paths = {
        "fastf1": fastf1_path,
        "openf1": openf1_path,
        "stints": stints_path,
        "candidate_events": candidate_events_path,
        "event_windows": event_windows_path,
    }
    source_counts = {
        name: len(pd.read_csv(root_relative_path(path), usecols=[0]))
        for name, path in source_paths.items()
    }

    fastf1 = prepare_fastf1(fastf1_path)
    openf1 = prepare_openf1(openf1_path)
    stints = prepare_stints(stints_path)
    events = prepare_candidate_events(candidate_events_path)
    windows = prepare_event_windows(event_windows_path)
    macro_laps = prepare_macro_laps(fastf1, openf1, stints)

    tables = {
        "macro_telemetry": pd.concat([fastf1, openf1], ignore_index=True, sort=False),
        "macro_laps": macro_laps,
        "stints": stints,
        "candidate_events": events,
        "micro_event_windows": windows,
    }
    output_dir = root_relative_path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)

    manifest = build_manifest(tables, source_counts)
    manifest.to_csv(output_dir / "manifest.csv", index=False)
    return tables


def build_manifest(tables: dict[str, pd.DataFrame], source_counts: dict[str, int]) -> pd.DataFrame:
    """Build row-count metadata used to verify reproducible TyreFrame generation."""
    table_rows = [
        {
            "record_type": "output_table",
            "name": name,
            "rows": len(table),
            "sample_level": (
                table["sample_level"].iloc[0]
                if "sample_level" in table and len(table)
                else pd.NA
            ),
            "notes": "normalized TyreFrame table",
        }
        for name, table in tables.items()
    ]
    source_rows = [
        {
            "record_type": "source_table",
            "name": name,
            "rows": rows,
            "sample_level": pd.NA,
            "notes": "raw extracted/source rows before TyreFrame normalization",
        }
        for name, rows in source_counts.items()
    ]
    totals = [
        {
            "record_type": "total",
            "name": "source_records",
            "rows": sum(source_counts.values()),
            "sample_level": pd.NA,
            "notes": "sum of raw extracted/source rows",
        },
        {
            "record_type": "total",
            "name": "output_records",
            "rows": sum(len(table) for table in tables.values()),
            "sample_level": pd.NA,
            "notes": "sum of normalized output rows across hierarchical tables",
        },
    ]
    return pd.DataFrame([*source_rows, *table_rows, *totals])


def main() -> None:
    """Run the TyreFrame normalization command-line entry point."""
    parser = argparse.ArgumentParser(description="Normalize SIDEWALL data into TyreFrame tables.")
    parser.add_argument("--fastf1", default=DEFAULT_FASTF1)
    parser.add_argument("--openf1", default=DEFAULT_OPENF1)
    parser.add_argument("--stints", default=DEFAULT_STINTS)
    parser.add_argument("--event-windows", default=DEFAULT_EVENT_WINDOWS)
    parser.add_argument("--candidate-events", default=DEFAULT_CANDIDATE_EVENTS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    tables = write_tyreframe_tables(
        fastf1_path=args.fastf1,
        openf1_path=args.openf1,
        stints_path=args.stints,
        event_windows_path=args.event_windows,
        candidate_events_path=args.candidate_events,
        out_dir=args.out_dir,
    )
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    main()
