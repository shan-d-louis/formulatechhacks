"""
extract_kaggle_tyre_strategy.py

Extracts stint-level tyre strategy features from community Kaggle F1 strategy
datasets, including F1-Tyre-Strategy-Engine style CSV exports.

Expected derived fields:
    Compound      Hard, Medium, Soft, Intermediate, or Wet.
    StintLength   Laps completed on a specific tyre set.
    AirTemp       Stint-level aggregated air temperature.
    TrackTemp     Stint-level aggregated track temperature.

Kaggle datasets are community-maintained and do not share one stable schema. This
adapter accepts either a downloaded CSV/directory or a Kaggle dataset slug, then maps
common column aliases into SIDEWALL's explicit stint contract.

Install Kaggle CLI only if you want automatic downloads:
    pip install kaggle

Example usage:
    python datasets/extract_kaggle_tyre_strategy.py --input data/raw/f1_strategy
    python datasets/extract_kaggle_tyre_strategy.py --kaggle-dataset owner/dataset-slug
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import zipfile

import pandas as pd

try:
    from .utils import root_relative_path
except ImportError:
    from utils import root_relative_path


COMPOUND_MAP = {
    "C1": "Hard",
    "C2": "Medium",
    "C3": "Soft",
    "C4": "Soft",
    "C5": "Soft",
    "H": "Hard",
    "HARD": "Hard",
    "M": "Medium",
    "MEDIUM": "Medium",
    "S": "Soft",
    "SOFT": "Soft",
    "I": "Intermediate",
    "INTER": "Intermediate",
    "INTERMEDIATE": "Intermediate",
    "W": "Wet",
    "WET": "Wet",
}
ALLOWED_COMPOUNDS = {"Hard", "Medium", "Soft", "Intermediate", "Wet"}

COLUMN_ALIASES = {
    "Compound": [
        "Compound",
        "compound",
        "Tyre",
        "Tire",
        "TyreCompound",
        "TireCompound",
        "tyre_compound",
        "tire_compound",
    ],
    "StintLength": [
        "StintLength",
        "stint_length",
        "stintLength",
        "StintLen",
        "stint_laps",
        "StintLaps",
        "laps_in_stint",
        "TyreLife",
        "tire_age",
        "tyre_age",
    ],
    "AirTemp": ["AirTemp", "air_temp", "air_temperature", "AvgAirTemp", "mean_air_temp"],
    "TrackTemp": [
        "TrackTemp",
        "track_temp",
        "track_temperature",
        "AvgTrackTemp",
        "mean_track_temp",
    ],
    "Year": ["Year", "year", "Season", "season"],
    "Event": ["Event", "event", "GrandPrix", "grand_prix", "Race", "race"],
    "Session": ["Session", "session"],
    "Driver": ["Driver", "driver", "DriverCode", "driver_code"],
    "Stint": ["Stint", "stint", "StintNumber", "stint_number"],
    "LapNumber": ["LapNumber", "lap_number", "Lap", "lap"],
}

OUTPUT_COLUMNS = [
    "Dataset",
    "SourceFile",
    "Year",
    "Event",
    "Session",
    "Driver",
    "Stint",
    "Compound",
    "StintLength",
    "AirTemp",
    "TrackTemp",
]

DATASET_SLUG_ALIASES = {
    "navenkumar1998/formula-1-dataset-with-weather-and-tyre-feature": (
        "navenkumar1998/formula-1-dataset-with-weather-and-tyre-features"
    ),
}


def first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    """Return the first dataframe column matching one of the supplied aliases."""
    lookup = {col.lower(): col for col in df.columns}
    for alias in aliases:
        match = lookup.get(alias.lower())
        if match is not None:
            return match
    return None


def normalize_compound(value: object) -> object:
    """Normalize common F1 tyre compound spellings into display labels."""
    if pd.isna(value):
        return pd.NA
    key = str(value).strip().upper().replace(" ", "_")
    key = key.removeprefix("COMPOUND_")
    return COMPOUND_MAP.get(key, str(value).strip().title())


SUPPORTED_DATA_EXTENSIONS = {".csv", ".parquet"}


def find_supported_data_files(input_path: Path) -> list[Path]:
    """Return supported local data files without raising if none exist."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS else []
    return sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DATA_EXTENSIONS
    )


def supported_data_files(input_path: Path) -> list[Path]:
    """Return supported local data files under a file or directory path."""
    paths = find_supported_data_files(input_path)
    if not paths:
        raise ValueError(f"No CSV or Parquet files found under {input_path}")
    return paths


def read_tabular_files(input_path: Path) -> list[tuple[Path, pd.DataFrame]]:
    """Read supported CSV or Parquet files under a file or directory path."""
    frames = []
    for path in supported_data_files(input_path):
        if path.suffix.lower() == ".csv":
            frames.append((path, pd.read_csv(path)))
        elif path.suffix.lower() == ".parquet":
            try:
                frames.append((path, pd.read_parquet(path)))
            except ImportError as exc:
                raise RuntimeError(
                    f"{path} is a Parquet file, but pandas cannot read Parquet without "
                    "`pyarrow` or `fastparquet`. Install one of those engines, or export "
                    "the Kaggle file to CSV and run with `--input path/to/csvs`."
                ) from exc
    return frames


def standardize_frame(df: pd.DataFrame, source_file: Path, dataset_name: str) -> pd.DataFrame:
    """
    Map a Kaggle strategy CSV into the SIDEWALL stint-summary contract.

    Args:
        df: Raw CSV dataframe.
        source_file: CSV path, used for provenance.
        dataset_name: Human-readable dataset label.

    Returns:
        Dataframe with OUTPUT_COLUMNS.

    Raises:
        ValueError: If required compound, stint length, or temperature columns cannot
            be mapped or derived.
    """
    mapped = pd.DataFrame()
    for output_col, aliases in COLUMN_ALIASES.items():
        source_col = first_existing_column(df, aliases)
        if source_col is not None:
            mapped[output_col] = df[source_col]

    if "Compound" not in mapped:
        raise ValueError(f"{source_file} does not include a recognizable compound column.")
    mapped["Compound"] = mapped["Compound"].map(normalize_compound)

    if "StintLength" not in mapped:
        mapped["StintLength"] = derive_stint_length(df, mapped, source_file)

    for required_col in ("AirTemp", "TrackTemp"):
        if required_col not in mapped:
            raise ValueError(
                f"{source_file} does not include a recognizable {required_col} column."
            )

    mapped["StintLength"] = pd.to_numeric(mapped["StintLength"], errors="coerce")
    mapped["AirTemp"] = pd.to_numeric(mapped["AirTemp"], errors="coerce")
    mapped["TrackTemp"] = pd.to_numeric(mapped["TrackTemp"], errors="coerce")
    mapped["Dataset"] = dataset_name
    project_root = root_relative_path(".").resolve()
    try:
        mapped["SourceFile"] = str(source_file.resolve().relative_to(project_root))
    except ValueError:
        mapped["SourceFile"] = str(source_file)

    group_cols = [
        col
        for col in ("Dataset", "SourceFile", "Year", "Event", "Session", "Driver", "Stint", "Compound")
        if col in mapped.columns
    ]
    summary = (
        mapped.groupby(group_cols, dropna=False)
        .agg(
            StintLength=("StintLength", "max"),
            AirTemp=("AirTemp", "mean"),
            TrackTemp=("TrackTemp", "mean"),
        )
        .reset_index()
    )

    for col in OUTPUT_COLUMNS:
        if col not in summary:
            summary[col] = pd.NA

    return summary[OUTPUT_COLUMNS]


def derive_stint_length(df: pd.DataFrame, mapped: pd.DataFrame, source_file: Path) -> pd.Series:
    """Derive stint length from lap ranges or lap counts when no direct field exists."""
    start_col = first_existing_column(df, ["StintStart", "stint_start", "StartLap", "start_lap"])
    end_col = first_existing_column(df, ["StintEnd", "stint_end", "EndLap", "end_lap"])
    if start_col and end_col:
        start = pd.to_numeric(df[start_col], errors="coerce")
        end = pd.to_numeric(df[end_col], errors="coerce")
        return end - start + 1

    if "LapNumber" in mapped:
        group_cols = [col for col in ("Year", "Event", "Session", "Driver", "Stint", "Compound") if col in mapped]
        if group_cols:
            laps = pd.to_numeric(mapped["LapNumber"], errors="coerce")
            return laps.groupby([mapped[col] for col in group_cols], dropna=False).transform("count")

    raise ValueError(f"{source_file} does not include enough information to derive StintLength.")


def extract_from_path(input_path: str, dataset_name: str) -> pd.DataFrame:
    """Extract all supported CSV files under a local path."""
    frames = [
        standardize_frame(df, path, dataset_name)
        for path, df in read_tabular_files(root_relative_path(input_path))
    ]
    return pd.concat(frames, ignore_index=True)


def download_kaggle_dataset(dataset_slug: str, download_dir: str) -> Path:
    """Download and unzip a Kaggle dataset through the Kaggle CLI."""
    requested_slug = dataset_slug
    dataset_slug = DATASET_SLUG_ALIASES.get(dataset_slug, dataset_slug)
    if dataset_slug != requested_slug:
        print(f"Using corrected Kaggle dataset slug: {dataset_slug}")

    target = root_relative_path(download_dir)
    target.mkdir(parents=True, exist_ok=True)
    if find_supported_data_files(target):
        print(f"Using existing downloaded Kaggle files in {target}")
        return target

    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target), "--force"],
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI was not found. Install and authenticate it with "
            "`uv pip install kaggle`, then place kaggle.json under your Kaggle config "
            "directory, or download the dataset manually and run this script with "
            "`--input path/to/downloaded/csvs`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Kaggle dataset download failed for '{dataset_slug}'. Make sure this is a "
            "real Kaggle dataset slug like `username/dataset-name`, not the placeholder "
            "`owner/dataset-slug`; confirm your Kaggle CLI is authenticated; and check "
            "that your account can access the dataset. You can also download the CSVs "
            "manually from Kaggle and run this script with `--input path/to/downloaded/csvs`."
        ) from exc
    for archive in target.glob("*.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(target)
    return target


def append_or_write_strategy_output(df: pd.DataFrame, out: str) -> None:
    """Append Kaggle strategy rows to a CSV output file and drop duplicate stints."""
    out_path = root_relative_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"Appending to an existing file: {out_path}")
        existing_df = pd.read_csv(out_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df = deduplicate_strategy_rows(df)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


def deduplicate_strategy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return strategy rows with a stable schema and duplicate logical stints removed."""
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    compound_key = df["Compound"].astype("string").str.strip()
    df = df[compound_key.isin(ALLOWED_COMPOUNDS)]

    dedupe_cols = [
        "Dataset",
        "Year",
        "Event",
        "Session",
        "Driver",
        "Stint",
        "Compound",
    ]
    df = df[OUTPUT_COLUMNS]
    dedupe_key = df[dedupe_cols].astype("string").apply(lambda col: col.str.strip())
    dedupe_key = dedupe_key.replace(
        {"": "<missing>", "nan": "<missing>", "NaN": "<missing>", "<NA>": "<missing>"}
    ).fillna("<missing>")
    key_cols = [f"__dedupe_{col}" for col in dedupe_cols]
    df_with_key = pd.concat(
        [df.reset_index(drop=True), dedupe_key.reset_index(drop=True).set_axis(key_cols, axis=1)],
        axis=1,
    )
    return df_with_key.drop_duplicates(subset=key_cols, keep="last")[OUTPUT_COLUMNS]


def display_path(path: str | Path) -> str:
    """Return a project-relative path label when possible."""
    resolved_path = root_relative_path(str(path)).resolve()
    project_root = root_relative_path(".").resolve()
    try:
        return str(resolved_path.relative_to(project_root))
    except ValueError:
        return str(resolved_path)


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(
        description="Extract Kaggle F1 tyre strategy stint features."
    )
    parser.add_argument("--input", help="Local CSV file or directory containing Kaggle CSV exports.")
    parser.add_argument("--kaggle-dataset", help="Kaggle dataset slug, e.g. owner/dataset-name.")
    parser.add_argument(
        "--download-dir",
        default="data/raw/kaggle_tyre_strategy",
        help="Directory for Kaggle CLI downloads.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Kaggle F1 tyre strategy dataset",
        help="Dataset label written to the output CSV.",
    )
    parser.add_argument(
        "--out",
        default="outputs/kaggle_tyre_strategy_output.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    if args.kaggle_dataset:
        input_path = download_kaggle_dataset(args.kaggle_dataset, args.download_dir)
        dataset_name = args.kaggle_dataset
    elif args.input:
        input_path = root_relative_path(args.input)
        dataset_name = args.dataset_name
    else:
        parser.error("one of --input or --kaggle-dataset is required")

    df = extract_from_path(str(input_path), dataset_name)
    append_or_write_strategy_output(df, args.out)
    print(f"Extracted {len(df)} rows from {display_path(input_path)} into {display_path(args.out)}")
    return df


# Example usages:
# uv run python datasets/extract_kaggle_tyre_strategy.py --kaggle-dataset \
# navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
# Useful Kaggle dataset names:
# 1. "navenkumar1998/formula-1-dataset-with-weather-and-tyre-features"
# 2. "aadigupta1601/f1-strategy-dataset-pit-stop-prediction"
# 3. "playground-series-s6e5"
# 4. "oshomuralidaran/pitwall-analytics"
if __name__ == "__main__":
    main()
