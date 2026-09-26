from pathlib import Path
import subprocess

import pandas as pd

from datasets import extract_kaggle_tyre_strategy as kaggle_strategy
from datasets.extract_kaggle_tyre_strategy import extract_from_path, standardize_frame


def test_standardize_frame_maps_common_strategy_columns(tmp_path: Path) -> None:
    source = tmp_path / "strategy.csv"
    df = pd.DataFrame(
        {
            "season": [2024, 2024],
            "race": ["Monza", "Monza"],
            "driver": ["VER", "VER"],
            "stint": [1, 1],
            "tyre_compound": ["MEDIUM", "MEDIUM"],
            "stint_laps": [18, 18],
            "air_temp": [25.0, 27.0],
            "track_temp": [39.0, 41.0],
        }
    )

    result = standardize_frame(df, source, "fixture")

    assert result.loc[0, "Compound"] == "Medium"
    assert result.loc[0, "StintLength"] == 18
    assert result.loc[0, "AirTemp"] == 26.0
    assert result.loc[0, "TrackTemp"] == 40.0


def test_extract_from_path_derives_stint_length_from_lap_rows(tmp_path: Path) -> None:
    source = tmp_path / "laps.csv"
    pd.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "Event": ["Spa", "Spa", "Spa"],
            "Driver": ["LEC", "LEC", "LEC"],
            "Stint": [2, 2, 2],
            "LapNumber": [11, 12, 13],
            "Compound": ["S", "S", "S"],
            "AirTemp": [20.0, 21.0, 22.0],
            "TrackTemp": [30.0, 31.0, 32.0],
        }
    ).to_csv(source, index=False)

    result = extract_from_path(str(tmp_path), "fixture")

    assert len(result) == 1
    assert result.loc[0, "Compound"] == "Soft"
    assert result.loc[0, "StintLength"] == 3
    assert result.loc[0, "AirTemp"] == 21.0
    assert result.loc[0, "TrackTemp"] == 31.0


def test_download_kaggle_dataset_corrects_known_slug_typo(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(command, check):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(kaggle_strategy.subprocess, "run", fake_run)

    kaggle_strategy.download_kaggle_dataset(
        "navenkumar1998/formula-1-dataset-with-weather-and-tyre-feature",
        str(tmp_path),
    )

    assert calls[0][4] == "navenkumar1998/formula-1-dataset-with-weather-and-tyre-features"


def test_strategy_output_appends_and_deduplicates(tmp_path: Path) -> None:
    output = tmp_path / "strategy_output.csv"
    first = pd.DataFrame(
        {
            "Dataset": ["fixture"],
            "SourceFile": ["strategy.csv"],
            "Year": [2024],
            "Event": ["Monza"],
            "Session": ["R"],
            "Driver": ["VER"],
            "Stint": [1],
            "Compound": ["Medium"],
            "StintLength": [18],
            "AirTemp": [25.0],
            "TrackTemp": [39.0],
        }
    )
    second = first.copy()
    second["StintLength"] = [19]

    kaggle_strategy.append_or_write_strategy_output(first, str(output))
    kaggle_strategy.append_or_write_strategy_output(second, str(output))

    saved = pd.read_csv(output)
    assert len(saved) == 1
    assert saved.loc[0, "StintLength"] == 19
