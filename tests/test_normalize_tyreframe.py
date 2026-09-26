from pathlib import Path

import pandas as pd

from datasets.normalize_tyreframe import (
    prepare_event_windows,
    prepare_fastf1,
    prepare_openf1,
    snake_case,
    write_tyreframe_tables,
)


def test_snake_case_standardizes_common_source_names() -> None:
    assert snake_case("DriverNumber") == "driver_number"
    assert snake_case("Track Temp C") == "track_temp_c"
    assert snake_case("psi_fl") == "psi_fl"


def test_public_f1_tables_flag_unavailable_tyre_channels(tmp_path: Path) -> None:
    fastf1_path = tmp_path / "fastf1.csv"
    pd.DataFrame(
        {
            "Year": [2024, 2024],
            "Event": ["Monza", "Monza"],
            "Session": ["R", "R"],
            "Driver": ["LEC", "LEC"],
            "Time": ["0 days 00:00:01", "0 days 00:00:03"],
            "Speed": [120.0, 130.0],
            "Throttle": [0.0, 100.0],
            "Brake": [True, False],
            "RPM": [9000, 11000],
        }
    ).to_csv(fastf1_path, index=False)

    result = prepare_fastf1(str(fastf1_path))

    assert result["elapsed_s"].tolist() == [0.0, 2.0]
    assert result["speed_kmh"].tolist() == [120.0, 130.0]
    assert result["estimated_via_twin"].eq(True).all()
    assert result["measurement_status"].eq("public_f1_tyre_channels_unavailable").all()
    assert "psi_fl" in result
    assert result["psi_fl"].isna().all()


def test_openf1_datetime_elapsed_seconds_are_session_relative(tmp_path: Path) -> None:
    openf1_path = tmp_path / "openf1.csv"
    pd.DataFrame(
        {
            "Year": [2024, 2024],
            "Event": ["Monza", "Monza"],
            "Session": ["R", "R"],
            "Driver": ["LEC", "LEC"],
            "date": [
                "2024-09-01T12:08:56.653Z",
                "2024-09-01T12:08:58.653Z",
            ],
            "speed": [0, 80],
            "throttle": [0, 10],
            "brake": [0, 20],
            "rpm": [0, 8000],
        }
    ).to_csv(openf1_path, index=False)

    result = prepare_openf1(str(openf1_path))

    assert result["elapsed_s"].tolist() == [0.0, 2.0]
    assert result["speed_kmh"].tolist() == [0, 80]


def test_event_windows_deduplicate_filter_low_speed_and_clamp_slip(tmp_path: Path) -> None:
    windows_path = tmp_path / "event_windows.csv"
    pd.DataFrame(
        {
            "event_id": [1, 1, 1],
            "elapsed_s": [10.0, 10.0, 11.0],
            "timestamp": [1_700_000_000.0, 1_700_000_000.0, 1_700_000_001.0],
            "speed_kmh": [80.0, 81.0, 5.0],
            "slip_proxy_fl": [9.0, -9.0, 0.1],
        }
    ).to_csv(windows_path, index=False)

    result = prepare_event_windows(str(windows_path))

    assert len(result) == 1
    assert result.iloc[0]["speed_kmh"] == 81.0
    assert result.iloc[0]["slip_proxy_fl"] == -3.0
    assert result.iloc[0]["sample_level"] == "event_window"


def test_write_tyreframe_tables_keeps_macro_and_micro_hierarchical(tmp_path: Path) -> None:
    fastf1_path = tmp_path / "fastf1.csv"
    openf1_path = tmp_path / "openf1.csv"
    stints_path = tmp_path / "stints.csv"
    events_path = tmp_path / "candidate_events.csv"
    windows_path = tmp_path / "event_windows.csv"
    out_dir = tmp_path / "tyreframe"

    pd.DataFrame(
        {
            "Year": [2024],
            "Event": ["Monza"],
            "Session": ["R"],
            "Driver": ["LEC"],
            "Time": ["0 days 00:00:01"],
            "LapNumber": [1],
            "Speed": [120.0],
            "Throttle": [10.0],
            "Brake": [False],
            "RPM": [9000],
            "Compound": ["Medium"],
        }
    ).to_csv(fastf1_path, index=False)
    pd.DataFrame(
        {
            "Year": [2024],
            "Event": ["Monza"],
            "Session": ["R"],
            "Driver": ["LEC"],
            "date": ["2024-09-01T12:08:56.653Z"],
            "speed": [0],
            "throttle": [0],
            "brake": [0],
            "rpm": [0],
        }
    ).to_csv(openf1_path, index=False)
    pd.DataFrame(
        {
            "Year": [2024],
            "Event": ["Monza"],
            "Session": ["R"],
            "Driver": ["LEC"],
            "Stint": [1],
            "Compound": ["Medium"],
            "StintLength": [18],
            "AirTemp": [25.0],
            "TrackTemp": [39.0],
        }
    ).to_csv(stints_path, index=False)
    pd.DataFrame(
        {
            "wheel": ["fr"],
            "start_s": [10.0],
            "end_s": [10.2],
            "duration_s": [0.2],
            "min_proxy_slip": [-0.8],
            "start_speed_kmh": [80.0],
            "peak_brake": [1.0],
        }
    ).to_csv(events_path, index=False)
    pd.DataFrame(
        {
            "event_id": [1],
            "elapsed_s": [10.0],
            "timestamp": [1_700_000_000.0],
            "speed_kmh": [80.0],
            "slip_proxy_fr": [-0.8],
        }
    ).to_csv(windows_path, index=False)

    tables = write_tyreframe_tables(
        fastf1_path=str(fastf1_path),
        openf1_path=str(openf1_path),
        stints_path=str(stints_path),
        event_windows_path=str(windows_path),
        candidate_events_path=str(events_path),
        out_dir=str(out_dir),
    )

    assert set(tables) == {
        "macro_telemetry",
        "macro_laps",
        "stints",
        "candidate_events",
        "micro_event_windows",
    }
    assert "event_id" not in tables["macro_laps"].columns
    assert tables["micro_event_windows"]["sample_level"].eq("event_window").all()
    manifest = pd.read_csv(out_dir / "manifest.csv")
    source_total = manifest.loc[manifest["name"] == "source_records", "rows"].iloc[0]
    output_total = manifest.loc[manifest["name"] == "output_records", "rows"].iloc[0]

    assert source_total == 5
    assert output_total == sum(len(table) for table in tables.values())
