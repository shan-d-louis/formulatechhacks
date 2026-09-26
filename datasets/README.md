# SIDEWALL Dataset Pipeline

This directory contains the source adapters and cleaning utilities used to reproduce
SIDEWALL's current TyreFrame data products. The pipeline keeps source tables at their
natural sampling levels: public F1 telemetry remains macro context, simulator windows
remain event-level micro telemetry, and Kaggle data remains stint/lap summary context.

## Purpose

The dataset pipeline supports three modelling needs:

| Source family | Files | Purpose |
|---|---|---|
| Public F1 telemetry | `outputs/telemetry_output.csv`, `outputs/openf1_output.csv` | Macro race context: driver, session, speed, throttle, brake, RPM, gear, lap context, compound, tyre life, air temperature, track temperature, rainfall, intervals, and timing context. |
| Simulator event telemetry | `datasets/spa/candidate_events.csv`, `datasets/spa/event_windows.csv` | Micro tyre behaviour: event windows around braking anomalies with per-wheel speeds, slip proxies, tyre pressure, and tyre core temperature when supplied by the simulator. |
| Kaggle tyre strategy data | `outputs/kaggle_tyre_strategy_output.csv` | Stint and degradation context: compound, stint length, weather aggregation, and lap/stint grouping for cliff-risk and survival-style experiments. |

Public FastF1/OpenF1 data does not directly measure per-wheel angular velocity, tyre
pressure, tyre temperature, or actual fuel mass. Treat those public-F1-only tyre
channels as unavailable or `estimated_via_twin`, not measured ground truth.

## Reproduction Order

Run commands from the repository root.

1. Install the declared environment.

   ```bash
   uv sync
   ```

2. Extract FastF1 macro telemetry.

   ```bash
   uv run python datasets/extract_fastf1.py
   ```

   Default sessions are defined in `datasets/extract_fastf1.py` and currently cover
   2024 Monza race drivers `LEC`, `PIA` and 2024 Las Vegas race drivers `VER`, `RUS`.
   The output is `outputs/telemetry_output.csv`.

3. Extract OpenF1 macro telemetry and timing intervals.

   ```bash
   uv run python datasets/extract_openf1.py
   ```

   Defaults mirror the FastF1 sessions above. The output is
   `outputs/openf1_output.csv`. OpenF1 uses the public API and has rate limits, so
   reruns can take time and may differ if upstream historical records change.

4. Extract Kaggle tyre strategy summaries.

   ```bash
   uv run python datasets/extract_kaggle_tyre_strategy.py --kaggle-dataset navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
   ```

   If the Kaggle CLI is unavailable or unauthenticated, manually place downloaded CSV
   or Parquet files under `data/raw/kaggle_tyre_strategy/` and run:

   ```bash
   uv run python datasets/extract_kaggle_tyre_strategy.py --input data/raw/kaggle_tyre_strategy --dataset-name navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
   ```

   The output is `outputs/kaggle_tyre_strategy_output.csv`.

5. Keep simulator fixtures available.

   The Spa simulator analysis files currently live at:

   ```text
   datasets/spa/candidate_events.csv
   datasets/spa/event_windows.csv
   ```

   These are not joined row-by-row to public telemetry. They are evaluated as
   candidate event metadata and high-frequency event-window telemetry.

6. Normalize into TyreFrame tables.

   ```bash
   uv run python datasets/normalize_tyreframe.py
   ```

   This writes hierarchical outputs under `outputs/tyreframe/`.

## TyreFrame Outputs

| Output | Sampling level | Purpose |
|---|---|---|
| `outputs/tyreframe/macro_telemetry.csv` | Public telemetry samples | Unified FastF1/OpenF1 macro telemetry with snake_case columns and session-relative `elapsed_s`. |
| `outputs/tyreframe/macro_laps.csv` | Lap-level macro table | Lap-context table for degradation, stint, weather, and public-telemetry anomaly features. |
| `outputs/tyreframe/stints.csv` | Stint summary | Kaggle-derived compound, stint length, and weather context. |
| `outputs/tyreframe/candidate_events.csv` | Event metadata | Simulator candidate anomaly rows, with stable `event_id` values when needed. |
| `outputs/tyreframe/micro_event_windows.csv` | Event-window samples | Simulator high-frequency wheel/slip/pressure/temperature windows after timestamp deduplication and low-speed filtering. |
| `outputs/tyreframe/manifest.csv` | Audit metadata | Source row counts, output row counts, and totals for reproducibility checks. |

The normalizer enforces lowercase snake_case names, converts timestamps into
session-relative `elapsed_s`, marks public F1 tyre-only channels as
`estimated_via_twin = True`, drops duplicate simulator timestamps within each
`event_id`, filters simulator rows below `10 km/h`, and clamps extreme slip-proxy
outliers caused by dropouts.

## Current Count Contract

For the current local extracts, `outputs/tyreframe/manifest.csv` should contain these
counts after normalization:

| Manifest record | Expected rows |
|---|---:|
| `source_table: fastf1` | 70,801 |
| `source_table: openf1` | 123,156 |
| `source_table: stints` | 1,010 |
| `source_table: candidate_events` | 5 |
| `source_table: event_windows` | 1,334 |
| `output_table: macro_telemetry` | 193,957 |
| `output_table: macro_laps` | 210 |
| `output_table: stints` | 1,010 |
| `output_table: candidate_events` | 5 |
| `output_table: micro_event_windows` | 1,334 |
| `total: source_records` | 196,306 |
| `total: output_records` | 196,516 |

If a rerun changes any count, update this file and `README.md` in the same change and
explain whether the difference came from a deliberate source refresh, an upstream API
change, a simulator fixture change, or normalization logic.

## Split and Leakage Rules

Do not randomly split individual rows for model training. Use chronological,
session-level, or circuit-level splits so neighbouring laps, stints, and event windows
do not leak between train and test sets. For simulator anomaly classifiers, deduplicate
timestamps within each `event_id` and keep nearby braking zones from crossing split
boundaries.

## Verification

Run the unit tests after changing adapters or normalization logic:

```bash
uv run pytest
```

The tests use synthetic fixtures and do not require downloading raw race data.
