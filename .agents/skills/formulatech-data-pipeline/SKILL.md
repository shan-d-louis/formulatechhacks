---
name: formulatech-data-pipeline
description: Use when working on data ingestion, sensitive datasets, dataset classes, data normalization, sequence padding or masks, train/validation/test splits, augmentation, or src/data code.
---

# FormulaTech Data Pipeline

For data pipeline work, read and follow:

- `../../../.github/instructions/copilot-instructions.md`

Ingest datasets from the following sources:
1. **OpenF1 API & FastF1**
  - `Speed`: Instantaneous car velocity.
  - `Throttle` & `Brake`: Pedal application traces to map severe deceleration or traction zones.
  - `RPM` & `Gear`: Useful for tracking engine load and acceleration phase efficiency.
  - `TrackTemp` & `AirTemp`: Essential ambient contexts that dictate how quickly tires heat up and degrade.
  - Treat tyre pressure, tyre temperature, individual wheel speed, and actual fuel mass as unavailable unless another source supplies them directly.
2. **Kaggle F1 Tyre Strategy Datasets**
  - Search Kaggle for `F1-Tyre-Strategy-Engine` style community datasets.
  - Extract `Compound`, `StintLength`, aggregated `AirTemp`, and aggregated `TrackTemp`.
  - Prefer `navenkumar1998/formula-1-dataset-with-weather-and-tyre-features` for
    lap-level tyre/weather modelling.
  - Treat these as tabular stint summaries for ML experiments, not direct tyre pressure,
    tyre temperature, wheel-speed, or safety-failure ground truth.
3. **Sim Racing Telemetry Datasets (Assetto Corsa / iRacing)**:
  - Prefer these for per-wheel slip, tyre temperature, tyre pressure, and wear ground truth when validating detectors or the digital twin.
4. **Academic Datasets**: Politecnico di Torino's open automotive datasets

## Data Contracts and Documentation
- Document each source adapter's measured fields, estimated fields, unavailable fields, sampling rate, and units.
- Keep a small synthetic fixture for every adapter so parsing and normalization can be tested without downloading or exposing raw data.
- Update schemas, README notes, and coverage docs in the same pass as any data contract change.
- For public telemetry, use "possible lock-up" or "braking anomaly" labels unless direct wheel-speed data or reviewed visual evidence is available.

Do not change split logic, normalization strategy, label schema, or output conventions
without explicit user confirmation.
