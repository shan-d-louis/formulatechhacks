# Model artefacts

Trained models used by the SIDEWALL server (`python -m sidewall.server.app` loads them at start-up). Each has a
`*_metrics.json` next to it with its held-out evaluation. Files are `joblib` dumps (Python pickles): load only
copies from this repository, never from an untrusted source.

| File | Size | What it is | Trained on | Evaluation | Rebuild with | SHA-256 (first 16) |
|---|---|---|---|---|---|---|
| `tierA_events.joblib` | 16.7 MB | LightGBM early warnings: lock-up, wheelspin, overheat, cold tyre (probability of the event within ~1 s) | Assetto Corsa Gym, Dallara F317, 116 human stints | `tierA_metrics.json`: leave-session-out, leave-track-out, external GT car | `python -m sidewall.models.event_detectors --rebuild` | `38973e00d735434c` |
| `risk.joblib` | 2.1 MB | Calibrated, explained lock-up / wheelspin risk: Stage-1 LightGBM + per-car Stage-2 logistic (demand, tyre temperature, pressure) for `acgym` and `sim` | Assetto Corsa Gym + a simulator calibration run | `risk_metrics.json`: ROC-AUC, Brier, calibration error, odds ratios | `python -m sidewall.models.risk` (`--sim-only` recalibrates just the simulator car) | `0eb60ab13c9a8662` |
| `twin_tpms.joblib` | 23.8 MB | Virtual TPMS: core and surface temperature per tyre from telemetry (pressure then follows the gas law) | Assetto Corsa Gym | `twin_metrics.json`: leave-track-out error vs a naive baseline | `python -m sidewall.twin.virtual_tpms` | `30f6dfafecd8e0e6` |
| `tierB_tyre_life.joblib` | 0.7 MB | Tyre life on real F1: cliff hazard, laps-to-cliff (conformal safe-laps bound), failure hazard, alert thresholds, circuit priors | FastF1 2018-21 + 2024 R1-12 (train), 2024 R13-24 (calibration) | `tierB_metrics.json`: test = all of 2025, never used for fitting | `python -m sidewall.models.tyre_life` | `d0328c7596c6aa4d` |
| `tierB_excl_2020.joblib` | 0.7 MB | Same models refitted without 2020, used by the Silverstone 2020 replays so they are out-of-sample | as above minus 2020 | – | built on demand by the replay | `5309ade2afdcee17` |
| `tierB_excl_2021.joblib` | 0.7 MB | Same, without 2021 (Baku 2021 replay) | as above minus 2021 | – | built on demand by the replay | `6fe4b38d5b9049cc` |

Headline held-out results (full tables in the JSON files and the project README):
- Lock-up / wheelspin warning ROC-AUC 0.96 / 0.98 on unseen driver sessions, 0.90 / 0.96 on an unseen track.
- Risk calibration error 0.1-0.2 % (Assetto Corsa); 3.5 % (lock-up) / 1.9 % (wheelspin) on the live sim car after per-car calibration.
- Cliff hazard ROC-AUC 0.866 on 2025 (train 0.915); safe-laps bound breached on 7.3 % of 2025 laps (target <= 10 %).
- Virtual TPMS: 5.9-9.4 °C temperature error on an unseen track (naive 10.9-14.2 °C), 0.98 psi pressure error.
