# SIDEWALL: AI tyre-safety pit wall

**FormulaTech Hacks: Track 1 (Safety Diagnosis) · Ollon (Data-Driven Motorsport Safety) · Ampere (AI for Motorsport Safety)**

Since 2022 every F1 car has carried a standard FIA tyre-pressure sensor, but there is no public analytics layer that
*predicts* tyre failure. Tyres keep causing dangerous incidents:
- **Baku 2021:** Verstappen and Stroll had blowouts at around 300 km/h.
- **Silverstone 2020:** three front-left failures in the final laps.
- **Qatar 2023:** kerbs caused sidewall damage, and the FIA imposed an emergency 18-lap cap.
- **Nürburgring 2005:** a flat spot vibrated a suspension to failure.

SIDEWALL watches every tyre in real time. It detects lock-ups, wheelspin, overheating, cold tyres, flat spots and air loss. It predicts how many laps each tyre has left, and radios the pit crew's phones when it's time to box.

## How it maps to the tracks
| | What SIDEWALL does |
|---|---|
| **Track 1: Safety Diagnosis** | Real-time per-tyre health (0–100) and an escalating pit call (OK, ADVISE, BOX THIS LAP, BOX NOW), sent to crew phones. Lock-up and wheelspin warnings come 0.5–1.75 s early, and the laps-to-failure bound is ready before the cliff arrives. |
| **Ollon: data-driven** | 2018–2025 FastF1 data (thousands of stints) turned into a tyre-safety dataset, and a **data-driven stint cap for every circuit** (Kaplan–Meier), i.e. a "Qatar rule" everywhere before anything breaks. |
| **Ampere: AI** | Early-warning event detectors trained on sim ground truth and run on real F1 telemetry, a virtual TPMS, and survival models with conformal calibration. |

- [Product Overview](./How%20SIDEWALL%20works.pdf)
- [Product Improvements](./SIDEWALL%20clearer%20pit%20wall%20concept.pdf)

## Models (all trained on existing public datasets)
| Model | Data | Held-out result |
|---|---|---|
| Lock-up early warning | Assetto Corsa Gym, Dallara F317, 116 human stints, per-wheel slip ratio as ground truth | ROC-AUC 0.96 on unseen drivers, 0.90 on an unseen track, 0.90 on a different car. 85% caught, median 0.5 s early |
| Wheelspin early warning | same | ROC-AUC 0.98 / 0.96 / 0.93. 99% caught, median 1.75 s early |
| Overheating / cold tyre | same | Overheating 0.89 (0.70 on an unseen track, so the dashboard leans on the virtual TPMS). Cold tyre 0.99 |
| Virtual TPMS (core and surface temperature per tyre, pressure via the gas law) | same | Unseen track: 5.9–9.4 °C error against 10.9–14.2 °C for a naive baseline; pressure within 0.98 psi |
| Cliff hazard + safe laps left | FastF1 real races (train 2018–21 + 2024 R1–12, calibrate 2024 R13–24, test all of 2025) | Cliff ROC-AUC **0.87 on unseen 2025** (train 0.92, overfit gap 0.05). The conformal safe-laps bound is breached **7.3%** of the time on 2025 (target ≤ 10%; 15% without calibration) |
| Tyre failure hazard | FastF1 + Jolpica retirements + detected lap-time spikes | ROC-AUC 0.82 on 2025, but only 5 failures there, so it's noisy; race-grouped CV 0.68 is the more honest number |
| Slow puncture / deflation | physics: gas-mass invariant P_abs/T_abs + CUSUM | Synthetic tests: a leak is caught while the tyre warms, before raw pressure moves; heat cycles never trigger it |
| Flat spot | locked-sliding distance + once-per-revolution vibration (order tracking) | Synthetic tests pass; there are no false alarms on pure noise |

The **feature contract** (`sidewall/features.py`) is the key to going from sim to real. The detectors only ever see what a public F1 feed contains: speed, throttle, an on/off brake flag, gear, RPM and position, all at 4 Hz. Sim data is degraded to look like that. Accelerations rebuilt from position match the sim's own accelerometer with correlations of 0.97 (lateral) and 0.91 (longitudinal).

## Lock-up and wheelspin risk: predictive, calibrated, explained
Each warning answers three questions: **how likely** (in the next second), **why**, and **what to do about it**.

1. **Stage 1: telemetry pattern.** LightGBM on the shared telemetry features, trained without class re-weighting, so its
   output is a real probability. On held-out driver sessions the calibration error is 0.2%: a "20%" really does lead to the
   event about 1 time in 5. TreeSHAP splits every prediction into **Braking**, **Throttle**, **Speed & cornering**,
   **Engine & gearing** and **Tyre heat history**.
2. **Stage 2: demand vs grip, per car.** A logistic regression adds driver demand (braking and throttle demand, cornering
   load) and measured tyre condition (surface temperature outside the 85–115 °C window, pressure off the operating target).
   Its coefficients read as odds ratios:
   - **Live simulator car** (fitted on an 8-minute calibration run with mixed styles and pressure set-ups):
     - fronts 10 °C above the window multiply lock-up odds ×14
     - hot rears multiply wheelspin odds ×3.2
     - each psi of low rear pressure multiplies wheelspin odds ×1.3
     - lock-up ROC-AUC improves 0.62 → 0.78 and calibration error falls 20% → 3%
   - **Assetto Corsa data (used for real-race replays):** lock-ups and wheelspin are driven by how the driver brakes and
     accelerates. Tyre temperature adds almost nothing, and we report that as found.
3. **Prevention.** The factor that dominated over the last few corners becomes a plain instruction to the driver and pit
   crew, for example "Front-left overheating (128 °C): lift and coast before the big stops" or "More throttle than the rears
   can put down: short-shift out of slow corners". Sustained risk raises a MANAGE call on the pit wall and the crew phones.

`python -m sidewall.models.risk` trains both stages and writes `models/weights/risk_metrics.json` (calibration tables and
odds ratios).

## Guarding against overfitting (Tier B)
The first version memorised races: training ROC-AUC 0.997 against 0.85 on held-out seasons. Weather columns alone
scored 0.90 on training and 0.51 on test, because they identify individual races. `python -m sidewall.models.diagnostics`
reproduces this. The fixes:
- **Causal features only.** The degradation trend is refitted each lap from completed laps (it previously saw the whole stint).
- **No race fingerprints.** Air temperature and humidity are dropped; track temperature is coarsened to 5 °C bands; an 18-inch-tyre era flag is added.
- **Out-of-season circuit priors.** A training row never sees statistics that include its own stint.
- **Cleaner labels.** Yellow-flag laps are not "clean", and a "cliff" shared by 3+ drivers on the same lap (traffic, weather) is dropped.
- **All at-risk laps are kept,** including short stints (they are censored observations).
- **Monotone constraints and model size chosen by race-grouped cross-validation.** Older tyres and faster degradation can never lower the risk.
- **Honest replays.** Each demo replay uses a model retrained without that race's season.

Result: the train–test gap fell from about 0.15 to 0.05 (cliff) and from about 0.35 to 0.09 (laps-to-cliff), with test accuracy the same or better. 2022–2023 were skipped to save download time.
## Data Sources and Evidence Limits

Expected data sources include:

- **FastF1 / OpenF1**: public lap, car, position, weather, tyre-stint, and race-context
  data.
- **Race-control/status data**: tyre, puncture, wheel, and wheel-nut event labels.
- **Sim racing or academic telemetry**: optional higher-frequency data with per-wheel
  slip, tyre temperatures, pressure, and wear when available.
- **Local demos**: simulated or phone-driven inputs for driver, crew, replay, or
  pit-wall experiences.

### TyreFrame Normalization

SIDEWALL maps source exports into hierarchical `TyreFrame` tables rather than one
flat row-level join. `outputs/telemetry_output.csv` and `outputs/openf1_output.csv`
provide public F1 macro context such as driver, elapsed time, speed, throttle, brake,
RPM, compound, tyre life, and weather. `datasets/spa/event_windows.csv` and
`datasets/spa/candidate_events.csv` provide simulator micro-behaviour at event/window
level, including wheel speeds, slip proxies, pressures, and tyre temperatures when
the simulator supplies them. `outputs/kaggle_tyre_strategy_output.csv` provides
stint-level compound, stint length, and weather context for degradation and
laps-to-cliff experiments.

The normalization step enforces lowercase snake_case names, converts source
timestamps into cumulative `elapsed_s`, flags unavailable public F1 tyre pressure,
temperature, wheel-speed, and slip channels as `estimated_via_twin = True`, and keeps
macro lap/stint context separate from high-frequency simulator windows to avoid
leakage between neighbouring braking zones.

```bash
uv run python datasets/normalize_tyreframe.py
```

The command writes hierarchical CSVs under `outputs/tyreframe/`: `macro_telemetry`,
`macro_laps`, `stints`, `candidate_events`, `micro_event_windows`, and a row-count
`manifest`. Use chronological or circuit/session-level splits for training rather
than random row splits when building lock-up, wheelspin, or cliff-forecast models.

For the checked-in local extracts used on this branch, reproducibility means the
source and normalized row counts should remain stable unless an extraction is
intentionally refreshed. After running the command above, verify
`outputs/tyreframe/manifest.csv` against these expected counts:

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

If these totals change, document why in the same change as the data refresh or
normalization update.

### Target Open Datasets and APIs

| Source | Access | Useful fields | SIDEWALL use |
|---|---|---|---|
| **FastF1 Python library / data API** | GitHub `theOehrly/Fast-F1` or PyPI package `fastf1` | `Speed`, `Throttle`, `Brake`, `RPM`, `Gear`, `TrackTemp`, `AirTemp`, lap and stint context | Primary public telemetry source for replay, braking-zone analysis, tyre-stint context, weather context, and fuel-adjusted lap-time modelling. |
| **OpenF1 API** | `https://openf1.org` | Historical and live JSON/CSV streams for speed, throttle, RPM, timing intervals, session context | API-friendly telemetry source for live-ish demos, driver-specific timing, micro-sector deltas, and acceleration-drop analysis. |
| **Kaggle F1 tyre strategy datasets** | Search Kaggle for F1 tyre strategy datasets such as `F1-Tyre-Strategy-Engine` projects | `Compound`, `StintLength`, aggregated `AirTemp`, `TrackTemp`, stint summaries | Tabular ML starting point for tyre degradation, stint-length, compound, and environmental feature experiments. |

Known useful Kaggle targets:

- `navenkumar1998/formula-1-dataset-with-weather-and-tyre-features`: primary Kaggle
  target for SIDEWALL stint and degradation experiments. It is lap-level and includes
  tyre/stint features such as compound, tyre life, fresh tyre, and stint, plus weather
  context such as air temperature, track temperature, rainfall, humidity, and wind.

Be precise about what the data can prove:

- Public FastF1 telemetry can support a **possible lock-up** or **braking anomaly**
  flag, but it cannot directly confirm individual wheel lock-up.
- Standard FastF1 telemetry does not directly provide tyre pressure, tyre temperature,
  actual fuel mass, or individual wheel rotation speed.
- Tyre state inferred from public telemetry must be labelled as `estimated`,
  `possible`, or `candidate` unless another source directly measures it.
- Visual cues such as smoke are reviewer evidence, not automatic confirmation.

## Candidate Models

These models are initial targets for the datasets above. Implement them test-first and
document each model's inputs, assumptions, limitations, and verification command as the
code lands.

### Bayesian State-Space Degradation Model

**Goal:** isolate tyre wear from confounders such as fuel burn and pit-stop resets.

Raw lap times often improve early in a race as cars lose fuel mass. A state-space model
should use lap-time evolution, stint boundaries, pit stops, compound, track
temperature, and air temperature to estimate latent tyre pace. The useful output is not
just a pace prediction, but a signal that identifies when observed degradation departs
from expected tyre behaviour.

Expected outputs:

- Estimated latent tyre pace or degradation state.
- Uncertainty around the estimated state.
- Human-readable reason text for degradation anomalies.
- Clear separation between fuel-adjusted pace estimates and directly measured data.

### Deceleration / Braking Anomaly Classifier

**Goal:** flag possible wheel lock-ups, tyre slides, or flat-spotting risk moments for
human review.

Using public telemetry, start with speed and brake traces around heavy braking zones.
Feature candidates include deceleration shape, speed-drop gradient, brake-on duration,
corner/location context, tyre age, compound, and track conditions. Simple thresholds or
unsupervised models such as Isolation Forests are acceptable starting points before
more complex classifiers.

Expected outputs:

- `possible_lockup` or `braking_anomaly` style labels, not confirmed lock-up labels
  unless direct wheel-speed or reviewed visual evidence is available.
- Explanation fields describing which telemetry signals triggered the flag.
- False-positive checks on normal heavy braking zones.
- Regression tests for alert wording and evidence level.

## Development Workflow

Use test-driven development for meaningful logic changes:

1. Write or update a focused failing test for the intended behaviour.
2. Implement the smallest useful change.
3. Run the focused test.
4. Refactor with tests passing.
5. Update docs, schemas, examples, and coverage notes before the change is complete.

Documentation is part of the development loop. When behaviour, commands, data
contracts, alert wording, model assumptions, or demo flows change, update this README
or the relevant local docs in the same pass.

## Testing Expectations

- Prefer synthetic, anonymized, minimal fixtures for tests.
- Do not require raw downloaded race data for unit tests.
- For diagnosis logic, test both the decision and the explanation shown to users.
- For public-telemetry-only lock-up logic, tests should assert wording such as
  `possible lock-up`, `braking anomaly`, or `candidate`, not confirmed lock-up.
- When coverage is run, update the relevant `TEST_COVERAGE.md` file with the command
  and observed percentage.

## Setup

This project uses Python 3.13 or newer for the main SIDEWALL app. The easiest
reproducible setup is with `uv`:

```bash
uv sync
```

If `uv` is not available, create a virtual environment and install the pinned runtime
requirements:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS/Linux, replace `.venv/Scripts/python` with `.venv/bin/python`.

## Start the Full-Stack App

The recommended full-stack entrypoint is `sidewall.server.app`. It serves the
FastAPI backend, model APIs, WebSocket relay, static browser UI, pit-wall dashboard,
crew phone, driver phone controller, live simulator, replay mode, and atlas from one
process on port 8000.

```bash
uv run python -m sidewall.server.app
```

Without `uv`:

```bash
.venv/Scripts/python -m sidewall.server.app
```

Then open:

- `http://localhost:8000/` - landing page with all demo paths.
- `http://localhost:8000/pitwall?mode=replay` - replay a real race scenario.
- `http://localhost:8000/pitwall?mode=live` - run the live phone-driven simulator
  and pit-wall frontend.
- `http://localhost:8000/crew` - pit-crew phone view.
- `http://localhost:8000/driver` - phone throttle/brake controller.
- `http://localhost:8000/atlas` - tyre-safety data atlas.
- `http://localhost:8000/docs` - FastAPI API docs.

For the live demo, start the server on the laptop, open `/pitwall?mode=live`, then use
the QR buttons on the pit wall to connect phones on the same Wi-Fi. The server binds to
`0.0.0.0` and `/api/qr` generates LAN URLs for `/crew` and `/driver`.

If you see a page that says **Backend offline**, you are probably on the older
`simulator/` or `dashboard/` UI. The full-stack app's live frontend is
`http://localhost:8000/pitwall?mode=live`.

Useful health checks:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/model/status
```

If you want to regenerate the checked-in demo data and model artifacts before serving,
run the pipeline in this order:

```bash
uv run python -m sidewall.data.ingest_fastf1 --telemetry
uv run python -m sidewall.data.ingest_acgym
uv run python -m sidewall.data.build_stints
uv run python -m sidewall.models.event_detectors --rebuild
uv run python -m sidewall.twin.virtual_tpms
uv run python -m sidewall.models.tyre_life
uv run python -m sidewall.data.build_atlas
```

Those commands may download or refresh external data. The repository already includes
demo outputs and model weights for local startup, so you do not need to run the data
pipeline just to try the app.

### Notes for troubleshooting

Make sure these datasets are installed:
```python
SCENARIOS = {
    "silverstone2020": Scenario(
        "silverstone2020", "British GP 2020: Hamilton's front-left", 2020, 4, "HAM", 20, 52,
        "Front-left tyre failed on the final lap after a ~40-lap stint on hards; won on three wheels.",
        failure_lap=52),
    "baku2021": Scenario(
        "baku2021", "Azerbaijan GP 2021: Verstappen's left-rear", 2021, 6, "VER", 14, 46,
        "Left-rear failed at ~300 km/h on lap 46 while leading; Pirelli blamed low running pressures.",
        failure_lap=46),
    "silverstone2020_whatif": Scenario(
        "silverstone2020_whatif", "What-if: slow puncture injected (Silverstone 2020)", 2020, 4, "HAM", 20, 40,
        "Synthetic: a 0.8 %/min leak is injected on the front-left TPMS channel at lap ~24 to show the "
        "temperature-independent leak detector. Not real data.",
        leaks={"fl": (400.0, 0.008)}),
}
```
If not installed, run the following scripts from the root of this repository: 
```bash
uv run python -m sidewall.data.ingest_fastf1 --years 2020 2021 --telemetry
uv run python -m sidewall.data.build_stints
```

### Feedback Loop
Check [feedback.py](./sidewall/server/feedback.py).

Tests:

```bash
uv run pytest -q
```

## Demo (about 4 minutes)
1. **Ghost of Silverstone 2020.** Replay Hamilton's stint at 20× with the Radio switch on and a judge's phone on the Crew QR code. The model has **never seen 2020**. The call reaches ADVISE on lap 36, BOX on lap 40 and sustained BOX from lap 44, eight laps before the real front-left failure on lap 52; failure risk hits the top 1% on lap 49, when Bottas and Sainz failed. Everything on screen uses only data available up to that moment.
2. **"Drive it yourself."** Pick *LIVE*; a judge scans the Driver QR code and uses the phone pedals. Braking late gives a lock-up (the phone buzzes) and then a flat spot. Flooring it out of slow corners gives wheelspin. **💥 Debris** starts a slow puncture, and the air-loss detector catches it while the raw pressure still looks normal. The crew phone flashes BOX.
3. **Atlas.** Circuits ranked by data-driven stint cap, survival curves, degradation per season, failures by tyre age and the model scorecard.

## Honest limitations
- Public F1 data has **no tyre pressure, temperature or wheel speed**. Temperatures and pressures in replays are *estimates* from the virtual TPMS, calibrated on an F3-class sim. Leaks appear in replays only in the clearly labelled what-if scenario.
- The ML lock-up detector learned the Dallara's lock-up signature. In the live sim the car also has wheel-speed sensors (as real race cars do), and the dashboard shows which source fired: `wheel-speed`, `AI` or `sensor+AI`.
- There are few tyre failures in public data, so the failure hazard is weak. Alerts use percentile thresholds (top 5% / top 1% of training laps) rather than being tuned to the demo races.

## Data sources
FastF1 (MIT), OpenF1, Jolpica/Ergast, Assetto Corsa Gym (CC-BY-4.0, only `.ld` logs are loaded, never `.pkl` pickles), THULab/Nasim435 Spa telemetry (MIT).
Extract downloaded Kaggle F1 tyre strategy CSVs, or a Kaggle dataset slug if the
Kaggle CLI is installed and authenticated:

```bash
uv run python datasets/extract_kaggle_tyre_strategy.py --input data/raw/f1_strategy
uv run python datasets/extract_kaggle_tyre_strategy.py --kaggle-dataset navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
```

If `uv` is not available, install dependencies with your preferred Python environment
manager using `pyproject.toml` as the source of truth.

## Lower-Level Simulator Pipeline

Most users should start the full app with `uv run python -m sidewall.server.app`.
The repository also keeps an older, lower-level simulator pipeline for testing the raw
browser simulator against the lightweight backend directly.

Browser simulator -> FastAPI backend -> pit-wall dashboard, over WebSockets at 10 Hz.
Message formats are specified in `contracts.md`. Tyre temperatures and pressures here
are simulated, not real.

- `simulator/` - drivable browser car; sends raw sensor frames only.
- `backend/` - feature extraction, detectors, alert engine, Tyre Health Index, and
  laps estimate.
- `dashboard/` - displays analysed backend output frames only.
- `training/` - offline laps model scripts. `training/train.py` saves
  `training/laps_model.joblib`, which the lightweight backend loads.
- `backend/scenarios.py` - runs the demo scenarios headless through the backend.

Run it with two terminals from the repository root:

```bash
uv run uvicorn backend.main:app --reload --port 8001
```

```bash
python -m http.server 5500
```

Then open both browser pages:

- `http://localhost:5500/simulator/?backend=localhost:8001` - sends raw simulated
  tyre sensor frames to `ws://localhost:8001/ws/sim`.
- `http://localhost:5500/dashboard/?backend=localhost:8001` - dashboard UI that reads
  analysed frames from `ws://localhost:8001/ws/dash`.

The dashboard will show **Backend offline** until `backend.main` is running and the
`backend=localhost:8001` query parameter points at the same port. After the dashboard
connects, it may still say it is waiting for frames until the simulator page is open
and sending data.

Backend-specific tests:

```bash
uv run pytest -q backend/tests
```

## Model Serving API

The SIDEWALL server also exposes the tyre-life model as a low-latency FastAPI surface.
Small predictions stay on the API path, while replay rebuilds and heavier analytics are
queued for a background worker so the API can respond immediately.

```bash
uv run python -m sidewall.server.app
```

- `GET /health` - service health and active laps-model source.
- `GET /api/model/status` - model path, source, bundle metadata, CV metrics, and evidence limits.
- `POST /api/predict/laps` - estimated laps remaining for one compound/age/track-temperature state.
- `POST /api/jobs/replay` - enqueue replay analytics and return a job id with `202 Accepted`.
- `GET /api/jobs/{job_id}` - poll queued/running/succeeded/failed status.
- `GET /api/jobs/{job_id}/result` - read the completed replay result.

Replay rebuilds should use `/api/jobs/replay`; direct `/api/replay/{key}?rebuild=true`
is rejected so parquet/model-heavy work does not block the request path.

## Project Hygiene

- Do not commit credentials, private tokens, raw large datasets, generated caches, or
  model checkpoints.
- Keep FastF1 caches and generated outputs out of reviewed application logic unless
  the code explicitly documents how they are produced.
- Use relative paths or configuration instead of hard-coded local paths.
- Record seeds, commands, and data-source assumptions for reproducible experiments.
- Keep alert copy honest about uncertainty and evidence level.

## Related Docs

- `TRACKS.md` - hackathon track summary.
- `.github/skills/AGENTS.md` - shared agent-facing repository guidance.
- `.github/instructions/copilot-instructions.md` - GitHub Copilot-specific
  instructions.
- `.agents/skills/` - local skills for testing, data pipeline, modelling, training,
  evaluation, notebooks, and repo guidance.

Different assistants should keep their runtime-specific instructions in their own
conventional directories or files, then refer back to the shared docs above for
repository facts, safety limits, data-source assumptions, and development workflow.
