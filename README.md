# FormulaTech Hacks - SIDEWALL

SIDEWALL is a FormulaTech Hacks tyre-safety diagnosis prototype: an AI-assisted
digital twin and pit-wall monitor that helps distinguish normal tyre degradation from
possible safety anomalies before they become dangerous.

The project focus is tyre safety, not generic telemetry. Acceleration anomalies can be
useful context, but the primary product direction is diagnosing tyre behaviour across
temperature, pressure, grip, wear, lock-up, wheelspin, flat spots, fatigue, and
pit-call risk.

## Current State

This repository is still in an early scaffold state. The current Python entry point is
`main.py`, dependencies are declared in `pyproject.toml`, and FastF1 cache/output files
may exist locally from exploratory work.

Do not treat cached telemetry, generated CSVs, or output folders as source-of-truth
application behaviour. As functionality lands, update this README alongside the code.

## Track Alignment

- **FormulaTech Track 1: Safety Diagnosis** - detect or identify tyre safety risk in
  real time or before it becomes critical.
- **Ollon: Data-Driven Motorsport Safety** - use telemetry and historical data to find
  patterns, predict risk, and explain tyre-safety insights.
- **Ampire/Ampere: AI for Motorsport Safety** - apply AI/ML models where they improve
  detection, forecasting, calibration, or explanations.

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

This project uses Python 3.13 or newer.

```bash
uv sync
```

Run the current entry point:

```bash
uv run python main.py
```

Extract downloaded Kaggle F1 tyre strategy CSVs, or a Kaggle dataset slug if the
Kaggle CLI is installed and authenticated:

```bash
uv run python datasets/extract_kaggle_tyre_strategy.py --input data/raw/f1_strategy
uv run python datasets/extract_kaggle_tyre_strategy.py --kaggle-dataset navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
```

If `uv` is not available, install dependencies with your preferred Python environment
manager using `pyproject.toml` as the source of truth.

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
