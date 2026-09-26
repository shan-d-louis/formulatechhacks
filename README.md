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

### Target Open Datasets and APIs

| Source | Access | Useful fields | SIDEWALL use |
|---|---|---|---|
| **FastF1 Python library / data API** | GitHub `theOehrly/Fast-F1` or PyPI package `fastf1` | `Speed`, `Throttle`, `Brake`, `RPM`, `Gear`, `TrackTemp`, `AirTemp`, lap and stint context | Primary public telemetry source for replay, braking-zone analysis, tyre-stint context, weather context, and fuel-adjusted lap-time modelling. |
| **OpenF1 API** | `https://openf1.org` | Historical and live JSON/CSV streams for speed, throttle, RPM, timing intervals, session context | API-friendly telemetry source for live-ish demos, driver-specific timing, micro-sector deltas, and acceleration-drop analysis. |
| **Kaggle F1 tyre strategy datasets** | Search Kaggle for F1 tyre strategy datasets such as `F1-Tyre-Strategy-Engine` projects | `Compound`, `StintLength`, aggregated `AirTemp`, `TrackTemp`, stint summaries | Tabular ML starting point for tyre degradation, stint-length, compound, and environmental feature experiments. |

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

If `uv` is not available, install dependencies with your preferred Python environment
manager using `pyproject.toml` as the source of truth.

## Live Pit-Wall Pipeline

Browser simulator → FastAPI backend → pit-wall dashboard, over WebSockets at 10 Hz.
Architecture, feature specs and build order are in `CLAUDE.md`; message formats are in
`contracts.md`. Tyre temperatures and pressures here are simulated, not real.

- `simulator/` - drivable car in the browser; sends raw sensor frames only.
- `backend/` - features, detectors, alert engine, Tyre Health Index, laps estimate.
- `dashboard/` - displays output frames only.
- `training/` - offline laps model (not implemented yet).

The backend's dependencies are in `requirements.txt` for now (not yet merged into
`pyproject.toml`); it runs on Python 3.11+.

```bash
pip install -r requirements.txt
cd backend && uvicorn main:app --reload --port 8000
```

In a second terminal, from the repository root:

```bash
python -m http.server 5500
```

Open `http://localhost:5500/simulator/` and `http://localhost:5500/dashboard/`.
Backend tests: `cd backend && python -m pytest -q tests`.

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
