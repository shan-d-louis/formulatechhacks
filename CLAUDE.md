# CLAUDE.md

Tire monitoring system for FormulaTech Hacks. 12-hour build, 3-person team. Favor working, demoable code over clever code. When in doubt, pick the simpler option and keep the demo running.

## What we're building

A pit-wall tool that predicts how many laps a set of tires has left and detects four live safety risks, then shows it all on a dashboard.

Features:
1. Laps remaining (ML model trained on historical F1 data, with a low/mid/high band)
2. Lock-up detection
3. Wheelspin detection
4. Overheating detection (predictive, 10 s forecast)
5. Pressure anomaly detection (catches slow punctures)
6. Tire Health Index (THI): one 0–100 score per tire combining all of the above

## Architecture

```
simulator (browser)  --raw frames, 10 Hz, WebSocket-->  backend (FastAPI)  --output frames-->  dashboard (browser)
                                                              ^
                                     training/ (offline) --laps_model.joblib
```

- `simulator/` Drivable car in the browser. Physics and controls only. Sends raw frames. Never computes scores or alerts.
- `backend/` Receives raw frames, computes features, runs detectors and the laps model, builds the THI, runs the alert engine, broadcasts output frames.
- `dashboard/` Displays output frames only. No business logic.
- `training/` Offline. Pulls FastF1 data, trains the laps model, saves `laps_model.joblib`. Runs before the demo, not live.

Two paths: the offline path (training) runs once; the live path (everything else) runs continuously. Only the laps model uses both.

## Repo layout

```
simulator/index.html        # drivable sim (started from tire-sim.html)
simulator/reference.html    # original all-in-one sim, DO NOT DELETE (demo fallback + reference logic)
backend/main.py             # FastAPI app, WebSocket endpoints
backend/features.py         # slip ratios, temp slopes, pressure residuals
backend/detectors.py        # lockup, wheelspin, overheat, pressure
backend/health.py           # THI aggregator
backend/alerts.py           # severity, dedupe, alert log
backend/laps.py             # loads model, predict_laps(); fallback if model missing
backend/state.py            # TireState per corner
backend/config.py           # ALL thresholds and constants live here
dashboard/index.html        # pit wall screen
training/fetch.py           # FastF1 download (cache enabled)
training/prepare.py         # clean laps, fuel-correct, build delta target
training/train.py           # baseline + LightGBM, grouped CV, saves model
contracts.md                # message formats (source of truth)
```

## Commands

```bash
# backend
pip install fastapi uvicorn numpy pandas lightgbm scikit-learn joblib fastf1
cd backend && uvicorn main:app --reload --port 8000

# simulator and dashboard (serve over http, not file://)
python -m http.server 5500      # then open /simulator/ and /dashboard/

# training
cd training && python fetch.py && python prepare.py && python train.py
```

## Stack and constraints

- Python 3.11, FastAPI, plain WebSockets. No Kafka, MQTT, Redis, or databases. In-memory state only.
- Frontend: plain HTML/CSS/JS in single files. No build step, no framework unless the team asks for one.
- WebSocket endpoints: `/ws/sim` (simulator sends raw frames in), `/ws/dash` (dashboards receive output frames).
- Frame rate: 10 Hz. Backend uses `DT = 0.1`.
- Keep the backend stateless across restarts; per-tire state lives in `TireState` objects in memory.

## Data contracts

Change these only if the whole team agrees. Update `contracts.md` in the same commit.

### Raw frame (simulator → backend)

Sensor readings only. No scores, flags, or alerts.

```json
{
  "t": 12.3,
  "lap": 2,
  "compound": "MEDIUM",
  "tire_age_laps": 1.4,
  "speed_kph": 241.0,
  "throttle": 1.0,
  "brake": 0.0,
  "steer": 0.1,
  "track_temp_c": 35,
  "air_temp_c": 24,
  "tires": {
    "FL": {"wheel_speed_kph": 240.1, "temp_c": 96.2, "pressure_psi": 21.2},
    "FR": {"wheel_speed_kph": 240.3, "temp_c": 97.0, "pressure_psi": 21.3},
    "RL": {"wheel_speed_kph": 242.0, "temp_c": 94.8, "pressure_psi": 21.1},
    "RR": {"wheel_speed_kph": 242.4, "temp_c": 95.5, "pressure_psi": 21.1}
  }
}
```

Throttle, brake: 0 to 1. Steer: -1 (left) to 1 (right). Tire keys are always `FL`, `FR`, `RL`, `RR`.

### Output frame (backend → dashboard)

```json
{
  "timestamp": 12.3,
  "lap": 2,
  "car": {"speed_kph": 241.0, "throttle": 1.0, "brake": 0.0, "steer": 0.1},
  "laps_remaining": {"low": 8.1, "mid": 10.4, "high": 12.7},
  "tires": {
    "FL": {
      "thi": 91,
      "status": "ok",
      "dominant": "wear",
      "components": {"wear": 88, "thermal": 97, "pressure": 100, "damage": 100},
      "temp_c": 96.2,
      "pressure_psi": 21.2,
      "pressure_residual": -0.05,
      "slip_ratio": -0.004,
      "flags": {"lockup": false, "wheelspin": false, "overheat": "none", "pressure": "none"}
    }
  },
  "alerts": [{"severity": "warn", "tire": "RL", "message": "Pressure anomaly: 0.7 psi below expected", "lap": 2, "t": 11.8}]
}
```

`status`: `ok` | `warn` | `bad`. `overheat` and `pressure` flags: `none` | `warning` | `critical`. `alerts` holds the most recent alerts, newest first.

## Feature specs

All numbers below go in `backend/config.py`. They match `simulator/reference.html`, which is the working reference implementation; port logic from its `detect()` and `health()` functions when in doubt.

### Shared features (`features.py`)

- Slip ratio per tire: `(wheel_speed - car_speed) / max(car_speed, 1)`. Negative = locking, positive = spinning.
- Temperature slope per tire: exponentially smoothed °C/s (time constant about 2 s).
- Expected pressure: `P_COLD * (temp_c + 273) / (T_COLD + 273)` with `P_COLD = 20.5` psi, `T_COLD = 85` °C.
- Pressure residual: `actual - expected`.

### Lock-up (front tires)

- Condition: slip < -0.15 AND brake > 0.15 AND speed > 18 kph.
- Must hold for 0.1 s before alerting.
- Severity: peak |slip| > 0.4 → `bad`, else `warn`.
- On event end, update the alert with duration and peak slip. Add damage penalty while active: `|slip| * speed_mps * dt * 0.25`. Damage never recovers.

### Wheelspin (rear tires)

- Condition: slip > 0.15 AND throttle > 0.2. Hold 0.1 s.
- Severity: peak slip > 0.4 → `bad`, else `warn`.
- Damage penalty while active: `slip * speed_mps * dt * 0.12`.

### Overheating

- Temperature window 90–110 °C, hard limit 118 °C.
- Forecast: `temp + slope * 10`.
- `critical` if temp > 118. `warning` if temp > 112 or forecast > 118. Clear to `none` only when temp < 108 and forecast < 115 (hysteresis).
- Heat damage accumulates while above 118: `(temp - 118) * dt * 0.15`.

### Pressure anomaly

- `critical` if residual < -1.5 psi. `warning` if residual < -0.6. Clear when residual > -0.4.
- The point is catching a leak even while absolute pressure still looks normal, because a hot tire should read higher.

### Tire Health Index (`health.py`)

Components, each clamped to 1–100:
- `wear = 100 * (1 - life_used)`, where `life_used = age / (age + laps_remaining_mid)` from the laps model.
- `thermal = 100 - 3.3 * degrees_outside_window - heat_damage`, smoothed (time constant 1.5 s).
- `pressure = 100 - 45 * max(0, -residual - 0.2) - 12 * max(0, |pressure - 21.4| - 1.5)`, smoothed.
- `damage = 100 - accumulated_penalties`, not smoothed.

Combine with a weighted geometric mean, weights wear 0.4, thermal 0.2, pressure 0.2, damage 0.2:
`thi = 100 * prod((c / 100) ** w)`.

Rules:
- Never use a plain weighted average; one critical problem must not be averaged away.
- Cap THI at 30 if the tire has any `critical` overheat or pressure state.
- Status bands with hysteresis: enter `warn` below 78, return to `ok` above 82; enter `bad` below 48, leave `bad` above 52.
- `dominant` = the lowest component. The dashboard shows it as the reason.

### Laps remaining (`laps.py` + `training/`)

Do not train on stint length directly; stints end for strategy reasons, so those labels are censored. Instead:
1. Target: fuel-corrected lap-time delta vs. the stint's baseline (median of laps 2–4). Fuel correction ≈ 0.03 s/kg, ≈ 1.7 kg/lap.
2. Drop in-laps, out-laps, safety car/VSC laps (`TrackStatus`), and laps where `IsAccurate` is false. Dry races only.
3. Baseline model: per-compound curve (linear + quadratic term). Ship this first.
4. Main model: LightGBM with `monotone_constraints` +1 on tire age, three quantile models (0.1, 0.5, 0.9).
5. Validate with grouped CV by race (never random lap splits). Report MAE and 80% interval coverage vs. the baseline.
6. Live: roll the model forward from current age; laps remaining = first future lap where predicted delta > 1.5 s.

`laps.py` must expose `predict_laps(compound, tire_age_laps, track_temp_c) -> {"low", "mid", "high"}`. If `laps_model.joblib` is missing, fall back to the baseline curve, and if that's missing too, return a fixed estimate. The live system must never crash because the model isn't ready.

## Alert engine (`alerts.py`)

- One alert per event, created when the event starts, updated in place while it continues, finalized when it ends. No repeated alerts every frame.
- Keep the last 40 alerts in memory.
- Messages are plain and actionable: name the tire, what happened, and the key number. Example: "Pressure critical: 1.8 psi below expected. Box this lap."

## Build priorities

Build in this order and don't skip ahead:
1. End-to-end pass-through: simulator → backend → dashboard with `process()` returning placeholder data. This is the most important milestone.
2. Detectors in this order: lock-up, wheelspin, pressure, overheating.
3. THI aggregator.
4. Laps model: fake → baseline → LightGBM.
5. Polish.

If behind schedule, cut in this order: live residual correction, THI history chart, tire detail panel. Never cut scenario injection or the per-corner car view.

## Coding conventions

- All thresholds and weights in `config.py`. No magic numbers in logic files.
- Pure functions where possible; state only in `TireState`.
- Type hints on public functions. Short docstrings.
- Round numbers before sending to the dashboard (1 decimal for temps and pressures, integers for THI).
- Log to stdout, not files.
- Write a small test per detector in `backend/tests/` that feeds hand-built frames and checks the flag. Keep tests fast.

## Demo scenarios

The simulator must support these on demand (buttons or keys). Every change should keep all of them working:
- Hard braking at high speed → front lock-up
- Full throttle from low speed → rear wheelspin
- Sustained cornering at speed → outside tires overheat
- Slow puncture on a chosen tire → pressure anomaly within about 10 s
- Fit new tires → state resets

## Don't

- Don't add infrastructure (queues, databases, Docker, auth).
- Don't put detection logic in the simulator or dashboard.
- Don't change the data contracts without updating `contracts.md`.
- Don't delete or break `simulator/reference.html`. It's the fallback if the backend fails on stage.
- Don't claim simulated sensor data is real. Tire temps and pressures are simulated; lap data is real (FastF1).