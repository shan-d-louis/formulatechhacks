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

## Run it
Setup:
```bash
python -m venv .venv
```
```bash
.venv/Scripts/python -m pip install -r requirements.txt
```
Data and training (order matters):
```bash
.venv/Scripts/python -m sidewall.data.ingest_fastf1 --telemetry
```
```bash
.venv/Scripts/python -m sidewall.data.ingest_acgym
```
```bash
.venv/Scripts/python -m sidewall.data.build_stints
```
```bash
.venv/Scripts/python -m sidewall.models.event_detectors --rebuild
```
```bash
.venv/Scripts/python -m sidewall.twin.virtual_tpms
```
```bash
.venv/Scripts/python -m sidewall.models.tyre_life
```
```bash
.venv/Scripts/python -m sidewall.data.build_atlas
```
Serve (phones on the same Wi-Fi scan the QR codes):
```bash
.venv/Scripts/python -m sidewall.server.app
```
Open http://localhost:8000. `/crew` is the pit-crew phone, `/driver` the phone controller and `/atlas` the Ollon insights.

Tests:
```bash
.venv/Scripts/python -m pytest -q tests
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
