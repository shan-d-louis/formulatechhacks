# Data contracts

Source of truth for messages between simulator, backend and dashboard. Change only if the whole team agrees, in the same commit as the code change.

## Transport

- Simulator → backend: WebSocket `ws://<host>:8000/ws/sim`, one JSON raw frame per message, 10 Hz.
- Backend → dashboards: WebSocket `ws://<host>:8000/ws/dash`, one JSON output frame per raw frame received. A newly connected dashboard immediately receives the latest output frame, if any.
- Both pages accept `?backend=host:port` to point at a backend other than `<page host>:8000`.
- A malformed raw frame is logged and dropped; it never closes the stream.

## Raw frame (simulator → backend)

Sensor readings only. No scores, flags, or alerts.

```json
{
  "t": 12.3,
  "lap": 2,
  "compound": "MEDIUM",
  "tire_age_laps": 1.4,
  "stint_id": 3,
  "demo_speed": 1,
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

- `throttle`, `brake`: 0 to 1. `steer`: -1 (left) to 1 (right).
- `compound`: `SOFT` | `MEDIUM` | `HARD`.
- `stint_id`: increases by one each time a set of tires is fitted (new or used). The backend resets all tire state (damage, heat damage, alert states, smoothing) when it changes, and announces the new set with an `info` alert. If absent, the backend falls back to treating a drop in `tire_age_laps` or a compound change as a new set.
- `tire_age_laps`: may jump up within a stint (the demo's "+5 laps"); that keeps tire history and only changes the laps estimate and wear.
- `demo_speed`: 1, 5 or 10. How many times faster than real tire age grows in the demo. It never changes the car's physics or speed. Defaults to 1 if absent.
- Tire keys are always `FL`, `FR`, `RL`, `RR`.

## Output frame (backend → dashboard)

```json
{
  "timestamp": 12.3,
  "lap": 2,
  "car": {"speed_kph": 241.0, "throttle": 1.0, "brake": 0.0, "steer": 0.1},
  "laps_remaining": {"low": 8.1, "mid": 10.4, "high": 12.7},
  "danger": {"low": 3.2, "mid": 6.9, "high": 9.8, "tire": "RL", "now": false, "thi_below": 48},
  "stint": {"id": 3, "compound": "MEDIUM", "tire_age_laps": 1.4, "demo_speed": 1},
  "tires": {
    "FL": {
      "thi": 91,
      "status": "ok",
      "dominant": "wear",
      "capped": false,
      "components": {"wear": 88, "thermal": 97, "pressure": 100, "damage": 100},
      "temp_c": 96.2,
      "pressure_psi": 21.2,
      "pressure_residual": -0.05,
      "slip_ratio": -0.004,
      "flags": {"lockup": false, "wheelspin": false, "overheat": "none", "pressure": "none"},
      "laps_to_danger": 7.4
    }
  },
  "alerts": [{"severity": "warn", "tire": "RL", "message": "Pressure anomaly: 0.7 psi below expected", "lap": 2, "t": 11.8, "pinned": false, "kind": "pressure"}]
}
```

- `laps_remaining`: laps until the pace cliff (predicted lap time 1.5 s slower than the stint's start), from the laps model. Also drives the wear component.
- `danger`: laps until the worst tire enters the danger zone, i.e. its THI falls below `thi_below` (the red band). Wear is projected forward with the laps model; the other components count only lasting harm (heat damage, air lost, lock-up/wheelspin damage), so a cold tire after stopping, or a briefly hot one, does not move it. `low`/`mid`/`high` use the low/mid/high laps estimates. `tire` is the tire that gets there first. `now` is true (and all three are 0) when that tire has a critical overheat or pressure state, or its lasting harm plus wear already puts it below `thi_below`. The dashboard's laps panel shows this.
- `tires.*.laps_to_danger`: the same `mid` figure per tire.
- `stint`: the current stint, echoed from the raw frame so the dashboard can show tire age and demo speed.
- `tires` contains all four corners, same keys as the raw frame.
- `status`: `ok` | `warn` | `bad`.
- `dominant`: what limits the tire. `pressure` or `thermal` when a critical state caps the THI (`pressure` if both), otherwise the lowest of the four components.
- `capped`: true while a critical overheat or pressure state caps THI at 30.
- `flags.overheat`, `flags.pressure`: `none` | `warning` | `critical`.
- `alerts`: most recent alerts, newest first (max 40). `severity`: `info` | `warn` | `bad` | `critical`. `info` marks a problem clearing (e.g. a tire back in its temperature window) or a new stint. `tire` is a corner, or `ALL` for alerts about the whole set (e.g. "New stint: used MEDIUM tires, 20 laps old."). `kind`: `lockup` | `wheelspin` | `overheat` | `pressure` | `stint` (what the alert is about). Critical alerts also carry `title` and `actions` (a list of immediate steps for the crew), taken from `backend/critical_actions.json`; edit that file to change them, and a running backend picks up the change on the next frame. The dashboard shows critical alerts in their own "Critical alerts" panel with these actions, and everything else in "Feedback". Critical alerts come from four kinds: `lockup` (slip above 0.4 held 1 s), `wheelspin` (above 0.4 held 0.5 s), `overheat` (above 118 °C) and `pressure` (more than 1.5 psi below expected). A critical lock-up or wheelspin stays pinned for 30 s after it ends, or until new tires. `pinned` is true for an active critical alert; pinned alerts come first and are never evicted by the 40-alert cap. Repeated lock-ups or wheelspins on one tire within a lap share one entry ("FL lock-up x3 this lap: ...").
- Rounding: 1 decimal for temps and pressures, integers for THI and components.
