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
- Tire keys are always `FL`, `FR`, `RL`, `RR`.

## Output frame (backend → dashboard)

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

- `tires` contains all four corners, same keys as the raw frame.
- `status`: `ok` | `warn` | `bad`.
- `flags.overheat`, `flags.pressure`: `none` | `warning` | `critical`.
- `alerts`: most recent alerts, newest first (max 40). `severity`: `warn` | `bad` | `critical`.
- Rounding: 1 decimal for temps and pressures, integers for THI and components.
