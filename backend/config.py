"""All thresholds, weights and constants. No magic numbers in logic files."""

# ---------- Timing ----------
DT = 0.1  # s, frame period at 10 Hz

# ---------- Tires ----------
CORNERS = ("FL", "FR", "RL", "RR")  # tire keys, in display order
FRONTS = ("FL", "FR")  # checked for lock-up
REARS = ("RL", "RR")  # checked for wheelspin

# ---------- Shared features (features.py) ----------
SLIP_MIN_SPEED_KPH = 1.0  # denominator floor for slip ratio, avoids divide-by-zero at standstill
TEMP_SLOPE_TAU_S = 2.0  # time constant of the exponentially smoothed temp slope, s
P_COLD_PSI = 20.5  # reference pressure of a healthy tire at T_COLD_C
T_COLD_C = 85.0  # reference temperature for expected pressure, °C
KELVIN_OFFSET = 273.0  # °C → K for the gas-law pressure estimate

# ---------- New-tire detection (main.py) ----------
NEW_TIRE_AGE_DROP_LAPS = 0.05  # (frames without stint_id) tire_age_laps falling by more than this = new set
USED_TIRE_MIN_AGE_LAPS = 0.5  # a stint starting at this tire age or more is announced as used tires

# ---------- Lock-up (front tires) ----------
LOCKUP_SLIP = -0.15  # slip below this counts as locking
LOCKUP_MIN_BRAKE = 0.15  # brake input must exceed this
LOCKUP_MIN_SPEED_KPH = 18.0  # ignore lock-ups below this speed
LOCKUP_HOLD_S = 0.1  # condition must hold this long before alerting
LOCKUP_BAD_SLIP = 0.4  # peak |slip| above this → severity "bad", else "warn"
LOCKUP_DAMAGE_K = 0.25  # damage per frame while active: |slip| * speed_mps * dt * K

# ---------- Wheelspin (rear tires) ----------
WHEELSPIN_SLIP = 0.15  # slip above this counts as spinning
WHEELSPIN_MIN_THROTTLE = 0.2  # throttle input must exceed this
WHEELSPIN_HOLD_S = 0.1  # condition must hold this long before alerting
WHEELSPIN_BAD_SLIP = 0.4  # peak slip above this → severity "bad", else "warn"
WHEELSPIN_DAMAGE_K = 0.12  # damage per frame while active: slip * speed_mps * dt * K

# ---------- Overheating ----------
TEMP_WINDOW_LOW_C = 90.0  # lower edge of the working temperature window
TEMP_WINDOW_HIGH_C = 110.0  # upper edge of the working temperature window
TEMP_HARD_LIMIT_C = 118.0  # above this → "critical"; heat damage accrues
OVERHEAT_FORECAST_S = 10.0  # forecast horizon: temp + slope * this
OVERHEAT_WARN_TEMP_C = 112.0  # current temp above this → "warning"
OVERHEAT_WARN_FORECAST_C = 118.0  # forecast temp above this → "warning"
OVERHEAT_CLEAR_TEMP_C = 108.0  # clear to "none" only when temp is below this...
OVERHEAT_CLEAR_FORECAST_C = 115.0  # ...and forecast is below this (hysteresis)
OVERHEAT_CRIT_CLEAR_C = 116.0  # critical drops back to warning only below this (hysteresis around the limit)
HEAT_DAMAGE_K = 0.15  # damage per frame above limit: (temp - limit) * dt * K

# ---------- Pressure anomaly ----------
PRESSURE_CRIT_RESIDUAL = -1.5  # residual below this → "critical", psi
PRESSURE_WARN_RESIDUAL = -0.6  # residual below this → "warning", psi
PRESSURE_CLEAR_RESIDUAL = -0.4  # clear to "none" when residual rises above this, psi
PRESSURE_CRIT_CLEAR_RESIDUAL = -1.2  # critical steps down to warning only above this (hysteresis), psi

# ---------- Tire Health Index (health.py) ----------
COMPONENT_MIN = 1  # each component clamped to [COMPONENT_MIN, COMPONENT_MAX]
COMPONENT_MAX = 100
THERMAL_PER_DEG = 3.3  # thermal loses this per °C outside the working window
THERMAL_TAU_S = 1.5  # smoothing time constant for thermal component, s
PRESSURE_RESID_K = 45.0  # pressure component loss per psi of residual beyond deadband
PRESSURE_RESID_DEADBAND = 0.2  # psi of negative residual tolerated before penalty
PRESSURE_NOMINAL_PSI = 21.4  # nominal hot running pressure
PRESSURE_ABS_DEADBAND = 1.5  # psi from nominal tolerated before penalty
PRESSURE_ABS_K = 12.0  # pressure component loss per psi beyond absolute deadband
PRESSURE_TAU_S = 1.5  # smoothing time constant for pressure component, s
THI_WEIGHTS = {"wear": 0.4, "thermal": 0.2, "pressure": 0.2, "damage": 0.2}  # weighted geometric mean
THI_CRITICAL_CAP = 30  # THI cap when any overheat/pressure flag is "critical"
STATUS_WARN_ENTER = 78  # THI below this → enter "warn"
STATUS_WARN_EXIT = 82  # THI above this → back to "ok"
STATUS_BAD_ENTER = 48  # THI below this → enter "bad"
STATUS_BAD_EXIT = 52  # THI above this → leave "bad"

# ---------- Laps remaining (laps.py / training) ----------
FUEL_S_PER_KG = 0.03  # lap-time gain per kg of fuel burned, s
FUEL_KG_PER_LAP = 1.7  # fuel burned per lap, kg
BASELINE_LAPS = (2, 4)  # stint laps whose median lap time is the baseline (inclusive)
BASELINE_MIN_LAPS = 2  # clean laps needed inside BASELINE_LAPS, else the stint is dropped
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")  # compounds kept for training
NEUTRALISED_TRACK_STATUS = "4567"  # FastF1 TrackStatus codes: 4 SC, 5 red flag, 6 VSC, 7 VSC ending
DELTA_OUTLIER_S = 5.0  # |delta| above this is a spin/traffic/yellow lap, not tire wear; dropped
STINT_MEDIAN_DELTA_FLOOR_S = -0.5  # stint median delta below this = baseline laps were compromised (traffic/restart); dropped
TRAIN_AGE_QUANTILE = 0.95  # per compound, ignore tire ages above this quantile (rare managed stints bias the curve down)
CLIFF_DELTA_S = 1.5  # laps remaining = first future lap with predicted delta above this
QUANTILES = (0.1, 0.5, 0.9)  # low / mid / high model quantiles
MAX_FORECAST_LAPS = 60  # stop rolling the model forward after this many laps
MODEL_PATH = "../training/laps_model.joblib"  # trained model (training/train.py), relative to backend/
MODEL_AGE_OFFSET_LAPS = 1.0  # model age (FastF1 TyreLife, 1 on a fresh set's first lap) = tire_age_laps + this
LAPS_CACHE_AGE_STEP = 0.05  # predictions are cached per this much tire age (laps); 10 Hz frames reuse them
DEFAULT_TRACK_TEMP_C = 35.0  # used if a frame has no track temp
FALLBACK_LAPS = {"low": 18.0, "mid": 22.0, "high": 26.0}  # used if no model or baseline is available
STUB_LIFE_LAPS = {"SOFT": 20.0, "MEDIUM": 30.0, "HARD": 40.0}  # stub estimate: total stint life per compound
STUB_LIFE_DEFAULT = 30.0  # stub life for an unknown compound
STUB_BAND = 0.2  # stub low/high = mid * (1 -/+ this)

# ---------- Alert engine (alerts.py) ----------
MAX_ALERTS = 40  # alerts kept in memory
