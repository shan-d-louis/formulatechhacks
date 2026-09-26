"""The feature contract shared by training and live inference.

Every source (FastF1 replay, Assetto Corsa sims, the phone-driven demo sim) is first converted to a
`CarStream` with these columns:

    t          seconds
    speed_kmh  vehicle speed
    throttle   0..1
    brake      0 or 1   (F1 public telemetry only has an on/off brake flag, so everything is binarised)
    gear       int
    rpm        engine rpm
    x, y       horizontal position in metres

`make_features` resamples to the F1 rate (4 Hz) and derives only causal features, i.e. each row uses
data up to and including its own timestamp, so the exact same code runs on a live stream.
"""
import numpy as np
import pandas as pd

RATE_HZ = 4.0
G = 9.81
STREAM_COLS = ["t", "speed_kmh", "throttle", "brake", "gear", "rpm", "x", "y"]


# ---------------------------------------------------------------- adapters

def stream_from_fastf1(tel: pd.DataFrame) -> pd.DataFrame:
    """FastF1 car+position data (as saved by ingest_fastf1.telemetry_frame) for ONE driver."""
    return pd.DataFrame({
        "t": tel["t"].values,
        "speed_kmh": tel["Speed"].values.astype(float),
        "throttle": np.clip(tel["Throttle"].values.astype(float), 0, 100) / 100.0,
        "brake": tel["Brake"].astype(float).values,
        "gear": tel["nGear"].values.astype(float),
        "rpm": tel["RPM"].values.astype(float),
        "x": tel["X"].values / 10.0,   # FastF1 positions are in decimetres
        "y": tel["Y"].values / 10.0,
    })


def stream_from_acgym(df: pd.DataFrame) -> pd.DataFrame:
    """Assetto Corsa Gym MoTeC data (the MoTeC export uses x/y as the ground plane, z as elevation)."""
    return pd.DataFrame({
        "t": df["t"].values,
        "speed_kmh": df["speed_kmh"].values,
        "throttle": df["throttle"].values / 100.0,
        "brake": (df["brake"].values > 10.0).astype(float),
        "gear": df["gear"].values,
        "rpm": df["rpm"].values,
        "x": df["x"].values,
        "y": df["y"].values,
    })


def stream_from_thulab(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "t": df["timestamp"].values - df["timestamp"].values[0],
        "speed_kmh": df["speed_kmh"].values,
        "throttle": df["throttle"].values,
        "brake": (df["brake"].values > 0.1).astype(float),
        "gear": df["gear"].values,
        "rpm": df["rpms"].values,
        "x": df["pos_x"].values,
        "y": df["pos_z"].values,
    })


# ---------------------------------------------------------------- resampling

def resample(stream: pd.DataFrame, rate_hz: float = RATE_HZ, jitter: float = 0.0,
             rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Resample a stream to `rate_hz`. `jitter` (0..1) makes the timestamps irregular like F1 telemetry."""
    s = stream.dropna(subset=["t"]).sort_values("t")
    t0, t1 = s["t"].iloc[0], s["t"].iloc[-1]
    step = 1.0 / rate_hz
    grid = np.arange(t0, t1, step)
    if jitter and rng is not None:
        grid = np.sort(grid + rng.uniform(-jitter, jitter, len(grid)) * step)
    out = {"t": grid}
    for c in ("speed_kmh", "throttle", "rpm", "x", "y", "tyre_age_s"):
        if c in s:
            out[c] = np.interp(grid, s["t"].values, s[c].values)
    # Step-like channels: take the latest value at or before each grid time.
    idx = np.clip(np.searchsorted(s["t"].values, grid, side="right") - 1, 0, len(s) - 1)
    out["gear"] = s["gear"].values[idx]
    # Brake: on if it was on at any point since the previous grid sample (F1 flag semantics).
    b = s["brake"].values
    prev = np.r_[idx[0], idx[:-1]]
    csum = np.r_[0, np.cumsum(b > 0.5)]
    out["brake"] = ((csum[idx + 1] - csum[prev]) > 0).astype(float)
    return pd.DataFrame(out)


def augment(stream: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Make clean sim data look like F1 public telemetry: noise, quantisation, small lags, dropouts."""
    s = stream.copy()
    n = len(s)
    s["speed_kmh"] = np.round(s["speed_kmh"] + rng.normal(0, 1.0, n))
    s["throttle"] = np.clip(np.round(s["throttle"] * 100 + rng.normal(0, 1.5, n)), 0, 100) / 100
    s["rpm"] = s["rpm"] + rng.normal(0, 40, n)
    s["x"] = s["x"] + rng.normal(0, 0.3, n)
    s["y"] = s["y"] + rng.normal(0, 0.3, n)
    lag = int(rng.integers(0, 2))
    if lag:
        s["throttle"] = s["throttle"].shift(lag).bfill()
    drop = rng.random(n) < 0.02
    s = s[~drop]
    return s.reset_index(drop=True)


# ---------------------------------------------------------------- features

def _ewm(x: pd.Series, span: float) -> pd.Series:
    return x.ewm(span=span, adjust=False).mean()


def make_features(stream: pd.DataFrame, already_resampled: bool = False) -> pd.DataFrame:
    """Causal features at the stream's own timestamps (resampled to 4 Hz unless told otherwise)."""
    s = stream if already_resampled else resample(stream)
    s = s.reset_index(drop=True)
    f = pd.DataFrame({"t": s["t"]})
    dt = s["t"].diff().clip(lower=1e-3).fillna(1.0 / RATE_HZ)
    v = s["speed_kmh"] / 3.6

    f["speed"] = s["speed_kmh"]
    f["throttle"] = s["throttle"]
    f["brake"] = s["brake"]
    f["gear"] = s["gear"]
    f["rpm"] = s["rpm"]

    # Longitudinal and lateral acceleration (g), lightly smoothed with causal filters.
    ax = (v.diff() / dt).fillna(0) / G
    f["ax"] = _ewm(ax, 2)
    heading = np.unwrap(np.arctan2(s["y"].diff().fillna(0), s["x"].diff().fillna(0)))
    yaw_rate = pd.Series(np.gradient(heading), index=s.index) / dt
    yaw_rate = yaw_rate.where(v > 5, 0).clip(-3, 3)
    f["yaw_rate"] = _ewm(yaw_rate, 2)
    f["ay"] = (v * f["yaw_rate"] / G).clip(-8, 8)
    f["g_comb"] = np.hypot(f["ax"], f["ay"])

    # Wheelspin proxy: engine speed vs road speed, relative to what this gear normally gives.
    ratio = (s["rpm"] / s["speed_kmh"].clip(lower=20)).where(s["speed_kmh"] > 20)
    base = ratio.groupby(s["gear"]).transform(lambda r: r.expanding().median())
    f["rpm_ratio_rel"] = (ratio / base).fillna(1.0).clip(0.5, 2.0)

    # Rates of change.
    for c in ("speed", "throttle", "rpm", "rpm_ratio_rel"):
        f[f"d_{c}"] = (f[c].diff() / dt).fillna(0)

    # Causal rolling windows (1 s and 2 s at 4 Hz).
    for w in (4, 8):
        for c in ("ax", "ay", "throttle", "rpm_ratio_rel", "speed", "d_speed"):
            r = f[c].rolling(w, min_periods=1)
            f[f"{c}_mean{w}"] = r.mean()
            f[f"{c}_min{w}"] = r.min()
            f[f"{c}_max{w}"] = r.max()
        f[f"brake_frac{w}"] = f["brake"].rolling(w, min_periods=1).mean()

    # Phase-of-corner timers.
    f["brake_dur"] = _run_length(f["brake"] > 0.5, dt)
    f["full_throttle_dur"] = _run_length(f["throttle"] > 0.95, dt)
    f["since_brake_release"] = _run_length(f["brake"] < 0.5, dt).clip(upper=10)

    # Energy-style load proxies.
    f["p_lon"] = f["ax"].abs() * v
    f["p_lat"] = f["ay"].abs() * v
    f["brake_x_speed"] = f["brake"] * f["speed"]
    f["throttle_x_ay"] = f["throttle"] * f["ay"].abs()

    # Thermal memory: tyre temperature integrates load over tens of seconds to minutes.
    # Signed lateral power separates left- from right-hand corners (which load opposite tyres), and
    # braking vs traction power separates front- from rear-axle work.
    ay_signed = f["ay"] * v
    f["p_left_turn"] = ay_signed.clip(lower=0)
    f["p_right_turn"] = (-ay_signed).clip(lower=0)
    f["p_brake"] = (-f["ax"]).clip(lower=0) * v
    f["p_drive"] = f["ax"].clip(lower=0) * v
    for span_s in (10, 60, 180):
        span = span_s * RATE_HZ
        f[f"p_lat_ewm{span_s}"] = _ewm(f["p_lat"], span)
        f[f"p_lon_ewm{span_s}"] = _ewm(f["p_lon"], span)
        f[f"speed_ewm{span_s}"] = _ewm(f["speed"], span)
        for c in ("p_left_turn", "p_right_turn", "p_brake", "p_drive"):
            f[f"{c}_ewm{span_s}"] = _ewm(f[c], span)
    # Seconds since these tyres were fitted (streams may supply it; otherwise time since stream start).
    age = s["tyre_age_s"] if "tyre_age_s" in s else s["t"] - s["t"].iloc[0]
    f["tyre_age_s"] = age.values
    return f


def _run_length(mask: pd.Series, dt: pd.Series) -> pd.Series:
    """Seconds the boolean mask has been continuously true."""
    grp = (~mask).cumsum()
    return (dt.where(mask, 0)).groupby(grp).cumsum()


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c != "t"]
