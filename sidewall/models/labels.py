"""Physics-rule event labels computed from high-rate simulator data (per-wheel truth).

These labels are only ever used as training TARGETS. The model's inputs come from the shared
feature contract, which contains nothing a real F1 feed doesn't have.
"""
import numpy as np
import pandas as pd

WHEELS = ("fl", "fr", "rl", "rr")

# Thresholds (see plan: research on slip-ratio peaks and thermal windows).
LOCK_KAPPA = -0.20       # slip ratio past the braking peak
LOCK_MIN_S = 0.08
SPIN_KAPPA = 0.15        # driven-wheel slip past the traction peak
SPIN_MIN_S = 0.10
SPIN_THROTTLE = 0.5
HOT_SURFACE_C = 125.0    # Dallara F317 surface temperature; top few % of the AC Gym distribution
HOT_MIN_S = 1.0
COLD_CORE_C = 50.0       # core below the working window while the tyre is being asked for grip
MIN_SPEED_KMH = 30.0

EVENTS = ("lockup", "wheelspin", "overheat", "cold")


def _sustained(mask: np.ndarray, n: int) -> np.ndarray:
    """True for samples inside runs of `mask` at least `n` samples long."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return mask
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    out = np.zeros_like(mask)
    for a, b in zip(starts, ends):
        if b - a >= n:
            out[a:b] = True
    return out


def sim_labels(df: pd.DataFrame, hz: float, throttle_scale: float = 1.0, brake_scale: float = 1.0) -> pd.DataFrame:
    """Per-sample event flags + which wheel. `df` needs kappa_*, t_core_*, t_in/mid/out_*, speed, throttle."""
    speed = df["speed_kmh"].values
    moving = speed > MIN_SPEED_KMH
    kappa = np.column_stack([df[f"kappa_{w}"].values for w in WHEELS])
    thr = df["throttle"].values / throttle_scale
    brk = df["brake"].values / brake_scale

    lock_any = (np.nanmin(kappa, axis=1) < LOCK_KAPPA) & moving
    rear = kappa[:, 2:]
    spin_any = (np.nanmax(rear, axis=1) > SPIN_KAPPA) & (thr > SPIN_THROTTLE) & (speed > 15)

    surf_cols = [c for c in df.columns if c.startswith(("t_in_", "t_mid_", "t_out_"))]
    core_cols = [f"t_core_{w}" for w in WHEELS if f"t_core_{w}" in df]
    surface = df[surf_cols].max(axis=1).values if surf_cols else np.full(len(df), np.nan)
    core_min = df[core_cols].min(axis=1).values if core_cols else np.full(len(df), np.nan)
    hot_any = surface > HOT_SURFACE_C
    demand = (brk > 0.1) | (thr > 0.9)
    cold_any = (core_min < COLD_CORE_C) & demand & (speed > 80)

    out = pd.DataFrame({"t": df["t"].values})
    out["lockup"] = _sustained(lock_any, int(LOCK_MIN_S * hz))
    out["wheelspin"] = _sustained(spin_any, int(SPIN_MIN_S * hz))
    out["overheat"] = _sustained(hot_any, int(HOT_MIN_S * hz))
    out["cold"] = cold_any
    # Most-affected wheel (for the dashboard's per-tyre cards).
    out["lock_wheel"] = np.array(WHEELS)[np.nanargmin(np.nan_to_num(kappa, nan=0), axis=1)]
    out["spin_wheel"] = np.array(WHEELS[2:])[np.nanargmax(np.nan_to_num(rear, nan=0), axis=1)]
    return out


def to_grid(labels: pd.DataFrame, grid_t: np.ndarray, back_s: float, ahead_s: float) -> pd.DataFrame:
    """For each grid time t, flag an event if it occurs anywhere in (t - back_s, t + ahead_s].

    back_s ~ one grid step gives detection of what just happened; ahead_s > 0 turns the target into
    an early warning ("this will happen within the next ahead_s seconds").
    """
    lt = labels["t"].values
    out = {"t": grid_t}
    for ev in EVENTS:
        flag = labels[ev].values.astype(int)
        csum = np.r_[0, np.cumsum(flag)]
        lo = np.searchsorted(lt, grid_t - back_s, side="right")
        hi = np.searchsorted(lt, grid_t + ahead_s, side="right")
        out[ev] = (csum[hi] - csum[lo]) > 0
        # Onset marker: first grid step of each event (for lead-time evaluation).
        now_lo = np.searchsorted(lt, grid_t - back_s, side="right")
        now_hi = np.searchsorted(lt, grid_t, side="right")
        out[f"{ev}_now"] = (csum[now_hi] - csum[now_lo]) > 0
    return pd.DataFrame(out)
