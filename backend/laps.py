"""Laps-remaining model: exposes predict_laps().

Loads training/laps_model.joblib (built by training/train.py) at import. The model
predicts the fuel-corrected lap-time delta vs. stint start for a given tire age;
laps remaining = how far we can roll forward from the current age before the
predicted delta passes CLIFF_DELTA_S.

Fallback chain, so the live system never crashes here:
  1. "lgbm"      per-compound LightGBM quantile models (+ linear tail past the ages seen in training)
  2. "baseline"  per-compound curve from the same file, if LightGBM fails
  3. "stub"      fixed per-compound life estimate, if the file is missing or unreadable
  4. FALLBACK_LAPS if anything else goes wrong
"""

import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("backend.laps")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / config.MODEL_PATH

_bundle: dict | None = None  # the loaded joblib dict, or None if missing
_boosters: dict | None = None  # {compound: [booster_lo, booster_mid, booster_hi]}, or None if unusable
_warned: set[str] = set()  # failure messages already logged, so 10 Hz frames don't spam stdout


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning(msg)


def load_model(path: str | Path | None = None) -> str:
    """(Re)load the model file. Returns the source predict_laps will use: lgbm | baseline | stub."""
    global _bundle, _boosters
    _bundle, _boosters = None, None
    _warned.clear()
    _cached.cache_clear()
    path = Path(path) if path is not None else DEFAULT_MODEL_PATH

    try:
        import joblib
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "baseline" not in bundle:
            raise ValueError("not a laps model bundle")
        _bundle = bundle
    except Exception as e:  # missing, corrupt, or wrong format: fixed estimate only
        log.warning("laps model not loaded from %s (%s: %s); using fixed per-compound estimate",
                    path, type(e).__name__, e)
        return source()

    try:
        import lightgbm as lgb
        _boosters = {
            c: [lgb.Booster(model_str=_bundle["lgbm"][c][q]["model_str"]) for q in _bundle["quantiles"]]
            for c in _bundle["compounds"]
        }
    except Exception as e:  # lightgbm missing or model strings bad: baseline curve still works
        log.warning("LightGBM models unusable (%s: %s); using baseline curve", type(e).__name__, e)

    log.info("laps model loaded from %s: source=%s, %s laps / %s races", path, source(),
             _bundle.get("n_laps"), _bundle.get("n_races"))
    return source()


def source() -> str:
    """Which predictor is active: lgbm | baseline | stub."""
    if _bundle is None:
        return "stub"
    return "lgbm" if _boosters is not None else "baseline"


# ---------- delta curves: (n,) model ages -> (n, 3) low/mid/high delta ----------

def _delta_lgbm(compound: str, ages: np.ndarray, track_temp_c: float) -> np.ndarray:
    """Mirror of training/train.py predict_lgbm for one compound."""
    max_age = _bundle["max_age"][compound]
    c2, b, _ = _bundle["baseline"][compound]["coef"]
    slope = max(0.0, 2 * c2 * max_age + b)
    clipped = np.minimum(ages, max_age)
    cols = {"TyreLife": clipped, "TrackTemp": np.full(len(ages), float(track_temp_c))}
    X = np.column_stack([cols[f] for f in _bundle["features"]])
    tail = slope * np.maximum(0.0, ages - max_age)
    preds = []
    for booster, q in zip(_boosters[compound], _bundle["quantiles"]):
        preds.append(booster.predict(X) + _bundle["lgbm"][compound][q]["init_score"] + tail)
    return np.sort(np.column_stack(preds), axis=1)


def _delta_baseline(compound: str, ages: np.ndarray, track_temp_c: float) -> np.ndarray:
    """Per-compound curve, band from its residual quantiles."""
    p = _bundle["baseline"][compound]
    mid = np.polyval(p["coef"], ages)
    return np.column_stack([mid + p["resid_lo"], mid, mid + p["resid_hi"]])


def _roll_forward(delta_fn, compound: str, tire_age_laps: float, track_temp_c: float) -> dict:
    """Laps until each band's delta first passes the cliff, interpolated between whole laps."""
    age0 = tire_age_laps + config.MODEL_AGE_OFFSET_LAPS
    steps = np.arange(0, config.MAX_FORECAST_LAPS + 1, dtype=float)
    delta = delta_fn(compound, age0 + steps, track_temp_c)
    if delta.shape != (len(steps), 3) or not np.all(np.isfinite(delta)):
        raise ValueError(f"bad delta prediction, shape {delta.shape}")

    def laps_until_cliff(d: np.ndarray) -> float:
        above = np.nonzero(d > config.CLIFF_DELTA_S)[0]
        if len(above) == 0:
            return float(config.MAX_FORECAST_LAPS)
        i = above[0]
        if i == 0:
            return 0.0
        frac = (config.CLIFF_DELTA_S - d[i - 1]) / (d[i] - d[i - 1])  # d[i] > cliff >= d[i-1]
        return float(steps[i - 1] + frac)

    # The pessimistic (high delta) band gives the low laps estimate, and vice versa.
    lo, mid, hi = laps_until_cliff(delta[:, 2]), laps_until_cliff(delta[:, 1]), laps_until_cliff(delta[:, 0])
    return {"low": round(lo, 1), "mid": round(mid, 1), "high": round(hi, 1)}


def _stub_estimate(compound: str, tire_age_laps: float) -> dict:
    """Linear life per compound: mid = life - age, with a fixed ± band."""
    life = config.STUB_LIFE_LAPS.get(str(compound).upper(), config.STUB_LIFE_DEFAULT)
    mid = max(0.0, life - max(0.0, tire_age_laps))
    return {
        "low": round(mid * (1 - config.STUB_BAND), 1),
        "mid": round(mid, 1),
        "high": round(mid * (1 + config.STUB_BAND), 1),
    }


@lru_cache(maxsize=512)
def _cached(compound: str, age_key: int, temp_key: int) -> dict:
    """Walk the fallback chain. Cached on quantised inputs; callers get a copy."""
    tire_age_laps = age_key * config.LAPS_CACHE_AGE_STEP
    if _bundle is not None:
        if _boosters is not None:
            try:
                return _roll_forward(_delta_lgbm, compound, tire_age_laps, temp_key)
            except Exception as e:
                _warn_once(f"LightGBM prediction failed for {compound} ({type(e).__name__}: {e}); using baseline")
        try:
            return _roll_forward(_delta_baseline, compound, tire_age_laps, temp_key)
        except Exception as e:
            _warn_once(f"baseline prediction failed for {compound} ({type(e).__name__}: {e}); using stub")
    return _stub_estimate(compound, tire_age_laps)


def predict_laps(compound: str, tire_age_laps: float, track_temp_c: float) -> dict:
    """Laps remaining as {"low", "mid", "high"}. Never raises."""
    try:
        age = float(tire_age_laps)
        temp = float(track_temp_c) if track_temp_c is not None else config.DEFAULT_TRACK_TEMP_C
        if not (math.isfinite(age) and math.isfinite(temp)):  # before max(): max(0.0, nan) is 0.0
            raise ValueError(f"non-finite input age={tire_age_laps!r} temp={track_temp_c!r}")
        age = max(0.0, age)
        key = (str(compound).upper(), round(age / config.LAPS_CACHE_AGE_STEP), round(temp))
        return dict(_cached(*key))
    except Exception as e:  # never let the laps model take down the live path
        _warn_once(f"predict_laps failed, using fallback: {type(e).__name__}: {e}")
        return dict(config.FALLBACK_LAPS)


load_model()
