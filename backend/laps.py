"""Laps-remaining model: exposes predict_laps().
Loads the trained Tier B tyre-life model from models/weights first. The legacy
training/laps_model.joblib bundle is kept as a fallback. The legacy model predicts
the fuel-corrected lap-time delta vs. stint start for a given tire age; laps
remaining = how far we can roll forward from the current age before the predicted
delta passes CLIFF_DELTA_S.

Fallback chain, so the live system never crashes here:
  1. "tierb"     trained Tier B tyre-life model in models/weights
  2. "lgbm"      legacy per-compound LightGBM quantile models
  3. "baseline"  legacy per-compound curve, if LightGBM fails
  4. "stub"      fixed per-compound life estimate, if files are missing or unreadable
  5. FALLBACK_LAPS if anything else goes wrong
"""

import logging
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("backend.laps")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

DEFAULT_TRAINED_MODEL_PATH = Path(__file__).resolve().parent / config.TRAINED_LAPS_MODEL_PATH
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / config.MODEL_PATH

_tierb_bundle: dict | None = None  # the trained models/weights bundle, or None if unavailable
_bundle: dict | None = None  # the loaded joblib dict, or None if missing
_boosters: dict | None = None  # {compound: [booster_lo, booster_mid, booster_hi]}, or None if unusable
_warned: set[str] = set()  # failure messages already logged, so 10 Hz frames don't spam stdout


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning(msg)


def _load_joblib(path: Path) -> dict:
    """Load a joblib bundle from a trusted local path."""
    import joblib
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("not a model bundle")
    return bundle


def _load_tierb(path: Path) -> dict:
    """Load the trained Tier B tyre-life bundle."""
    bundle = _load_joblib(path)
    required = {"models", "features", "fail_features"}
    if not required.issubset(bundle):
        raise ValueError("not a Tier B tyre-life bundle")
    return bundle


def _load_legacy(path: Path) -> tuple[dict, dict | None]:
    """Load the legacy laps bundle and optional LightGBM boosters."""
    bundle = _load_joblib(path)
    if "baseline" not in bundle:
        raise ValueError("not a laps model bundle")

    boosters = None
    try:
        import lightgbm as lgb
        boosters = {
            c: [lgb.Booster(model_str=bundle["lgbm"][c][q]["model_str"]) for q in bundle["quantiles"]]
            for c in bundle["compounds"]
        }
    except Exception as e:  # lightgbm missing or model strings bad: baseline curve still works
        log.warning("LightGBM models unusable (%s: %s); using baseline curve", type(e).__name__, e)
    return bundle, boosters


def load_model(path: str | Path | None = None) -> str:
    """(Re)load models. Returns the source predict_laps will use: tierb | lgbm | baseline | stub."""
    global _tierb_bundle, _bundle, _boosters
    _tierb_bundle, _bundle, _boosters = None, None, None
    _warned.clear()
    _cached.cache_clear()

    if path is not None:
        selected = Path(path)
        try:
            _tierb_bundle = _load_tierb(selected)
            log.info("trained Tier B laps model loaded from %s", selected)
            return source()
        except Exception:
            pass
        try:
            _bundle, _boosters = _load_legacy(selected)
        except Exception as e:
            log.warning("laps model not loaded from %s (%s: %s); using fixed per-compound estimate",
                        selected, type(e).__name__, e)
        return source()

    try:
        _tierb_bundle = _load_tierb(DEFAULT_TRAINED_MODEL_PATH)
        log.info("trained Tier B laps model loaded from %s", DEFAULT_TRAINED_MODEL_PATH)
        return source()
    except Exception as e:
        log.warning("trained Tier B laps model not loaded from %s (%s: %s); trying legacy fallback",
                    DEFAULT_TRAINED_MODEL_PATH, type(e).__name__, e)

    try:
        _bundle, _boosters = _load_legacy(DEFAULT_MODEL_PATH)
    except Exception as e:  # missing, corrupt, or wrong format: fixed estimate only
        log.warning("legacy laps model not loaded from %s (%s: %s); using fixed per-compound estimate",
                    DEFAULT_MODEL_PATH, type(e).__name__, e)
        return source()

    log.info("legacy laps model loaded from %s: source=%s, %s laps / %s races", DEFAULT_MODEL_PATH, source(),
             _bundle.get("n_laps"), _bundle.get("n_races"))
    return source()


def source() -> str:
    """Which predictor is active: tierb | lgbm | baseline | stub."""
    if _tierb_bundle is not None:
        return "tierb"
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


def _compound_rank(compound: str) -> float:
    """Map backend dry compounds onto the Tier B relative-compound rank."""
    ranks = {"SOFT": 0.0, "MEDIUM": 1.0, "HARD": 2.0}
    return ranks.get(str(compound).upper(), 1.0)


def _tierb_priors(compound: str) -> dict:
    """Use median circuit priors for the selected compound rank when no circuit context is available."""
    out = {"circuit_deg_prior": np.nan, "circuit_cliff_life_prior": np.nan}
    priors = _tierb_bundle.get("priors") if _tierb_bundle else None
    if priors is None or not len(priors):
        return out
    rank = _compound_rank(compound)
    rows = priors[priors["compound_rank"] == rank]
    if rows.empty:
        rows = priors
    for col in out:
        if col in rows:
            out[col] = float(rows[col].median())
    return out


def _predict_tierb(compound: str, tire_age_laps: float, track_temp_c: float) -> dict:
    """Predict laps remaining with the trained Tier B tyre-life bundle."""
    import pandas as pd
    from sidewall.models.tyre_life import lap_predictions

    age = tire_age_laps + config.MODEL_AGE_OFFSET_LAPS
    row = {
        "tyre_life": age,
        "compound_rank": _compound_rank(compound),
        "fresh_tyre": 1.0 if age <= 1.5 else 0.0,
        "stint": 1.0,
        "race_frac": 0.5,
        "fuel_kg": max(0.0, 100.0 - age * config.FUEL_KG_PER_LAP),
        "track_temp_bin": round(track_temp_c / 5.0) * 5.0,
        "era18": 1.0,
        "slope_so_far": np.nan,
        "resid_last": np.nan,
        "resid_mean3": np.nan,
        "resid_std5": np.nan,
        "deg_delta_last": np.nan,
        **_tierb_priors(compound),
    }
    pred = lap_predictions(_tierb_bundle, pd.DataFrame([row])).iloc[0]
    low = float(pred["safe_laps"])
    mid = float(pred["median_laps"])
    horizon = float(_tierb_bundle.get("max_horizon", config.MAX_FORECAST_LAPS))
    high = min(horizon, max(mid, low) + max(3.0, 0.25 * max(mid, low)))
    return {"low": round(low, 1), "mid": round(mid, 1), "high": round(high, 1)}


@lru_cache(maxsize=512)
def _cached(compound: str, age_key: int, temp_key: int) -> dict:
    """Walk the fallback chain. Cached on quantised inputs; callers get a copy."""
    tire_age_laps = age_key * config.LAPS_CACHE_AGE_STEP
    if _tierb_bundle is not None:
        try:
            return _predict_tierb(compound, tire_age_laps, temp_key)
        except Exception as e:
            _warn_once(f"trained Tier B prediction failed for {compound} ({type(e).__name__}: {e}); using legacy")
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
