"""Laps-remaining model: exposes predict_laps().

STUB: until training/ produces laps_model.joblib, this returns a simple
per-compound estimate from tire age. The live system must never crash here.
"""

import logging

import config

log = logging.getLogger("backend.laps")


def _stub_estimate(compound: str, tire_age_laps: float) -> dict:
    """Linear life per compound: mid = life - age, with a fixed ± band."""
    life = config.STUB_LIFE_LAPS.get(str(compound).upper(), config.STUB_LIFE_DEFAULT)
    mid = max(0.0, life - max(0.0, tire_age_laps))
    return {
        "low": round(mid * (1 - config.STUB_BAND), 1),
        "mid": round(mid, 1),
        "high": round(mid * (1 + config.STUB_BAND), 1),
    }


def predict_laps(compound: str, tire_age_laps: float, track_temp_c: float) -> dict:
    """Laps remaining as {"low", "mid", "high"}. Falls back to a fixed estimate on any error."""
    try:
        return _stub_estimate(compound, tire_age_laps)
    except Exception as e:  # never let the laps model take down the live path
        log.warning("predict_laps failed, using fallback: %s", e)
        return dict(config.FALLBACK_LAPS)
