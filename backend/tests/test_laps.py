"""Laps model: the trained model, and every step of the fallback chain."""

import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pytest

import config
import laps
import main
from tests.test_passthrough import RAW


@pytest.fixture(autouse=True)
def restore_model():
    yield
    laps.load_model()  # put the real model back for other tests


def baseline_bundle(tmp_path, lgbm_model_str="not a lightgbm model"):
    """Hand-built bundle: MEDIUM delta = 0.05 * age, band ±0.25 s, LightGBM strings unusable.

    With age offset 1 and cliff 1.5 s, from tire_age 0: mid crosses at 29 laps,
    the pessimistic band (+0.25) at 24, the optimistic band (-0.25) at 34.
    """
    q = list(config.QUANTILES)
    bundle = {
        "compounds": ["MEDIUM"],
        "quantiles": q,
        "features": ["TyreLife"],
        "baseline": {"MEDIUM": {"coef": [0.0, 0.05, 0.0], "resid_lo": -0.25, "resid_hi": 0.25}},
        "max_age": {"MEDIUM": 30.0},
        "lgbm": {"MEDIUM": {x: {"model_str": lgbm_model_str, "init_score": 0.0} for x in q}},
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    return path


BASELINE_EXPECTED = {"low": 24.0, "mid": 29.0, "high": 34.0}
STUB_MEDIUM_AGE_10 = {"low": 16.0, "mid": 20.0, "high": 24.0}


class ConstantProba:
    """Pickle-friendly predict_proba stub for Tier B bundle tests."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        import numpy as np
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def tierb_bundle(tmp_path):
    bundle = {
        "models": {
            "cliff": ConstantProba(0.1),
            "failure": ConstantProba(0.01),
            "cliff_horizon": ConstantProba(0.02),
        },
        "features": [
            "tyre_life", "compound_rank", "fresh_tyre", "stint", "race_frac", "fuel_kg",
            "track_temp_bin", "era18", "slope_so_far", "resid_last", "resid_mean3",
            "resid_std5", "deg_delta_last", "circuit_deg_prior", "circuit_cliff_life_prior",
        ],
        "fail_features": [
            "tyre_life", "compound_rank", "era18", "race_frac", "track_temp_bin",
            "slope_so_far", "resid_last", "deg_delta_last", "circuit_deg_prior",
        ],
        "safe_risk_level": 0.1,
        "max_horizon": 10,
        "priors": __import__("pandas").DataFrame({
            "compound_rank": [0.0, 1.0, 2.0],
            "circuit_deg_prior": [0.02, 0.03, 0.04],
            "circuit_cliff_life_prior": [20.0, 30.0, 40.0],
        }),
    }
    path = tmp_path / "tierb.joblib"
    joblib.dump(bundle, path)
    return path


# ---------- 1. trained LightGBM model ----------

@pytest.mark.skipif(not laps.DEFAULT_TRAINED_MODEL_PATH.exists(), reason="build models/weights/tierB_tyre_life.joblib first")
def test_trained_model_loads_and_is_sane():
    assert laps.load_model() == "tierb"
    for compound in config.DRY_COMPOUNDS:
        for age in (0, 5, 10, 20):
            r = laps.predict_laps(compound, age, 35)
            assert 0 <= r["low"] <= r["mid"] <= r["high"] <= config.MAX_FORECAST_LAPS
            assert all(math.isfinite(v) for v in r.values())
    assert laps.predict_laps("SOFT", 0, 35)["low"] <= laps.predict_laps("HARD", 0, 35)["low"]


def test_tierb_bundle_is_used_when_available(tmp_path):
    assert laps.load_model(tierb_bundle(tmp_path)) == "tierb"
    assert laps.predict_laps("MEDIUM", 5.0, 35) == {"low": 5.0, "mid": 10.0, "high": 10.0}


# ---------- 2. baseline, when LightGBM fails ----------

def test_unloadable_lgbm_falls_back_to_baseline(tmp_path):
    assert laps.load_model(baseline_bundle(tmp_path)) == "baseline"
    assert laps.predict_laps("MEDIUM", 0.0, 35) == BASELINE_EXPECTED


def test_lgbm_prediction_error_falls_back_to_baseline(tmp_path):
    laps.load_model(baseline_bundle(tmp_path))

    class Broken:
        def predict(self, X):
            raise RuntimeError("boom")

    laps._boosters = {"MEDIUM": [Broken(), Broken(), Broken()]}  # as if LightGBM loaded but then fails
    laps._cached.cache_clear()
    assert laps.source() == "lgbm"
    assert laps.predict_laps("MEDIUM", 0.0, 35) == BASELINE_EXPECTED


def test_baseline_counts_down_with_age(tmp_path):
    laps.load_model(baseline_bundle(tmp_path))
    assert laps.predict_laps("MEDIUM", 10.0, 35)["mid"] == 19.0
    assert laps.predict_laps("MEDIUM", 40.0, 35) == {"low": 0.0, "mid": 0.0, "high": 0.0}  # past the cliff


# ---------- 3. fixed per-compound estimate, when the file is missing or unusable ----------

def test_missing_file_uses_fixed_estimate(tmp_path):
    assert laps.load_model(tmp_path / "does_not_exist.joblib") == "stub"
    assert laps.predict_laps("MEDIUM", 10.0, 35) == STUB_MEDIUM_AGE_10
    assert laps.predict_laps("SOFT", 50.0, 35)["mid"] == 0.0


def test_corrupt_file_uses_fixed_estimate(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"definitely not a pickle")
    assert laps.load_model(path) == "stub"
    assert laps.predict_laps("MEDIUM", 10.0, 35) == STUB_MEDIUM_AGE_10


def test_wrong_format_uses_fixed_estimate(tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump([1, 2, 3], path)
    assert laps.load_model(path) == "stub"
    assert laps.predict_laps("MEDIUM", 10.0, 35) == STUB_MEDIUM_AGE_10


def test_compound_missing_from_model_uses_fixed_estimate(tmp_path):
    laps.load_model(baseline_bundle(tmp_path))  # only knows MEDIUM
    assert laps.predict_laps("INTERMEDIATE", 0.0, 35)["mid"] == config.STUB_LIFE_DEFAULT


# ---------- 4. FALLBACK_LAPS on bad inputs; never raises ----------

@pytest.mark.parametrize("args", [
    (None, "garbage", None),
    ("MEDIUM", float("nan"), 35),
    ("MEDIUM", 5.0, float("inf")),
    ("MEDIUM", None, 35),
])
def test_bad_inputs_return_fixed_fallback(args):
    assert laps.predict_laps(*args) == config.FALLBACK_LAPS


def test_missing_track_temp_is_tolerated():
    r = laps.predict_laps("MEDIUM", 5.0, None)
    assert r != config.FALLBACK_LAPS and r["low"] <= r["mid"] <= r["high"]


def test_callers_cannot_corrupt_the_cache():
    r = laps.predict_laps("MEDIUM", 5.0, 35)
    r["mid"] = -999
    assert laps.predict_laps("MEDIUM", 5.0, 35)["mid"] != -999


# ---------- backend keeps running ----------

def test_backend_process_runs_without_model(tmp_path):
    laps.load_model(tmp_path / "does_not_exist.joblib")
    main.reset_tires()
    out = main.process(copy.deepcopy(RAW))
    life = config.STUB_LIFE_LAPS["MEDIUM"]
    assert out["laps_remaining"]["mid"] == round(life - RAW["tire_age_laps"], 1)
    assert 1 <= out["tires"]["FL"]["components"]["wear"] <= 100
