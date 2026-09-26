"""Overfitting / leakage diagnostics for the Tier B tyre-life models.

    python -m sidewall.models.diagnostics

Reports:
1. Data profile per season: stints, laps, cliff rate, failure rate, compound mix, tyre era.
2. Train vs held-out AUC for each model (a large gap = overfitting).
3. "Race fingerprint" test: how well race-identifying features alone predict the target.
4. Forward-chaining season evaluation (train on seasons < Y, test on Y).
"""
import json
import logging
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sidewall import config
from sidewall.data.build_stints import OUT as STINTS
from sidewall.models import tyre_life as T

warnings.filterwarnings("ignore")
log = logging.getLogger("diagnostics")

# The original (pre-fix) model settings, kept for before/after comparison.
OLD_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                  n_estimators=400, verbose=-1)


def profile(L: pd.DataFrame) -> pd.DataFrame:
    key = ["year", "round", "driver", "stint"]
    st = L[L["stint_usable"]].groupby(key).agg(cliff=("cliff", "any"), life=("tyre_life", "max"))
    g = st.groupby(level="year")
    out = pd.DataFrame({
        "races": L.groupby("year")["round"].nunique(),
        "laps": L.groupby("year").size(),
        "usable_stints": g.size(),
        "cliff_rate": g["cliff"].mean().round(3),
        "median_stint_len": g["life"].median(),
        "failures": L.groupby("year")["failure"].sum(),
        "clean_lap_frac": L.groupby("year")["clean"].mean().round(3),
    })
    return out


def auc(model, X, y):
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1])) if 0 < y.sum() < len(y) else float("nan")


def gap_report(L, feats, train_years, test_years, params):
    rows = T.hazard_rows(L, "cliff")
    tr, te = rows[rows["year"].isin(train_years)], rows[rows["year"].isin(test_years)]
    m = lgb.LGBMClassifier(**params).fit(tr[feats].astype(float), tr["y"])
    return {"train_auc": auc(m, tr[feats].astype(float), tr["y"]),
            "test_auc": auc(m, te[feats].astype(float), te["y"]),
            "train_events": int(tr["y"].sum()), "test_events": int(te["y"].sum())}


def fingerprint(L, train_years, test_years, params, cols=("track_temp", "air_temp", "humidity")):
    """Can weather columns alone (which mostly identify the race) predict the cliff? High train AUC with
    low test AUC means the model can memorise races through them."""
    rows = T.hazard_rows(L, "cliff")
    tr, te = rows[rows["year"].isin(train_years)], rows[rows["year"].isin(test_years)]
    cols = list(cols)
    m = lgb.LGBMClassifier(**params).fit(tr[cols].astype(float), tr["y"])
    return {"train_auc": auc(m, tr[cols].astype(float), tr["y"]), "test_auc": auc(m, te[cols].astype(float), te["y"])}


def forward_chain(L, feats, params, years):
    out = {}
    rows = T.hazard_rows(L, "cliff")
    for y in years:
        tr, te = rows[rows["year"] < y], rows[rows["year"] == y]
        if tr.empty or te.empty or te["y"].sum() == 0:
            continue
        m = lgb.LGBMClassifier(**params).fit(tr[feats].astype(float), tr["y"])
        out[int(y)] = {"train_auc": round(auc(m, tr[feats].astype(float), tr["y"]), 3),
                       "test_auc": round(auc(m, te[feats].astype(float), te["y"]), 3)}
    return out


def main():
    L = pd.read_parquet(STINTS)
    years = sorted(L["year"].unique())
    print("== data profile per season"); print(profile(L).to_string())
    test_years = tuple(y for y in years if y >= 2021)
    train_years = tuple(y for y in years if y < 2021)
    priors = T._circuit_priors(L[L["year"].isin(train_years)])
    P = T.prepare_laps(L, priors)
    feats = T.FEATURES
    res = {"train_years": train_years, "test_years": test_years,
           "gap_current_params": gap_report(P, feats, train_years, test_years, OLD_PARAMS),
           "fingerprint_weather_only": fingerprint(P, train_years, test_years, OLD_PARAMS),
           "forward_chain": forward_chain(P, feats, OLD_PARAMS, years[1:])}
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
