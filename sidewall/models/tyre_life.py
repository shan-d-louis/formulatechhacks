"""Tier B: laps-to-cliff and tyre-failure hazard models on real F1 data (FastF1 2018-2025).

Both are discrete-time survival models: a classifier predicts the hazard h = P(event on this lap | tyre
survived so far). Stints that end in a planned pit stop before any event are censored, which this
formulation handles naturally (they simply stop contributing rows).

Laps-to-cliff uses landmarking: from lap a, the hazard of a cliff on each later lap a + j is predicted from
the features known at a, then chained into P(cliff within k laps). The headline "safe laps" number is the
largest k whose risk stays under a level chosen by split-conformal calibration on a held-out season, so that
the bound is breached on only ~10% of laps.

    python -m sidewall.models.tyre_life
"""
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from sidewall import config
from sidewall.data.build_stints import OUT as STINTS

log = logging.getLogger("tyre_life")

TEST_YEARS = (2024, 2025)
CALIB_YEARS = (2023,)
MAX_HORIZON = 40
FEATURES = ["tyre_life", "compound_rank", "fresh_tyre", "stint", "race_frac", "fuel_kg",
            "track_temp", "air_temp", "humidity",
            "slope_so_far", "resid_last", "resid_mean3", "resid_std5", "deg_delta_last",
            "circuit_deg_prior", "circuit_cliff_life_prior"]
EXTRA_FEATURES = ["tel_lockup_rate", "tel_spin_rate", "tel_energy_lat", "tel_energy_lon"]

CLIFF_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
                    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                    n_estimators=400, verbose=-1)
# Very few failures: a small, heavily regularised model.
FAIL_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=7, min_child_samples=300,
                   feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=20.0,
                   n_estimators=200, verbose=-1)


def _causal_stint_features(L: pd.DataFrame) -> pd.DataFrame:
    """Features known at the start of each lap (only uses completed laps of the same stint)."""
    L = L.sort_values(["year", "round", "driver", "stint", "lap"]).copy()
    g = L.groupby(["year", "round", "driver", "stint"], sort=False)
    r = L["resid"].where(L["clean"])
    L["resid_last"] = g["resid"].shift(1)
    L["resid_mean3"] = r.groupby([L["year"], L["round"], L["driver"], L["stint"]]).transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    L["resid_std5"] = r.groupby([L["year"], L["round"], L["driver"], L["stint"]]).transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).std())
    L["deg_delta_last"] = g["deg_delta"].shift(1)

    # Least-squares slope of fuel-corrected lap time vs tyre age over the stint's completed clean laps,
    # from running sums (so it only ever uses earlier laps).
    x = L["tyre_life"].astype(float)
    y = L["t_corr"].where(L["clean"])
    ok = x.notna() & y.notna()
    x0, y0 = x.where(ok, 0.0), y.where(ok, 0.0)
    keys = [L["year"], L["round"], L["driver"], L["stint"]]
    sums = {}
    for name, col in (("n", ok.astype(float)), ("sx", x0), ("sy", y0), ("sxx", x0 * x0), ("sxy", x0 * y0)):
        sums[name] = col.groupby(keys).cumsum() - col  # exclude the current lap
    den = sums["n"] * sums["sxx"] - sums["sx"] ** 2
    L["slope_so_far"] = ((sums["n"] * sums["sxy"] - sums["sx"] * sums["sy"]) / den).where((sums["n"] >= 3) & (den > 0))
    return L


def _circuit_priors(train: pd.DataFrame) -> pd.DataFrame:
    """Per circuit + compound: typical degradation slope and tyre life at the cliff (training seasons only)."""
    st = train[train["stint_usable"]].groupby(["year", "round", "driver", "stint"]).agg(
        circuit=("circuit", "first"), compound_rank=("compound_rank", "first"),
        slope=("deg_slope", "first"), cliff=("cliff", "any"), life=("tyre_life", "max"))
    cl = st[st["cliff"]].groupby(["circuit", "compound_rank"])["life"].median().rename("circuit_cliff_life_prior")
    sl = st.groupby(["circuit", "compound_rank"])["slope"].median().rename("circuit_deg_prior")
    return pd.concat([sl, cl], axis=1).reset_index()


def hazard_rows(L: pd.DataFrame, event: str) -> pd.DataFrame:
    """Rows at risk: each lap of a usable stint up to and including the first event lap."""
    L = L[L["stint_usable"] | (event == "failure")].copy()
    key = ["year", "round", "driver", "stint"]
    ev = L[event].astype(int)
    after = ev.groupby([L[k] for k in key]).cumsum() - ev
    return L[after == 0].assign(y=ev[after == 0].values)


HORIZON_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                      n_estimators=500, verbose=-1)


def landmark_rows(L: pd.DataFrame, every: int = 2) -> pd.DataFrame:
    """Landmark dataset for laps-to-cliff.

    For each landmark lap a (every `every` laps of a usable stint, before any cliff) and each later lap
    a + j of the same stint, one row with the landmark's features plus `ahead` = j and target
    y = cliff on lap a + j. Rows stop at the cliff or when the stint ended (pit stop = censoring),
    so every row is an observed at-risk lap and the hazard h(a, j) is unbiased.
    """
    key = ["year", "round", "driver", "stint"]
    U = L[L["stint_usable"]].sort_values(key + ["tyre_life"])
    parts = []
    for _, st in U.groupby(key, sort=False):
        cliff = st["cliff"].values
        n = int(np.argmax(cliff)) + 1 if cliff.any() else len(st)   # at-risk laps incl. the cliff lap
        for a in range(0, n, every):
            if st["tyre_life"].iloc[a] < 2:
                continue
            j = np.arange(0, min(n - a, MAX_HORIZON))
            base = st.iloc[[a] * len(j)].copy()
            base["ahead"] = j
            base["y"] = cliff[a + j].astype(int)
            parts.append(base)
    return pd.concat(parts, ignore_index=True)


def cliff_cdf(model, rows: pd.DataFrame, feats: list[str], horizon: int = MAX_HORIZON) -> np.ndarray:
    """P(cliff within k laps), k = 1..horizon, for each landmark row: 1 - prod(1 - h(a, j))."""
    X = pd.concat([rows[feats].assign(ahead=j) for j in range(horizon)], ignore_index=True).astype(float)
    h = model.predict_proba(X[feats + ["ahead"]])[:, 1].reshape(horizon, len(rows)).T
    return 1.0 - np.cumprod(1.0 - h, axis=1)


def safe_laps(cdf_row: np.ndarray, risk: float) -> int:
    """Largest k whose probability of reaching the cliff within k laps is still <= risk."""
    over = np.flatnonzero(cdf_row > risk)
    return int(over[0]) if len(over) else len(cdf_row)


def median_laps(cdf_row: np.ndarray) -> int:
    over = np.flatnonzero(cdf_row >= 0.5)
    return int(over[0]) + 1 if len(over) else len(cdf_row)


def _eval(model, test: pd.DataFrame, feats: list[str]) -> dict:
    p = model.predict_proba(test[feats].astype(float))[:, 1]
    y = test["y"].values
    return {"rows": int(len(y)), "events": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "base_rate": float(y.mean())}


def _outcomes(model, L_part: pd.DataFrame, feats: list[str]):
    """Landmarks of a season split with their predicted CDFs and observed laps-to-cliff / laps-to-end."""
    key = ["year", "round", "driver", "stint"]
    U = L_part[L_part["stint_usable"]]
    info = U.groupby(key).agg(end_life=("tyre_life", "max"))
    info = info.join(U[U["cliff"]].groupby(key)["tyre_life"].min().rename("cliff_life"))
    land = U.join(info, on=key)
    land = land[(land["tyre_life"] >= 3) & ~(land["tyre_life"] > land["cliff_life"])]
    cdf = cliff_cdf(model, land, feats) if len(land) else np.zeros((0, MAX_HORIZON))
    return land, cdf, (land["cliff_life"] - land["tyre_life"]).values, (land["end_life"] - land["tyre_life"]).values


def violation_rate(cdf, to_cliff, to_end, risk: float) -> tuple[float, int]:
    safe = np.array([safe_laps(r, risk) for r in cdf])
    hit = to_cliff < safe
    obs = (hit | (to_end >= safe)) & (safe > 0)
    return (float(hit[obs].mean()) if obs.any() else np.nan), int(obs.sum())


def conformal_risk(model, L_cal: pd.DataFrame, feats: list[str], target: float = 0.10) -> float:
    """Split-conformal calibration of the safe-lap bound: the largest per-bound risk level whose empirical
    violation rate on the calibration season stays <= target."""
    _, cdf, to_cliff, to_end = _outcomes(model, L_cal, feats)
    best = 0.001
    for r in np.linspace(0.001, 0.3, 120):
        v, n = violation_rate(cdf, to_cliff, to_end, r)
        if n and v <= target:
            best = float(r)
    return best


def coverage(model, L_test: pd.DataFrame, feats: list[str], risk: float = 0.1) -> dict:
    """Held-out check of the 'safe laps' bound and of calibration, on every at-risk lap of every test stint.

    A lap's outcome within k laps is observable if the cliff happened within k laps or the stint ran at least
    k more laps; laps cut short by a pit stop are excluded for that k.
    - violation_rate: share of laps whose cliff came inside the predicted 90%-safe window (target <= ~0.10)
    - calibration_k5: predicted vs observed P(cliff within 5 laps), by predicted-risk quintile
    """
    key = ["year", "round", "driver", "stint"]
    land, cdf, to_cliff, to_end = _outcomes(model, L_test, feats)
    if land.empty:
        return {}
    safe = np.array([safe_laps(r, risk) for r in cdf])
    viol, n_viol = violation_rate(cdf, to_cliff, to_end, risk)

    k = 5
    hit5 = to_cliff < k
    obs5 = hit5 | (to_end >= k)
    p5 = cdf[:, k - 1]
    q = pd.qcut(p5[obs5], 5, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"q": q, "pred": p5[obs5], "obs": hit5[obs5]})
             .groupby("q").agg(pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size")))
    return {"stints": int(land.groupby(key).ngroups), "landmarks": int(len(land)), "risk_level": risk,
            "mean_safe90_laps": float(safe.mean()),
            "violation_rate_safe90": viol, "violation_n": n_viol,
            "calibration_k5": calib.round(3).to_dict(orient="index")}


def prepare_laps(L: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Stint table -> model-ready lap features (identical for training and live use)."""
    L = _causal_stint_features(L)
    L = L.merge(priors, on=["circuit", "compound_rank"], how="left")
    L["fresh_tyre"] = L["fresh_tyre"].astype(float)
    return L


ALERT_QUANTILES = {"advise": 0.95, "box": 0.99}


def alert_thresholds(bundle: dict, laps: pd.DataFrame) -> dict:
    """Alert levels as percentiles of risk over the training laps: 'box' fires only on laps whose risk is in
    the top 1 % of everything seen in training, 'advise' in the top 5 %. Transparent and not tuned to any race."""
    p = _raw_predictions(bundle, laps)
    return {k: {lvl: float(np.nanquantile(p[k], q)) for lvl, q in ALERT_QUANTILES.items()}
            for k in ("p_cliff_3", "failure_hazard")}


def _raw_predictions(bundle: dict, laps: pd.DataFrame) -> pd.DataFrame:
    feats = bundle["features"]
    X = laps[feats].astype(float)
    cdf = cliff_cdf(bundle["models"]["cliff_horizon"], laps, feats, bundle.get("max_horizon", MAX_HORIZON))
    risk = bundle.get("safe_risk_level", 0.1)
    return pd.DataFrame({
        "cliff_hazard": bundle["models"]["cliff"].predict_proba(X)[:, 1],
        "failure_hazard": bundle["models"]["failure"].predict_proba(X)[:, 1],
        "p_cliff_3": cdf[:, 2],
        "safe_laps": [safe_laps(r, risk) for r in cdf],
        "median_laps": [median_laps(r) for r in cdf],
    }, index=laps.index)


def lap_predictions(bundle: dict, laps: pd.DataFrame) -> pd.DataFrame:
    """Per lap: hazards, P(cliff within 3 laps), safe laps (conformal), median laps and alert levels (0/1/2)."""
    p = _raw_predictions(bundle, laps)
    thr = bundle.get("alerts", {})
    for k, name in (("p_cliff_3", "cliff_alert"), ("failure_hazard", "failure_alert")):
        t = thr.get(k)
        p[name] = 0 if t is None else (p[k] >= t["advise"]).astype(int) + (p[k] >= t["box"]).astype(int)
        p[f"{k}_box"] = np.nan if t is None else t["box"]
    return p


def main():
    L = pd.read_parquet(STINTS)
    # Circuit priors come from training seasons only, so calibration and test seasons stay unseen.
    held_out = L["year"].isin(TEST_YEARS + CALIB_YEARS)
    priors = _circuit_priors(L[~held_out])
    L = prepare_laps(L, priors)
    feats = FEATURES + [c for c in EXTRA_FEATURES if c in L and L[c].notna().any()]
    metrics, models = {"test_years": TEST_YEARS, "features": feats}, {}

    # Live per-lap hazards (unweighted, so the probabilities stay calibrated).
    for event, params in (("cliff", CLIFF_PARAMS), ("failure", FAIL_PARAMS)):
        rows = hazard_rows(L, event)
        tr, te = rows[~rows["year"].isin(TEST_YEARS)], rows[rows["year"].isin(TEST_YEARS)]
        model = lgb.LGBMClassifier(**params)
        model.fit(tr[feats].astype(float), tr["y"])
        m = {"train": {"rows": len(tr), "events": int(tr["y"].sum())}}
        if len(te) and te["y"].sum():
            m["test"] = _eval(model, te, feats)
        imp = pd.Series(model.booster_.feature_importance("gain"), index=feats).nlargest(8)
        m["top_features"] = {k: float(v) for k, v in imp.items()}
        final = lgb.LGBMClassifier(**params)
        final.fit(rows[feats].astype(float), rows["y"])
        models[event] = final
        metrics[event] = m
        log.info("%s hazard: %s", event, json.dumps(m, indent=1))

    # Laps-to-cliff via landmarking: h(a, j) = P(cliff on lap a+j | at risk), chained into a CDF.
    # Seasons: train < CALIB < TEST. The safe-lap risk level is conformally calibrated on CALIB.
    hfeats = feats + ["ahead"]
    H = landmark_rows(L)
    is_cal, is_test = H["year"].isin(CALIB_YEARS), H["year"].isin(TEST_YEARS)
    tr = H[~is_cal & ~is_test]
    model = lgb.LGBMClassifier(**HORIZON_PARAMS)
    model.fit(tr[hfeats].astype(float), tr["y"])
    risk = conformal_risk(model, L[L["year"].isin(CALIB_YEARS)], feats)
    m = {"train": {"rows": len(tr), "events": int(tr["y"].sum())}, "calib_years": CALIB_YEARS,
         "conformal_risk_level": risk}
    if is_test.any():
        m["test"] = _eval(model, H[is_test], hfeats)
        m["coverage_uncalibrated_r0.10"] = coverage(model, L[L["year"].isin(TEST_YEARS)], feats, 0.10)
        m["coverage_conformal"] = coverage(model, L[L["year"].isin(TEST_YEARS)], feats, risk)
    # Final model: everything except the calibration season, so the conformal level stays valid.
    final = lgb.LGBMClassifier(**HORIZON_PARAMS)
    final.fit(H[~is_cal][hfeats].astype(float), H[~is_cal]["y"])
    models["cliff_horizon"] = final
    metrics["cliff_horizon"] = m
    log.info("cliff horizon: %s", json.dumps(m, indent=1, default=str))

    bundle = {"models": models, "features": feats, "priors": priors,
              "safe_risk_level": risk, "max_horizon": MAX_HORIZON}
    train_laps = L[~L["year"].isin(TEST_YEARS + CALIB_YEARS) & L["tyre_life"].notna()]
    bundle["alerts"] = alert_thresholds(bundle, train_laps.sample(min(20000, len(train_laps)), random_state=0))
    metrics["alert_thresholds"] = bundle["alerts"]
    if L["year"].isin(TEST_YEARS).any():
        # How often each alert level fires on held-out laps, and on laps just before a real failure.
        te = L[L["year"].isin(TEST_YEARS) & L["tyre_life"].notna()]
        p = lap_predictions(bundle, te)
        key = ["year", "round", "driver"]
        fail_laps = te[te["failure"]]
        pre = te.merge(fail_laps[key + ["lap"]].rename(columns={"lap": "fail_lap"}), on=key)
        pre = pre[(pre["lap"] <= pre["fail_lap"]) & (pre["lap"] > pre["fail_lap"] - 3)]
        pp = lap_predictions(bundle, pre) if len(pre) else None
        metrics["alert_rates_test"] = {
            "box_rate_all_laps": float((p["failure_alert"] == 2).mean()),
            "advise_rate_all_laps": float((p["failure_alert"] >= 1).mean()),
            "failures": int(len(fail_laps)),
            "box_within_3_laps_of_failure": None if pp is None else float(
                pre.assign(a=pp["failure_alert"].values).groupby(key)["a"].max().eq(2).mean()),
        }
        log.info("alert rates: %s", metrics["alert_rates_test"])
    joblib.dump(bundle, config.WEIGHTS / "tierB_tyre_life.joblib")
    (config.WEIGHTS / "tierB_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
