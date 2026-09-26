"""Tier B: laps-to-cliff and tyre-failure hazard models on real F1 data (FastF1 2018-2025).

Both are discrete-time survival models: a classifier predicts the hazard h = P(event on this lap | tyre
survived so far). Stints that end in a planned pit stop before any event are censored, which this
formulation handles naturally (they simply stop contributing rows).

Laps-to-cliff uses landmarking: from lap a, the hazard of a cliff on each later lap a + j is predicted from
the features known at a, then chained into P(cliff within k laps). The headline "safe laps" number is the
largest k whose risk stays under a level chosen by split-conformal calibration on a held-out season, so that
the bound is breached on only ~10% of laps.

Guarding against overfitting and leakage (see models/diagnostics.py for the evidence):
- every feature is causal: the stint's degradation trend is refitted each lap from completed laps only
- no race-fingerprint features (air temperature and humidity identified individual races); track temperature
  is kept but coarsened to 5 degC bands
- circuit priors are out-of-season for training rows (a row never sees statistics from its own season)
- all at-risk laps are used, including short stints (they are censored observations, not noise)
- physically monotone constraints (older tyre / faster degradation can never lower the risk)
- model size chosen by cross-validation grouped by race; failures (rare) use the simpler of a small tree
  model and a regularised logistic regression, whichever cross-validates better

    python -m sidewall.models.tyre_life
"""
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sidewall import config
from sidewall.data.build_stints import DRY, OUT as STINTS

log = logging.getLogger("tyre_life")

# Data split. 2022-2023 are not used (download budget). The 18-inch-tyre era (2022+) must appear in training,
# so 2024 is split by round: rounds 1-12 train, 13-24 calibrate the conformal bound; 2025 is the test season.
TEST_YEARS = (2025,)
CALIB_ROUNDS = {2024: 13}          # year -> first round of that year used for calibration


def assert_no_overlap(L: pd.DataFrame) -> None:
    """Raise if any race or stint falls in more than one of train / calib / test, or the parts are out of
    time order (every calibration race after every training race, every test race after both)."""
    part = split_of(L)
    for key in (["year", "round"], KEY):
        n = part.groupby([L[k] for k in key]).nunique()
        if (n > 1).any():
            raise ValueError(f"split overlap on {key}: {n[n > 1].index[:3].tolist()}")
    order = pd.DataFrame({"y": L["year"], "r": L["round"], "p": part.map({"train": 0, "calib": 1, "test": 2})})
    order = order.drop_duplicates(["y", "r"]).sort_values(["y", "r"])["p"]
    if not order.is_monotonic_increasing:
        raise ValueError("split overlap in time: a later race is in an earlier split")


def split_of(df: pd.DataFrame) -> pd.Series:
    """'train' / 'calib' / 'test' for every row, by season and round."""
    part = pd.Series("train", index=df.index)
    for y, r0 in CALIB_ROUNDS.items():
        part[(df["year"] == y) & (df["round"] >= r0)] = "calib"
    part[df["year"].isin(TEST_YEARS)] = "test"
    return part
MAX_HORIZON = 40
KEY = ["year", "round", "driver", "stint"]

FEATURES = ["tyre_life", "compound_rank", "fresh_tyre", "stint", "race_frac", "fuel_kg",
            "track_temp_bin", "era18",
            "slope_so_far", "resid_last", "resid_mean3", "resid_std5", "deg_delta_last",
            "circuit_deg_prior", "circuit_cliff_life_prior"]
# Risk can only go up with these (older tyre, faster degradation, laps drifting above the trend).
MONOTONE_UP = {"tyre_life", "slope_so_far", "resid_last", "resid_mean3", "deg_delta_last", "ahead"}
FAIL_FEATURES = ["tyre_life", "compound_rank", "era18", "race_frac", "track_temp_bin",
                 "slope_so_far", "resid_last", "deg_delta_last", "circuit_deg_prior"]

BASE_PARAMS = dict(objective="binary", learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.8,
                   bagging_freq=1, lambda_l2=10.0, verbose=-1)
GRID = [dict(num_leaves=nl, min_child_samples=mc, n_estimators=ne)
        for nl in (7, 15, 31) for mc in (200, 600) for ne in (150, 400)]


# ------------------------------------------------------------------ features

def _causal_stint_features(L: pd.DataFrame) -> pd.DataFrame:
    """Features known at the start of each lap: only completed laps of the same stint are used."""
    L = L.sort_values(KEY + ["lap"]).copy()
    keys = [L[k] for k in KEY]
    x = L["tyre_life"].astype(float)
    y = L["t_corr"].where(L["clean"])
    ok = x.notna() & y.notna()
    x0, y0 = x.where(ok, 0.0), y.where(ok, 0.0)
    s = {}
    for name, col in (("n", ok.astype(float)), ("sx", x0), ("sy", y0), ("sxx", x0 * x0), ("sxy", x0 * y0)):
        s[name] = col.groupby(keys).cumsum() - col          # sums over EARLIER laps only
    den = s["n"] * s["sxx"] - s["sx"] ** 2
    valid = (s["n"] >= 3) & (den > 0)
    slope = ((s["n"] * s["sxy"] - s["sx"] * s["sy"]) / den).where(valid)
    icpt = ((s["sy"] - slope * s["sx"]) / s["n"]).where(valid)
    L["slope_so_far"] = slope
    # Residual of this lap against the trend fitted on earlier laps; the features use previous laps' values.
    resid_c = (y - (icpt + slope.clip(lower=0) * x)).where(valid)
    g = resid_c.groupby(keys)
    L["resid_last"] = g.shift(1)
    prev = g.shift(1)
    gp = prev.groupby(keys)
    L["resid_mean3"] = gp.transform(lambda v: v.rolling(3, min_periods=1).mean())
    L["resid_std5"] = gp.transform(lambda v: v.rolling(5, min_periods=2).std())
    # Loss vs the first clean lap of the stint that has already been completed.
    first = y.groupby(keys).transform("first")               # the stint's first clean lap...
    first_done = s["n"] >= 1                                  # ...used only once that lap is in the past
    last_clean = y.groupby(keys).shift(1)
    L["deg_delta_last"] = (last_clean - first).where(first_done)
    L["track_temp_bin"] = (L["track_temp"] / 5).round() * 5
    L["era18"] = (L["year"] >= 2022).astype(float)        # 18-inch tyres from 2022
    L["fresh_tyre"] = L["fresh_tyre"].astype(float)
    return L


def _circuit_priors(train: pd.DataFrame) -> pd.DataFrame:
    """Per circuit + compound: typical degradation slope and tyre life at the cliff."""
    st = train[train["stint_usable"]].groupby(KEY).agg(
        circuit=("circuit", "first"), compound_rank=("compound_rank", "first"),
        slope=("deg_slope", "first"), cliff=("cliff", "any"), life=("tyre_life", "max"))
    cl = st[st["cliff"]].groupby(["circuit", "compound_rank"])["life"].median().rename("circuit_cliff_life_prior")
    sl = st.groupby(["circuit", "compound_rank"])["slope"].median().rename("circuit_deg_prior")
    return pd.concat([sl, cl], axis=1).reset_index()


def prepare_laps(L: pd.DataFrame, priors: pd.DataFrame) -> pd.DataFrame:
    """Stint table -> model-ready lap features (live use and non-training rows)."""
    L = _causal_stint_features(L)
    return L.merge(priors, on=["circuit", "compound_rank"], how="left")


def prepare_training(L: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for all rows. Training rows get OUT-OF-SEASON priors (computed from the other training
    seasons), so a row never benefits from statistics that include its own stint. Other rows get priors
    from all training rows. Returns (features, priors-for-inference)."""
    F = _causal_stint_features(L)
    train_mask = train_mask.reindex(L.index).fillna(False).astype(bool)
    T = L[train_mask]
    priors_all = _circuit_priors(T)
    in_train = F.index.isin(T.index)
    parts = []
    for y in sorted(T["year"].unique()):
        p = _circuit_priors(T[T["year"] != y])
        parts.append(F[in_train & (F["year"] == y)].merge(p, on=["circuit", "compound_rank"], how="left"))
    parts.append(F[~in_train].merge(priors_all, on=["circuit", "compound_rank"], how="left"))
    return pd.concat(parts, ignore_index=True), priors_all


# ------------------------------------------------------------------ survival datasets

def _at_risk(L: pd.DataFrame) -> pd.DataFrame:
    """Dry-compound laps with a known tyre age. Short stints stay in: they are censored observations."""
    return L[L["compound"].isin(DRY) & L["tyre_life"].notna()]


def hazard_rows(L: pd.DataFrame, event: str) -> pd.DataFrame:
    """Every at-risk lap up to and including the first event lap of its stint."""
    L = _at_risk(L).copy()
    ev = L[event].fillna(False).astype(int)
    after = ev.groupby([L[k] for k in KEY]).cumsum() - ev
    keep = after == 0
    return L[keep].assign(y=ev[keep].values)


def landmark_rows(L: pd.DataFrame, every: int = 3, horizon: int = MAX_HORIZON) -> pd.DataFrame:
    """Landmark dataset for laps-to-cliff: for landmark lap a and each later lap a + j of the same stint
    (until the cliff or the stint's end), one row with a's features, `ahead` = j and y = cliff on lap a + j."""
    U = _at_risk(L).sort_values(KEY + ["tyre_life"])
    parts = []
    for _, st in U.groupby(KEY, sort=False):
        cliff = st["cliff"].fillna(False).values
        n = int(np.argmax(cliff)) + 1 if cliff.any() else len(st)
        for a in range(0, n, every):
            if st["tyre_life"].iloc[a] < 2:
                continue
            j = np.arange(0, min(n - a, horizon))
            base = st.iloc[[a] * len(j)].copy()
            base["ahead"] = j
            base["y"] = cliff[a + j].astype(int)
            parts.append(base)
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ models

def _monotone(feats):
    return [1 if f in MONOTONE_UP else 0 for f in feats]


def _race_groups(df):
    return (df["year"].astype(int) * 100 + df["round"].astype(int)).values


def _cv_auc(make_model, X, y, groups, n_splits=4):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = make_model().fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return float(roc_auc_score(y, oof)), oof


def tune_lgbm(rows: pd.DataFrame, feats: list[str], grid=GRID, max_rows: int = 250_000):
    """Pick the tree model size by race-grouped cross-validation on the training seasons."""
    if len(rows) > max_rows:
        rows = rows.sample(max_rows, random_state=0)
    X, y, groups = rows[feats].astype(float), rows["y"].values, _race_groups(rows)
    results = []
    for g in grid:
        params = {**BASE_PARAMS, **g, "monotone_constraints": _monotone(feats)}
        auc, _ = _cv_auc(lambda: lgb.LGBMClassifier(**params), X, y, groups)
        results.append((auc, g))
    results.sort(key=lambda r: -r[0])
    best_auc, best = results[0]
    return {**BASE_PARAMS, **best, "monotone_constraints": _monotone(feats)}, best_auc, results


def make_logistic():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000))


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


# ------------------------------------------------------------------ evaluation

def _eval(model, df: pd.DataFrame, feats: list[str]) -> dict:
    p = model.predict_proba(df[feats].astype(float))[:, 1]
    y = df["y"].values
    return {"rows": int(len(y)), "events": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
            "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
            "base_rate": float(y.mean()), "mean_pred": float(p.mean())}


def _outcomes(model, L_part: pd.DataFrame, feats: list[str]):
    """Landmarks (laps of stints where a cliff could be labelled) with predicted CDFs and observed outcomes."""
    U = L_part[L_part["stint_usable"] & L_part["tyre_life"].notna()]
    info = U.groupby(KEY).agg(end_life=("tyre_life", "max"))
    info = info.join(U[U["cliff"]].groupby(KEY)["tyre_life"].min().rename("cliff_life"))
    land = U.join(info, on=KEY)
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
    """Held-out check of the 'safe laps' bound and of calibration on every landmark of every test stint.
    - violation_rate: share of laps whose cliff came inside the predicted safe window (target ~0.10)
    - calibration_k5: predicted vs observed P(cliff within 5 laps), by predicted-risk quintile"""
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
    return {"stints": int(land.groupby(KEY).ngroups), "landmarks": int(len(land)), "risk_level": risk,
            "mean_safe_laps": float(safe.mean()), "violation_rate": viol, "violation_n": n_viol,
            "calibration_k5": calib.round(3).to_dict(orient="index")}


# ------------------------------------------------------------------ inference helpers

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
        "failure_hazard": bundle["models"]["failure"].predict_proba(laps[bundle["fail_features"]].astype(float))[:, 1],
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


# ------------------------------------------------------------------ training

def _fit_report(name, model, tr, te_sets: dict, feats, cv_auc=None):
    rep = {"train_in_sample": _eval(model, tr, feats)}
    if cv_auc is not None:
        rep["train_cv_grouped_by_race_auc"] = cv_auc
    for k, df in te_sets.items():
        if len(df) and df["y"].sum():
            rep[k] = _eval(model, df, feats)
    tr_auc = rep["train_in_sample"]["roc_auc"]
    held = [v["roc_auc"] for k, v in rep.items() if k not in ("train_in_sample",) and isinstance(v, dict) and v.get("roc_auc")]
    if tr_auc and held:
        rep["overfit_gap_auc"] = round(tr_auc - float(np.mean(held)), 3)
    log.info("%s: %s", name, json.dumps(rep, indent=1, default=str))
    return rep


def main():
    L = pd.read_parquet(STINTS)
    part_L = split_of(L)
    P, priors = prepare_training(L, part_L == "train")
    feats = FEATURES

    def split(df):
        p = split_of(df)
        return df[p == "train"], df[p == "calib"], df[p == "test"]

    races = L.groupby(["year", "round"]).size().reset_index()
    races["part"] = split_of(races)
    assert_no_overlap(L)
    metrics = {"split": {k: {"races": int(len(g)), "seasons": sorted(int(y) for y in g["year"].unique())}
                         for k, g in races.groupby("part")},
               "calib_rounds": CALIB_ROUNDS, "test_years": TEST_YEARS,
               "features": feats, "fail_features": FAIL_FEATURES}
    log.info("split: %s", metrics["split"])
    models = {}

    # 1. Live cliff hazard.
    rows = hazard_rows(P, "cliff")
    tr, ca, te = split(rows)
    params, cv, grid = tune_lgbm(tr, feats)
    metrics["cliff_params"] = {k: v for k, v in params.items() if k != "monotone_constraints"}
    m = lgb.LGBMClassifier(**params).fit(tr[feats].astype(float), tr["y"])
    metrics["cliff"] = _fit_report("cliff hazard", m, tr, {"calib": ca, "test": te}, feats, cv)
    models["cliff"] = m

    # 2. Tyre failure hazard: small tree model vs regularised logistic regression, chosen by grouped CV.
    frows = hazard_rows(P, "failure")
    ftr, fca, fte = split(frows)
    X, y, groups = ftr[FAIL_FEATURES].astype(float), ftr["y"].values, _race_groups(ftr)
    fparams = {**BASE_PARAMS, "num_leaves": 4, "min_child_samples": 400, "n_estimators": 150,
               "monotone_constraints": _monotone(FAIL_FEATURES)}
    cv_tree, _ = _cv_auc(lambda: lgb.LGBMClassifier(**fparams), X, y, groups)
    cv_log, _ = _cv_auc(make_logistic, X, y, groups)
    use_log = cv_log >= cv_tree
    fm = (make_logistic() if use_log else lgb.LGBMClassifier(**fparams)).fit(X, y)
    metrics["failure_model_choice"] = {"lgbm_cv_auc": cv_tree, "logistic_cv_auc": cv_log,
                                       "chosen": "logistic" if use_log else "lgbm"}
    metrics["failure"] = _fit_report("failure hazard", fm, ftr, {"calib": fca, "test": fte}, FAIL_FEATURES,
                                     max(cv_tree, cv_log))
    models["failure"] = fm

    # 3. Laps-to-cliff (landmark hazard), conformally calibrated on the calibration season.
    hfeats = feats + ["ahead"]
    H = landmark_rows(P)
    htr, hca, hte = split(H)
    grid_small = [dict(num_leaves=nl, min_child_samples=600, n_estimators=ne) for nl in (7, 15) for ne in (200, 400)]
    hparams, hcv, _ = tune_lgbm(htr, hfeats, grid_small, max_rows=150_000)
    hm = lgb.LGBMClassifier(**hparams).fit(htr[hfeats].astype(float), htr["y"])
    rep = _fit_report("laps-to-cliff", hm, htr, {"calib": hca, "test": hte}, hfeats, hcv)
    _, Lcal, Ltest = split(P)
    risk = conformal_risk(hm, Lcal, feats) if len(Lcal) else 0.05
    rep["conformal_risk_level"] = risk
    if len(Ltest):
        rep["coverage_uncalibrated_r0.10"] = coverage(hm, Ltest, feats, 0.10)
        rep["coverage_conformal"] = coverage(hm, Ltest, feats, risk)
    metrics["cliff_horizon"] = rep
    metrics["cliff_horizon_params"] = {k: v for k, v in hparams.items() if k != "monotone_constraints"}
    models["cliff_horizon"] = hm
    log.info("conformal risk %.4f, coverage %s", risk, json.dumps(rep.get("coverage_conformal", {}), default=str))

    # Final models for the demo: refit on train + test seasons (never on the calibration season, so the
    # conformal level stays valid) with the hyper-parameters chosen above.
    fit_mask = lambda df: split_of(df) != "calib"  # noqa: E731
    models["cliff"] = lgb.LGBMClassifier(**params).fit(rows[fit_mask(rows)][feats].astype(float), rows[fit_mask(rows)]["y"])
    fr = frows[fit_mask(frows)]
    models["failure"] = (make_logistic() if use_log else lgb.LGBMClassifier(**fparams)).fit(
        fr[FAIL_FEATURES].astype(float), fr["y"])
    models["cliff_horizon"] = lgb.LGBMClassifier(**hparams).fit(H[fit_mask(H)][hfeats].astype(float), H[fit_mask(H)]["y"])
    _, priors_final = prepare_training(L, part_L != "calib")

    bundle = {"models": models, "features": feats, "fail_features": FAIL_FEATURES, "priors": priors_final,
              "safe_risk_level": risk, "max_horizon": MAX_HORIZON}
    train_laps = split(P)[0]
    train_laps = train_laps[train_laps["tyre_life"].notna()]
    bundle["alerts"] = alert_thresholds(bundle, train_laps.sample(min(20000, len(train_laps)), random_state=0))
    metrics["alert_thresholds"] = bundle["alerts"]
    if len(Ltest):
        te_l = Ltest[Ltest["tyre_life"].notna()]
        p = lap_predictions(bundle, te_l)
        fail = te_l[te_l["failure"]]
        pre = te_l.merge(fail[["year", "round", "driver", "lap"]].rename(columns={"lap": "fail_lap"}),
                         on=["year", "round", "driver"])
        pre = pre[(pre["lap"] <= pre["fail_lap"]) & (pre["lap"] > pre["fail_lap"] - 3)]
        pp = lap_predictions(bundle, pre) if len(pre) else None
        metrics["alert_rates_test"] = {
            "box_rate_all_laps": float((p["failure_alert"] == 2).mean()),
            "failures": int(len(fail)),
            "box_within_3_laps_of_failure": None if pp is None else float(
                pre.assign(a=pp["failure_alert"].values).groupby(["year", "round", "driver"])["a"].max().eq(2).mean()),
        }
    bundle["hyperparams"] = {"cliff": params, "failure": fparams, "failure_logistic": bool(use_log),
                             "cliff_horizon": hparams}
    joblib.dump(bundle, config.WEIGHTS / "tierB_tyre_life.joblib")
    (config.WEIGHTS / "tierB_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    for f in config.WEIGHTS.glob("tierB_excl_*.joblib"):   # held-out replay models are now stale
        f.unlink()


def bundle_excluding(base: dict, years) -> dict:
    """A copy of the Tier B models refitted WITHOUT the given seasons (and without the calibration season),
    using the already-chosen hyper-parameters. Replays of a race use the model that never saw its season,
    so what the pit wall shows is a genuine out-of-sample prediction. Cached on disk."""
    years = sorted(int(y) for y in years)
    path = config.WEIGHTS / f"tierB_excl_{'_'.join(map(str, years))}.joblib"
    if path.exists():
        return joblib.load(path)
    hp = base["hyperparams"]
    L = pd.read_parquet(STINTS)
    keep = lambda df: (split_of(df) != "calib") & ~df["year"].isin(years)  # noqa: E731
    P, priors = prepare_training(L, keep(L))
    fit = lambda df: df[keep(df)]  # noqa: E731
    feats, ffeats, hfeats = base["features"], base["fail_features"], base["features"] + ["ahead"]
    rows, frows, H = fit(hazard_rows(P, "cliff")), fit(hazard_rows(P, "failure")), fit(landmark_rows(P))
    models = {
        "cliff": lgb.LGBMClassifier(**hp["cliff"]).fit(rows[feats].astype(float), rows["y"]),
        "failure": (make_logistic() if hp["failure_logistic"] else lgb.LGBMClassifier(**hp["failure"])).fit(
            frows[ffeats].astype(float), frows["y"]),
        "cliff_horizon": lgb.LGBMClassifier(**hp["cliff_horizon"]).fit(H[hfeats].astype(float), H["y"]),
    }
    out = {**base, "models": models, "priors": priors, "excluded_years": years}
    joblib.dump(out, path)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
