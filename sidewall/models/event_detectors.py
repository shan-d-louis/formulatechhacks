"""Tier A: early-warning detectors for lock-up, wheelspin, overheating and cold tyres.

Training data: Assetto Corsa Gym (Dallara F317, 15 human drivers) with per-wheel slip ratio and tyre
temperatures as ground truth. Inputs are restricted to the shared feature contract (features.py), with
the sim streams degraded to look like F1 public telemetry, so the models run on FastF1 data too.

    python -m sidewall.models.event_detectors            # build dataset, evaluate, train, save
"""
import argparse
import glob
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold

from sidewall import config
from sidewall.features import (augment, feature_columns, make_features, resample,
                               stream_from_acgym, stream_from_thulab)
from sidewall.models.labels import EVENTS, sim_labels, to_grid

log = logging.getLogger("event_detectors")

AHEAD_S = 1.0     # warn up to 1 s before the event
BACK_S = 0.25     # one F1 sample
DATASET = config.PROCESSED / "tierA_dataset.parquet"
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=50,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=600, verbose=-1)


# ---------------------------------------------------------------- dataset

def build_from_acgym(copies: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    files = sorted(glob.glob(str(config.PROCESSED / "acgym" / "*.parquet")))
    for i, fpath in enumerate(files):
        df = pd.read_parquet(fpath)
        df = df[df["speed_kmh"].notna()].reset_index(drop=True)
        if len(df) < 1000:
            continue
        labels = sim_labels(df, hz=100, throttle_scale=100, brake_scale=100)
        stream = stream_from_acgym(df)
        for c in range(copies):
            s = stream if c == 0 else augment(stream, rng)
            grid = resample(s, jitter=0.0 if c == 0 else 0.3, rng=rng)
            feats = make_features(grid, already_resampled=True)
            lab = to_grid(labels, feats["t"].values, BACK_S, AHEAD_S)
            part = pd.concat([feats, lab.drop(columns="t")], axis=1)
            part["session"] = df["session"].iloc[0]
            part["track"] = df["track"].iloc[0]
            part["file_id"] = i
            part["copy"] = c
            parts.append(part)
        log.info("[%d/%d] %s", i + 1, len(files), fpath.rsplit("\\", 1)[-1][:70])
    return pd.concat(parts, ignore_index=True)


def build_thulab() -> pd.DataFrame:
    """External test set: a different car (GT, with ABS/TC) at Spa. Slip ratio from wheel speeds."""
    raw = pd.read_parquet(config.THULAB_PARQUET)
    v = raw["speed_kmh"] / 3.6
    coast = (raw.throttle < 0.05) & (raw.brake < 0.05) & (raw.g_lat.abs() < 0.3) & (raw.speed_kmh > 60)
    df = pd.DataFrame({"t": raw["timestamp"] - raw["timestamp"].iloc[0],
                       "speed_kmh": raw["speed_kmh"], "throttle": raw["throttle"], "brake": raw["brake"]})
    for w in ("fl", "fr", "rl", "rr"):
        radius = np.median((v / raw[f"wheel_speed_{w}"])[coast])
        df[f"kappa_{w}"] = ((raw[f"wheel_speed_{w}"] * radius - v) / v.clip(lower=1)).values
        df[f"t_core_{w}"] = raw[f"tyre_core_{w}"]
        df[f"t_in_{w}"], df[f"t_mid_{w}"], df[f"t_out_{w}"] = raw[f"temp_i_{w}"], raw[f"temp_m_{w}"], raw[f"temp_o_{w}"]
    hz = 1.0 / np.median(np.diff(df["t"]))
    labels = sim_labels(df, hz=hz)
    feats = make_features(resample(stream_from_thulab(raw)), already_resampled=True)
    lab = to_grid(labels, feats["t"].values, BACK_S, AHEAD_S)
    return pd.concat([feats, lab.drop(columns="t")], axis=1)


# ---------------------------------------------------------------- evaluation helpers

def lead_times(t: np.ndarray, prob: np.ndarray, now: np.ndarray, thr: float, window_s: float = 3.0):
    """For each event onset, how early (s) the probability first crossed `thr`; NaN if missed."""
    onsets = np.flatnonzero(now & ~np.r_[False, now[:-1]])
    leads = []
    for o in onsets:
        lo = np.searchsorted(t, t[o] - window_s)
        hi = min(o + 2, len(t))  # allow detection up to ~0.5 s after onset
        hits = np.flatnonzero(prob[lo:hi] >= thr)
        leads.append(t[o] - t[lo + hits[0]] if len(hits) else np.nan)
    return np.array(leads)


def threshold_at_precision(y, p, target=0.5):
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.flatnonzero(prec[:-1] >= target)
    return float(thr[ok[0]]) if len(ok) else 0.5


def evaluate(data: pd.DataFrame, feats: list[str], event: str, groups: np.ndarray, n_splits: int = 5) -> dict:
    y = data[event].values.astype(int)
    oof = np.zeros(len(data))
    for tr, te in GroupKFold(n_splits).split(data, y, groups):
        pos = y[tr].mean()
        m = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=min(20.0, (1 - pos) / max(pos, 1e-6)))
        m.fit(data.iloc[tr][feats], y[tr])
        oof[te] = m.predict_proba(data.iloc[te][feats])[:, 1]
    thr = threshold_at_precision(y, oof, 0.5)
    clean = (data["copy"] == 0).values if "copy" in data else np.ones(len(data), bool)
    leads = []
    for fid in np.unique(data.loc[clean, "file_id"]):
        m = clean & (data["file_id"].values == fid)
        leads.append(lead_times(data["t"].values[m], oof[m], data[f"{event}_now"].values[m], thr))
    leads = np.concatenate(leads) if leads else np.array([])
    return {
        "positives_frac": float(y.mean()),
        "pr_auc": float(average_precision_score(y, oof)),
        "roc_auc": float(roc_auc_score(y, oof)),
        "threshold_at_p50": thr,
        "events": int(len(leads)),
        "detected_frac": float(np.mean(~np.isnan(leads))) if len(leads) else None,
        "median_lead_s": float(np.nanmedian(leads)) if np.any(~np.isnan(leads)) else None,
    }


# ---------------------------------------------------------------- main

def main(rebuild: bool):
    if rebuild or not DATASET.exists():
        data = build_from_acgym()
        data.to_parquet(DATASET)
    data = pd.read_parquet(DATASET)
    feats = feature_columns(data.drop(columns=[c for c in data.columns if c in EVENTS or c.endswith("_now")
                                               or c in ("session", "track", "file_id", "copy")]))
    log.info("dataset %s, %d features, %d sessions, tracks %s", data.shape, len(feats),
             data["session"].nunique(), sorted(data["track"].unique()))

    thulab = build_thulab()
    metrics = {"n_rows": len(data), "features": feats, "ahead_s": AHEAD_S, "events": {}}
    models = {}
    for ev in EVENTS:
        if data[ev].sum() < 50:
            log.warning("skipping %s: too few positives", ev)
            continue
        m_session = evaluate(data, feats, ev, data["file_id"].values)
        m_track = evaluate(data, feats, ev, data["track"].values, n_splits=data["track"].nunique())
        y = data[ev].values.astype(int)
        pos = y.mean()
        model = lgb.LGBMClassifier(**PARAMS, scale_pos_weight=min(20.0, (1 - pos) / max(pos, 1e-6)))
        model.fit(data[feats], y)
        models[ev] = model
        ext = {}
        if ev in ("lockup", "wheelspin") and thulab[ev].sum() > 5:
            p = model.predict_proba(thulab[feats])[:, 1]
            ext = {"pr_auc": float(average_precision_score(thulab[ev], p)),
                   "roc_auc": float(roc_auc_score(thulab[ev], p)),
                   "positives_frac": float(thulab[ev].mean())}
        imp = pd.Series(model.booster_.feature_importance("gain"), index=feats).nlargest(8)
        metrics["events"][ev] = {"leave_session_out": m_session, "leave_track_out": m_track,
                                 "external_thulab_gt_car": ext,
                                 "top_features": {k: float(v) for k, v in imp.items()}}
        log.info("%s: %s", ev, json.dumps(metrics["events"][ev], indent=1))

    joblib.dump({"models": models, "features": feats, "ahead_s": AHEAD_S,
                 "thresholds": {ev: metrics["events"][ev]["leave_session_out"]["threshold_at_p50"] for ev in models}},
                config.WEIGHTS / "tierA_events.joblib")
    (config.WEIGHTS / "tierA_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    main(ap.parse_args().rebuild)
