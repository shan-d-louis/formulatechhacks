"""Virtual TPMS: estimate each tyre's core and surface temperature (and hence pressure) from the shared
feature contract, i.e. from data a public F1 feed has, with no tyre sensors.

Trained on Assetto Corsa Gym (Dallara F317) where the per-wheel temperatures and pressures are known.
Temperatures are learned (LightGBM on load-history features); pressure then follows from physics
(gas law, twin/gas.py), so the pressure estimate is only as good as the temperature estimate and the
cold set-up pressure, which the team knows.

    python -m sidewall.twin.virtual_tpms
"""
import glob
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sidewall import config
from sidewall.features import augment, feature_columns, make_features, resample, stream_from_acgym
from sidewall.twin.gas import hot_pressure

log = logging.getLogger("virtual_tpms")

WHEELS = ("fl", "fr", "rl", "rr")
DATASET = config.PROCESSED / "twin_dataset.parquet"
PARAMS = dict(objective="regression_l1", learning_rate=0.05, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, n_estimators=500, verbose=-1)


def build(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for i, fpath in enumerate(sorted(glob.glob(str(config.PROCESSED / "acgym" / "*.parquet")))):
        df = pd.read_parquet(fpath)
        df = df[df["speed_kmh"].notna()].reset_index(drop=True)
        if len(df) < 1000:
            continue
        stream = stream_from_acgym(df)
        for c in range(2):
            s = stream if c == 0 else augment(stream, rng)
            feats = make_features(resample(s), already_resampled=True)
            for w in WHEELS:
                feats[f"core_{w}"] = np.interp(feats["t"], df["t"], df[f"t_core_{w}"])
                surf = df[[f"t_in_{w}", f"t_mid_{w}", f"t_out_{w}"]].max(axis=1)
                feats[f"surf_{w}"] = np.interp(feats["t"], df["t"], surf)
                feats[f"psi_{w}"] = np.interp(feats["t"], df["t"], df[f"psi_{w}"])
                # Reference gas state: the stint's median gas-mass index, expressed at 20 degC. (The first
                # sample is often the car sitting in the pits and is a poor reference.)
                m_ref = np.nanmedian((df[f"psi_{w}"] + 14.696) / (df[f"t_core_{w}"] + 273.15))
                feats[f"t0_{w}"] = 20.0
                feats[f"p0_{w}"] = m_ref * (20.0 + 273.15) - 14.696
            feats["track"], feats["file_id"], feats["copy"] = df["track"].iloc[0], i, c
            parts.append(feats)
    return pd.concat(parts, ignore_index=True)


def main():
    if not DATASET.exists():
        build().to_parquet(DATASET)
    data = pd.read_parquet(DATASET)
    targets = [c for c in data.columns if c.startswith(("core_", "surf_", "psi_", "p0_", "t0_"))]
    feats = feature_columns(data.drop(columns=targets + ["track", "file_id", "copy"]))
    metrics = {"features": feats, "leave_track_out": {}}
    models = {}

    tracks = sorted(data["track"].unique())
    for tgt in [f"{k}_{w}" for k in ("core", "surf") for w in WHEELS]:
        errs, base = [], []
        for tr_name in tracks:
            te = data["track"] == tr_name
            m = lgb.LGBMRegressor(**PARAMS).fit(data.loc[~te, feats], data.loc[~te, tgt])
            pred = m.predict(data.loc[te, feats])
            errs.append(np.abs(pred - data.loc[te, tgt]).values)
            base.append(np.abs(data.loc[~te, tgt].median() - data.loc[te, tgt]).values)
        metrics["leave_track_out"][tgt] = {"mae_c": float(np.concatenate(errs).mean()),
                                           "baseline_mae_c": float(np.concatenate(base).mean())}
        models[tgt] = lgb.LGBMRegressor(**PARAMS).fit(data[feats], data[tgt])
        log.info("%s %s", tgt, metrics["leave_track_out"][tgt])

    # Pressure via the gas law from the *predicted* core temperature (leave-track-out).
    p_err = []
    for tr_name in tracks:
        te = data["track"] == tr_name
        for w in WHEELS:
            m = lgb.LGBMRegressor(**PARAMS).fit(data.loc[~te, feats], data.loc[~te, f"core_{w}"])
            core = m.predict(data.loc[te, feats])
            p = hot_pressure(data.loc[te, f"p0_{w}"].values, data.loc[te, f"t0_{w}"].values, core)
            p_err.append(np.abs(p - data.loc[te, f"psi_{w}"].values))
    metrics["pressure_from_gas_law_mae_psi"] = float(np.concatenate(p_err).mean())
    log.info("pressure MAE (psi): %.3f", metrics["pressure_from_gas_law_mae_psi"])

    joblib.dump({"models": models, "features": feats}, config.WEIGHTS / "twin_tpms.joblib")
    (config.WEIGHTS / "twin_metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
