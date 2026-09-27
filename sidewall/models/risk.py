"""Explainable, calibrated, predictive lock-up / wheelspin risk: "demand vs grip".

A lock-up happens when braking demand exceeds the grip the front tyres can give; wheelspin when drive demand
exceeds what the rear tyres can put down. Grip itself depends on tyre temperature (outside the working window
there is less of it) and pressure (off the operating pressure the contact patch is wrong).

  Stage 1 (pattern recognition): LightGBM on the shared telemetry features (inputs a public F1 feed has),
          trained WITHOUT class re-weighting so its output is a calibrated probability.
          TreeSHAP splits every prediction into factor families the pit wall understands:
          Braking · Throttle · Speed & cornering · Engine & gearing · Tyre heat history.
  Stage 2 (per car): a logistic regression on the Stage-1 log-odds plus
          - driver demand (braking demand, throttle demand, speed, cornering load)
          - measured tyre condition: axle surface temperature below / above the window, pressure below / above
            the operating pressure.
          Its coefficients are readable as odds ratios ("each 10 degC below the window multiplies the odds by x").

Output: P(event within the next second), the share of that risk coming from each factor, and a preventive
instruction chosen from the factors that dominated over the last few corners.

Stage 2 is fitted per car ("domain"):
  acgym  Assetto Corsa Gym, Dallara F317 (used for public-data replays)
  sim    the live simulator car, from a short calibration run with mixed driving styles and pressure set-ups
         (as a team would calibrate on a new car in first practice)

    python -m sidewall.models.risk
"""
import glob
import json
import logging

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from sidewall import config
from sidewall.models.event_detectors import DATASET as TIER_A_DATASET

log = logging.getLogger("risk")

EVENTS = ("lockup", "wheelspin")
WINDOW = (85.0, 115.0)          # tyre surface working window (degC)
STAGE1_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                     feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
                     n_estimators=300, verbose=-1)
OUT = config.WEIGHTS / "risk.joblib"
METRICS = config.WEIGHTS / "risk_metrics.json"

# Stage-2 inputs (besides the Stage-1 log-odds) and the factor they explain.
DEMAND = {"lockup": [("brake_demand", "Braking"), ("brake_speed", "Braking"), ("lat_load", "Speed & cornering")],
          "wheelspin": [("throttle_demand", "Throttle"), ("low_gear_throttle", "Throttle"), ("lat_load", "Speed & cornering")]}
TYRE = [("cold", "Tyre temperature"), ("hot", "Tyre temperature"), ("psi_low", "Tyre pressure"), ("psi_high", "Tyre pressure")]
FACTORS = ["Braking", "Throttle", "Speed & cornering", "Engine & gearing", "Tyre heat history",
           "Tyre temperature", "Tyre pressure"]


def feature_family(name: str) -> str:
    """Map a shared-contract feature to the factor family it describes."""
    if "ewm" in name or name == "tyre_age_s":
        return "Tyre heat history"
    if name.startswith(("brake", "since_brake", "p_brake")) or name.startswith(("ax", "d_speed")) or name == "p_lon":
        return "Braking"
    if name.startswith(("throttle", "d_throttle", "full_throttle", "p_drive")):
        return "Throttle"
    if name.startswith(("rpm", "d_rpm", "gear")):
        return "Engine & gearing"
    return "Speed & cornering"


# ------------------------------------------------------------------ stage-2 inputs

def tyre_terms(surf, psi, psi_ref) -> dict:
    """Heat and pressure terms for an axle, in units of 10 degC and 1 psi."""
    surf, psi, psi_ref = (np.asarray(v, float) for v in (surf, psi, psi_ref))
    return {"cold": np.clip(WINDOW[0] - surf, 0, None) / 10.0, "hot": np.clip(surf - WINDOW[1], 0, None) / 10.0,
            "psi_low": np.clip(psi_ref - psi, 0, None), "psi_high": np.clip(psi - psi_ref, 0, None)}


def demand_terms(f: pd.DataFrame) -> dict:
    """What the driver is asking of the tyres, from the shared telemetry features."""
    thr = f["throttle"].values
    return {"brake_demand": f["brake_frac4"].values * np.clip(-f["ax"].values, 0, None),
            "brake_speed": f["brake"].values * f["speed"].values / 300.0,
            "throttle_demand": thr * (1 + np.clip(f["d_throttle"].values, 0, None) / 4.0),
            "low_gear_throttle": thr * (f["gear"].values <= 3),
            "lat_load": np.abs(f["ay"].values) / 3.0}


def stage2_inputs(ev: str) -> list[str]:
    return ["ml_logit"] + [n for n, _ in DEMAND[ev]] + [n for n, _ in TYRE]


def stage2_frame(ev: str, f: pd.DataFrame, ml_logit, surf, psi, psi_ref) -> pd.DataFrame:
    X = {"ml_logit": np.asarray(ml_logit, float), **demand_terms(f), **tyre_terms(surf, psi, psi_ref)}
    return pd.DataFrame({k: X[k] for k in stage2_inputs(ev)})


def _logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error: average |predicted - observed| across probability bins."""
    q = np.minimum((np.asarray(p) * bins).astype(int), bins - 1)
    return float(sum(abs(p[q == b].mean() - y[q == b].mean()) * (q == b).mean() for b in range(bins) if (q == b).any()))


def reliability(y, p, bins=(0, .02, .05, .1, .2, .4, 1.0)) -> list:
    """Predicted vs observed rate per probability band (for the judges' calibration chart)."""
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 30:
            out.append({"band": f"{lo:.0%}-{hi:.0%}", "pred": round(float(p[m].mean()), 4),
                        "obs": round(float(y[m].mean()), 4), "n": int(m.sum())})
    return out


# ------------------------------------------------------------------ inference

class RiskModel:
    """Stage-1 models + per-domain Stage-2 models: calibrated risk, factor contributions."""

    def __init__(self, bundle: dict, domain: str):
        self.b = bundle
        self.domain = domain if domain in bundle["stage2"] else "acgym"
        # A lift off the throttle is the approach to a braking zone, and releasing the brake is the start of the
        # exit, so for each event the opposite pedal's signals belong to that event's own phase.
        phase = {"lockup": {"Throttle": "Braking"}, "wheelspin": {"Braking": "Throttle"}}
        self.families = {ev: np.array([phase[ev].get(feature_family(n), feature_family(n)) for n in bundle["features"]])
                         for ev in EVENTS}

    def predict(self, f: pd.DataFrame, tyres: dict) -> dict:
        out = {}
        X1 = f[self.b["features"]]
        for ev in EVENTS:
            booster = self.b["stage1"][ev].booster_
            contrib = booster.predict(X1, pred_contrib=True)          # (n, n_features + 1) in log-odds
            ml_logit = contrib.sum(axis=1)
            st = tyres[ev]
            X2 = stage2_frame(ev, f, ml_logit, st["surf"], st["psi"], st["psi_ref"])
            m = self.b["stage2"][self.domain][ev]
            coef = dict(zip(m["inputs"], m["coef"]))
            z = m["intercept"] + X2.values @ m["coef"]
            groups = {k: np.zeros(len(f)) for k in FACTORS}
            # Telemetry pattern: split the Stage-1 log-odds (minus its average) into families, scaled by Stage 2.
            fam = self.families[ev]
            for k in set(fam):
                groups[k] += coef["ml_logit"] * contrib[:, :-1][:, fam == k].sum(axis=1)
            for name, grp in DEMAND[ev] + TYRE:
                groups[grp] += coef[name] * (X2[name].values - m["ref"][name])
            out[ev] = {"p": 1 / (1 + np.exp(-z)), "groups": groups}
        return out


def advice(ev: str, top: str | None, surf: float, psi: float, psi_ref: float, wheel: str | None) -> str:
    """Preventive instruction for the driver / pit wall, from the dominant factor."""
    names = {"fl": "front-left", "fr": "front-right", "rl": "rear-left", "rr": "rear-right"}
    where = names[wheel] if wheel else ("fronts" if ev == "lockup" else "rears")
    if top in ("Tyre temperature", "Tyre heat history"):
        if surf < WINDOW[0]:
            return (f"{where.capitalize()} cold ({surf:.0f}°C, window {WINDOW[0]:.0f}-{WINDOW[1]:.0f}°C): build heat; " +
                    ("brake bias 1-2 clicks rearward until they're up to temperature." if ev == "lockup"
                     else "feed the throttle in gently until the rears are up to temperature."))
        if surf > WINDOW[1]:
            return (f"{where.capitalize()} overheating ({surf:.0f}°C): " +
                    ("lift and coast before the big stops to cool them." if ev == "lockup"
                     else "short-shift and avoid wheelspin on exits to cool them."))
        return ("Tyres have worked hard in the last minutes: ease the braking attack for a lap." if ev == "lockup"
                else "Rears have worked hard in the last minutes: smoother exits for a lap.")
    if top == "Tyre pressure":
        d = psi - psi_ref
        return (f"{where.capitalize()} pressure {d:+.1f} psi vs the {psi_ref:.1f} psi target: less grip. " +
                ("Check for a slow puncture. " if d < -1.5 else "") + "Correct it at the next stop.")
    if ev == "lockup":
        return {"Braking": "Braking harder than the fronts can take: brake a few metres earlier and ease the initial pedal hit.",
                "Speed & cornering": "Braking while still turning: straighten the car before the heavy stop.",
                "Engine & gearing": "Downshifting too early: engine braking is locking the fronts; delay the downshifts.",
                }.get(top, "Lock-up pattern building: smoother, earlier braking at the next heavy stop.")
    return {"Throttle": "More throttle than the rears can put down: feed it in progressively, short-shift out of slow corners.",
            "Speed & cornering": "Power applied while still turning: wait for the exit, then open the throttle.",
            "Engine & gearing": "Too much torque in low gear: short-shift or use a higher gear out of slow corners.",
            }.get(top, "Wheelspin pattern building: smoother throttle on the next exit.")


# ------------------------------------------------------------------ training helpers

def _acgym_tyre_state(data: pd.DataFrame) -> pd.DataFrame:
    """Axle surface temperature and pressure (plus the stint's operating pressure) for every Tier A row."""
    files = sorted(glob.glob(str(config.PROCESSED / "acgym" / "*.parquet")))
    out = pd.DataFrame(index=data.index, columns=["surf_f", "surf_r", "psi_f", "psi_r", "psi_ref_f", "psi_ref_r"],
                       dtype=float)
    for fid, rows in data.groupby("file_id"):
        df = pd.read_parquet(files[int(fid)])
        df = df[df["speed_kmh"].notna()]
        t = rows["t"].values
        for ax, ws in (("f", ("fl", "fr")), ("r", ("rl", "rr"))):
            surf = df[[f"t_{p}_{w}" for p in ("in", "mid", "out") for w in ws]].mean(axis=1)
            psi = df[[f"psi_{w}" for w in ws]].mean(axis=1)
            out.loc[rows.index, f"surf_{ax}"] = np.interp(t, df["t"], surf)
            out.loc[rows.index, f"psi_{ax}"] = np.interp(t, df["t"], psi)
            hot = psi[df[[f"t_core_{w}" for w in ws]].mean(axis=1) > 70]
            out.loc[rows.index, f"psi_ref_{ax}"] = float(hot.median() if len(hot) else psi.median())
    return out


def _stage1_logit(stage1, X) -> np.ndarray:
    return stage1.booster_.predict(X, raw_score=True)


def fit_stage2(ev, f, ml_logit, surf, psi, psi_ref, y, groups, calm_mask) -> tuple[dict, dict]:
    """Fit Stage 2 with grouped cross-validation for honest metrics, then on everything."""
    X = stage2_frame(ev, f, ml_logit, surf, psi, psi_ref)
    p_oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        p_oof[te] = LogisticRegression(C=1.0, max_iter=3000).fit(X.values[tr], y[tr]).predict_proba(X.values[te])[:, 1]
    m = LogisticRegression(C=1.0, max_iter=3000).fit(X.values, y)
    calm = X[calm_mask] if calm_mask.sum() > 50 else X
    ref = {c: float(calm[c].median()) for c in X.columns}
    model = {"inputs": list(X.columns), "coef": m.coef_[0], "intercept": float(m.intercept_[0]), "ref": ref}
    s1 = 1 / (1 + np.exp(-np.asarray(ml_logit)))
    metrics = {
        "positives_frac": float(y.mean()),
        "stage1_only": {"roc_auc": float(roc_auc_score(y, s1)), "brier": float(brier_score_loss(y, s1)), "ece": ece(y, s1)},
        "stage1_plus_stage2": {"roc_auc": float(roc_auc_score(y, p_oof)), "brier": float(brier_score_loss(y, p_oof)),
                               "ece": ece(y, p_oof)},
        "reliability": reliability(y, p_oof),
        "odds_ratios": {n: round(float(np.exp(c)), 3) for n, c in zip(X.columns, m.coef_[0])},
    }
    return model, metrics


# ------------------------------------------------------------------ Assetto Corsa Gym domain

def train_acgym():
    data = pd.read_parquet(TIER_A_DATASET)
    tierA = joblib.load(config.WEIGHTS / "tierA_events.joblib")
    feats = tierA["features"]
    tyre = _acgym_tyre_state(data)
    ok = tyre.notna().all(axis=1).values
    data, tyre = data[ok].reset_index(drop=True), tyre[ok].reset_index(drop=True)
    stage1, stage2, metrics = {}, {}, {}
    for ev in EVENTS:
        y = data[ev].values.astype(int)
        # Out-of-fold Stage-1 log-odds (by session) so Stage 2 never sees an over-confident in-sample signal.
        oof = np.zeros(len(data))
        for tr, te in GroupKFold(5).split(data, y, data["file_id"].values):
            m = lgb.LGBMClassifier(**STAGE1_PARAMS).fit(data.iloc[tr][feats], y[tr])
            oof[te] = _stage1_logit(m, data.iloc[te][feats])
        stage1[ev] = lgb.LGBMClassifier(**STAGE1_PARAMS).fit(data[feats], y)
        ax = "f" if ev == "lockup" else "r"
        calm = ((data["brake"] == 0) & (data["throttle"] < 0.5)).values if ev == "lockup" else (data["throttle"] < 0.2).values
        stage2[ev], metrics[ev] = fit_stage2(ev, data, oof, tyre[f"surf_{ax}"], tyre[f"psi_{ax}"],
                                             tyre[f"psi_ref_{ax}"], y, data["file_id"].values, calm)
        log.info("acgym %s: %s", ev, json.dumps({k: v for k, v in metrics[ev].items() if k != "reliability"}, indent=1))
    return {"features": feats, "stage1": stage1, "stage2": {"acgym": stage2}}, {"acgym": metrics}


# ------------------------------------------------------------------ simulator domain (per-car calibration)

def simulate_calibration(runs: int = 24, seconds: float = 150.0, seed: int = 0) -> pd.DataFrame:
    """Mixed driving on the live simulator: tidy laps, late braking, flooring it, safety-car pace, with
    randomised cold pressures and occasional slow punctures, so temperature and pressure effects are visible.
    Labels are the simulator's physics truth (event within the next second)."""
    from sidewall.features import make_features, resample
    from sidewall.models.labels import to_grid
    from sidewall.sources.sim import WHEELS, TyreSim, track_profile

    rng = np.random.default_rng(seed)
    profile = track_profile()
    styles = ["tidy", "tidy", "late_brake", "floor_it", "slow", "both"]
    parts = []
    for run in range(runs):
        sim = TyreSim(profile, seed=int(rng.integers(1e9)))
        sim.crashes = False                                       # a crashed car would stand still: no signal
        for w in WHEELS:                                          # set-ups from 2 psi under to 2 psi over
            sim.p_cold[w] += float(rng.uniform(-2.0, 2.0))
        if rng.random() < 0.3:
            sim.leak_rate[WHEELS[rng.integers(4)]] = float(rng.uniform(0.0005, 0.002))
        warm = float(rng.uniform(0, 90))                          # some runs start straight off the blankets
        rows, meas, truth = [], [], []
        style, style_until = "tidy", 0.0
        while sim.t < seconds + warm:
            if sim.t >= style_until:
                style, style_until = styles[rng.integers(len(styles))], sim.t + float(rng.uniform(6, 14))
            thr, brk = sim._autopilot()
            if style in ("late_brake", "both") and brk > 0:
                brk, thr = 1.0, 0.0
            if style in ("floor_it", "both") and thr > 0.2:
                thr = 1.0
            if style == "slow":
                thr, brk = (0.0, 0.3) if sim.v * 3.6 > 150 else (min(thr, 0.35), brk)
            sim.autopilot = False
            sim.inputs = {"throttle": thr, "brake": brk}
            r = sim.step(0.05)
            m, tr = r.pop("measured"), r.pop("truth")
            if sim.t > warm:
                rows.append(r)
                meas.append(m)
                truth.append({"t": r["t"], "lockup": tr["lockup"], "wheelspin": tr["wheelspin"],
                              "overheat": False, "cold": False})
        S, M, T = pd.DataFrame(rows), pd.DataFrame(meas), pd.DataFrame(truth)
        f = make_features(resample(S), already_resampled=True)
        lab = to_grid(T, f["t"].values, 0.25, 1.0)
        idx = np.clip(np.searchsorted(M["t"].values, f["t"].values, side="right") - 1, 0, len(M) - 1)
        part = f.copy()
        part["lockup"], part["wheelspin"] = lab["lockup"].values, lab["wheelspin"].values
        for ax, ws in (("f", ("fl", "fr")), ("r", ("rl", "rr"))):
            part[f"surf_{ax}"] = M[[f"ir_surf_{w}" for w in ws]].mean(axis=1).values[idx]
            part[f"psi_{ax}"] = M[[f"psi_{w}" for w in ws]].mean(axis=1).values[idx]
            part[f"psi_ref_{ax}"] = float(np.mean([sim.psi_target(w) for w in ws]))
        part["run"] = run
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def calibrate_sim(bundle: dict, runs: int = 24) -> dict:
    data = simulate_calibration(runs)
    stage2, metrics = {}, {}
    for ev in EVENTS:
        y = data[ev].values.astype(int)
        ml = _stage1_logit(bundle["stage1"][ev], data[bundle["features"]])
        ax = "f" if ev == "lockup" else "r"
        calm = ((data["brake"] == 0) & (data["throttle"] < 0.5)).values if ev == "lockup" else (data["throttle"] < 0.2).values
        stage2[ev], metrics[ev] = fit_stage2(ev, data, ml, data[f"surf_{ax}"], data[f"psi_{ax}"],
                                             data[f"psi_ref_{ax}"], y, data["run"].values, calm)
        log.info("sim %s: %s", ev, json.dumps({k: v for k, v in metrics[ev].items() if k != "reliability"}, indent=1))
    bundle["stage2"]["sim"] = stage2
    return metrics


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if "--sim-only" in sys.argv:
        # Recalibrate only the live simulator car (Stage 2 "sim"); the Assetto Corsa models are left as they are.
        bundle, metrics = joblib.load(OUT), json.loads(METRICS.read_text())
    else:
        bundle, metrics = train_acgym()
    metrics["sim"] = calibrate_sim(bundle)
    joblib.dump(bundle, OUT)
    METRICS.write_text(json.dumps(metrics, indent=2))
