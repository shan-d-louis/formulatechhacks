"""Train the laps-remaining models and save training/laps_model.joblib.

1. Baseline: per-compound curve delta = a + b*age + c*age^2. If the fit bends downward
   (c < 0) it is refit as a straight line, since tires don't get faster with age.
   Residual quantiles give it a low/high band.
2. Main: per compound, three LightGBM quantile models (0.1, 0.5, 0.9), monotone +1 on age.
   Trees can't extrapolate, and teams pit before the cliff, so past the oldest age seen
   for a compound the prediction continues with the baseline's slope.
Evaluation discipline (no race is ever in two places):
  - races are split chronologically: the oldest ~60% train, the next ~20% validate, the newest ~20% test
  - hyper-parameters (tree size, learning rate, rounds, track temp on/off) are chosen on validation only
  - the chosen settings are refit on train + validation, then scored ONCE on the test races
  - the per-compound age cap is computed from the fitting data only
The shipped model is the train + validation fit: it has never seen a test race.

The saved bundle is a plain dict (LightGBM models as model strings) so backend/laps.py
can use it without importing this file. Laps remaining = first future age where
predicted delta > CLIFF_DELTA_S.

Usage:
    python train.py
"""

import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
import config as cfg  # noqa: E402  - thresholds shared with the live backend

DATA_PATH = HERE / "data" / "train.parquet"
MODEL_PATH = HERE / "laps_model.joblib"
PLOT_DIR = HERE / "plots"

Q_LO, Q_MID, Q_HI = cfg.QUANTILES
VAL_FRAC, TEST_FRAC = 0.2, 0.2  # share of races (newest last) held out for validation and for the final test
SPLITS = ("train", "val", "test")
# Candidates scored on the validation races; the lowest mean pinball loss wins.
TUNING_GRID = [
    {"num_leaves": nl, "min_data_in_leaf": md, "learning_rate": lr, "rounds": n, "use_temp": ut}
    for nl in (4, 7, 15) for md in (50, 100, 200) for lr, n in ((0.03, 400), (0.05, 200)) for ut in (False, True)
]
# Track temp is nearly constant within a race, so trees can use it to memorise races.
# Default for direct fit_lgbm calls; the pipeline tunes it (on/off) on the validation races.
USE_TRACK_TEMP = False
LGBM_PARAMS = {
    # Built-in "quantile" objective refuses monotone constraints, so pinball loss is a custom objective.
    "learning_rate": 0.03, "num_leaves": 7, "min_data_in_leaf": 100,
    "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0,
}
LGBM_ROUNDS = 400

# Chart colors: reference categorical slots 1-2 + neutral inks (dataviz skill palette).
C_LGBM, C_BASE = "#2a78d6", "#eb6834"
C_ACTUAL, C_INK, C_MUTED, C_GRID, C_SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


# ---------- baseline ----------

def fit_baseline(df: pd.DataFrame) -> dict:
    """Per-compound convex quadratic (or linear) in tire age, plus residual quantiles for the band."""
    curves = {}
    for c in cfg.DRY_COMPOUNDS:
        sub = df[df["Compound"] == c]
        age, y = sub["TyreLife"].to_numpy(float), sub["Delta_s"].to_numpy(float)
        coef = np.polyfit(age, y, 2)
        if coef[0] < 0:  # bends downward: refit linear
            coef = np.concatenate([[0.0], np.polyfit(age, y, 1)])
        resid = y - np.polyval(coef, age)
        curves[c] = {
            "coef": [float(x) for x in coef],  # np.polyval order: [c, b, a]
            "resid_lo": float(np.quantile(resid, Q_LO)),
            "resid_hi": float(np.quantile(resid, Q_HI)),
        }
    return curves


def predict_baseline(curves: dict, compound: pd.Series, age: pd.Series) -> np.ndarray:
    """Returns an (n, 3) array of low/mid/high delta predictions."""
    out = np.zeros((len(age), 3))
    for c, p in curves.items():
        m = (compound == c).to_numpy()
        mid = np.polyval(p["coef"], age.to_numpy(float)[m])
        out[m] = np.column_stack([mid + p["resid_lo"], mid, mid + p["resid_hi"]])
    return out


def baseline_slope(curve: dict, age: float) -> float:
    """d(delta)/d(age) of a baseline curve, floored at 0."""
    c, b, _ = curve["coef"]
    return max(0.0, 2 * c * age + b)


# ---------- LightGBM ----------

def make_features(age, track_temp, use_temp: bool) -> pd.DataFrame:
    """Model inputs. Only what the live system knows: tire age (and optionally track temp)."""
    X = pd.DataFrame({"TyreLife": np.asarray(age, float)})
    if use_temp:
        X["TrackTemp"] = np.asarray(track_temp, float)
    return X


def pinball_objective(q: float):
    """Gradient/hessian of the pinball (quantile) loss for LightGBM's custom-objective API."""
    def objective(preds: np.ndarray, data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        y = data.get_label()
        return np.where(y > preds, -q, 1.0 - q), np.ones_like(preds)
    return objective


def fit_lgbm(df: pd.DataFrame, use_temp: bool = USE_TRACK_TEMP, params: dict | None = None,
             rounds: int = LGBM_ROUNDS) -> dict:
    """Per compound: monotone quantile boosters + the max age seen + the baseline for the tail.

    `params` overrides LGBM_PARAMS (tuned on the validation races). Boosters are stored as
    {"model_str", "init_score"} so the bundle loads without this module.
    """
    model = {"use_temp": use_temp, "baseline": fit_baseline(df), "max_age": {}, "lgbm": {}, "_boosters": {}}
    for c in cfg.DRY_COMPOUNDS:
        sub = df[df["Compound"] == c]
        X = make_features(sub["TyreLife"], sub["TrackTemp"], use_temp)
        y = sub["Delta_s"].to_numpy(float)
        params_c = {**LGBM_PARAMS, **(params or {}), "monotone_constraints": [1] + [0] * (X.shape[1] - 1)}
        model["max_age"][c] = float(sub["TyreLife"].max())
        model["lgbm"][c], model["_boosters"][c] = {}, {}
        for q in cfg.QUANTILES:
            init = float(np.quantile(y, q))  # start at the unconditional quantile, trees learn the rest
            data = lgb.Dataset(X, y, init_score=np.full(len(y), init))
            booster = lgb.train({**params_c, "objective": pinball_objective(q)}, data, rounds)
            model["lgbm"][c][q] = {"model_str": booster.model_to_string(), "init_score": init}
            model["_boosters"][c][q] = booster
    return model


def predict_lgbm(model: dict, compound, age, track_temp) -> np.ndarray:
    """(n, 3) low/mid/high delta. Beyond max_age, extend linearly with the baseline slope."""
    boosters = model.setdefault("_boosters", {})
    compound, age, temp = np.asarray(compound), np.asarray(age, float), np.asarray(track_temp, float)
    out = np.zeros((len(age), 3))
    for c in cfg.DRY_COMPOUNDS:
        m = compound == c
        if not m.any():
            continue
        max_age = model["max_age"][c]
        X = make_features(np.minimum(age[m], max_age), temp[m], model["use_temp"])
        tail = baseline_slope(model["baseline"][c], max_age) * np.maximum(0.0, age[m] - max_age)
        cols = []
        for q in cfg.QUANTILES:
            spec = model["lgbm"][c][q]
            b = boosters.setdefault(c, {}).get(q) or lgb.Booster(model_str=spec["model_str"])
            boosters[c][q] = b
            cols.append(b.predict(X) + spec["init_score"] + tail)
        out[m] = np.sort(np.column_stack(cols), axis=1)  # quantiles never cross
    return out


# ---------- evaluation ----------

def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    """MAE of the median, 80% interval coverage and mean width."""
    return {
        "mae": float(np.mean(np.abs(pred[:, 1] - y))),
        "coverage": float(np.mean((y >= pred[:, 0]) & (y <= pred[:, 2]))),
        "width": float(np.mean(pred[:, 2] - pred[:, 0])),
    }


def pinball(pred: np.ndarray, y: np.ndarray) -> float:
    """Mean pinball (quantile) loss over the low/mid/high columns: the tuning objective."""
    losses = []
    for j, q in enumerate(cfg.QUANTILES):
        e = y - pred[:, j]
        losses.append(np.mean(np.maximum(q * e, (q - 1) * e)))
    return float(np.mean(losses))


# ---------- splits: whole races, oldest -> newest, never overlapping ----------

def chronological_split(df: pd.DataFrame, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC) -> pd.Series:
    """Label every lap 'train' / 'val' / 'test' by its race, oldest races first. Each part gets >= 1 race."""
    races = df.drop_duplicates("RaceId").sort_values(["Year", "Round"])["RaceId"].tolist()
    n = len(races)
    if n < 3:
        raise ValueError(f"need at least 3 races for train/val/test, got {n}")
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(f"{n} races is too few for val_frac={val_frac}, test_frac={test_frac}")
    label = {r: ("train" if i < n_train else "val" if i < n_train + n_val else "test") for i, r in enumerate(races)}
    return df["RaceId"].map(label).rename("split")


def assert_no_overlap(df: pd.DataFrame, part: pd.Series) -> None:
    """Raise if any race, stint or lap is in more than one split, or if the splits are out of time order."""
    tagged = df.assign(_split=part.values)
    for key in (["RaceId"], ["RaceId", "Driver", "Stint"], ["RaceId", "Driver", "Stint", "TyreLife"]):
        n_parts = tagged.groupby(key)["_split"].nunique()
        if (n_parts > 1).any():
            raise ValueError(f"split overlap on {key}: {n_parts[n_parts > 1].index[:3].tolist()}")
    order = tagged.drop_duplicates("RaceId").sort_values(["Year", "Round"])["_split"].map(SPLITS.index)
    if not order.is_monotonic_increasing:
        raise ValueError("split overlap in time: a later race is in an earlier split")


def age_caps(fit_df: pd.DataFrame) -> pd.Series:
    """Per-compound tyre-age cap for training (rare managed stints bias the curve), from fitting data only."""
    return fit_df.groupby("Compound")["TyreLife"].quantile(cfg.TRAIN_AGE_QUANTILE)


def capped(fit_df: pd.DataFrame) -> pd.DataFrame:
    over = fit_df["TyreLife"] > fit_df["Compound"].map(age_caps(fit_df))
    return fit_df[~over]


def predictions(fit_df: pd.DataFrame, eval_df: pd.DataFrame, params: dict) -> dict:
    """Fit on fit_df (age-capped), predict eval_df. Returns {name: (n, 3) low/mid/high} + the fitted model."""
    fit_df = capped(fit_df)
    p = {k: v for k, v in params.items() if k not in ("rounds", "use_temp")}
    model = fit_lgbm(fit_df, params["use_temp"], p, params["rounds"])
    args = (eval_df["Compound"], eval_df["TyreLife"], eval_df["TrackTemp"])
    const = float(np.median(fit_df["Delta_s"]))
    return {"lgbm": predict_lgbm(model, *args),
            "base": predict_baseline(fit_baseline(fit_df), eval_df["Compound"], eval_df["TyreLife"]),
            "const": np.column_stack([np.full(len(eval_df), const)] * 3), "_model": model}


def tune(train_df: pd.DataFrame, val_df: pd.DataFrame, grid: list[dict]) -> list[dict]:
    """Score every candidate: fit on the training races, measure on the validation races."""
    y = val_df["Delta_s"].to_numpy(float)
    rows = []
    for params in grid:
        pred = predictions(train_df, val_df, params)["lgbm"]
        rows.append({"params": params, "val_pinball": pinball(pred, y),
                     **{f"val_{k}": v for k, v in scores(pred, y).items()}})
    return rows


def run(df: pd.DataFrame, grid: list[dict] | None = None, val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC,
        plot_path: Path | None = None) -> dict:
    """Split -> tune on validation -> refit on train + validation -> score once on test."""
    grid = grid or TUNING_GRID
    part = chronological_split(df, val_frac, test_frac)
    assert_no_overlap(df, part)
    tr, va, te = (df[(part == s).values] for s in SPLITS)
    races = {s: sorted(d["RaceId"].unique().tolist()) for s, d in zip(SPLITS, (tr, va, te))}
    print(f"Split by race, oldest to newest: train {len(races['train'])} races ({len(tr)} laps), "
          f"validation {len(races['val'])} ({len(va)}), test {len(races['test'])} ({len(te)})")
    print(f"  train {races['train'][0]} .. {races['train'][-1]} | val {races['val'][0]} .. {races['val'][-1]} | "
          f"test {races['test'][0]} .. {races['test'][-1]}")

    tuning = tune(tr, va, grid)
    tuning.sort(key=lambda r: r["val_pinball"])
    chosen = tuning[0]["params"]
    print(f"\nTuned {len(grid)} candidates on the validation races (lower pinball = better):")
    print(f"  {'leaves':>6} {'min_leaf':>8} {'lr':>5} {'rounds':>6} {'temp':>5} | {'pinball':>7} {'MAE':>6} {'cov':>5}")
    for r in tuning[:5]:
        q = r["params"]
        print(f"  {q['num_leaves']:>6} {q['min_data_in_leaf']:>8} {q['learning_rate']:>5} {q['rounds']:>6} "
              f"{str(q['use_temp']):>5} | {r['val_pinball']:>7.4f} {r['val_mae']:>6.3f} {r['val_coverage']:>5.0%}")

    fit_df = pd.concat([tr, va])
    fit_races = sorted(fit_df["RaceId"].unique().tolist())
    if set(fit_races) & set(races["test"]):
        raise ValueError("split overlap: a test race reached the final fit")
    pred = predictions(fit_df, te, chosen)
    y = te["Delta_s"].to_numpy(float)
    test = {"lightgbm": {**scores(pred["lgbm"], y), "pinball": pinball(pred["lgbm"], y)},
            "baseline": {**scores(pred["base"], y), "pinball": pinball(pred["base"], y)},
            "constant": {"mae": float(np.mean(np.abs(y - pred["const"][:, 1])))}}
    if plot_path is not None:
        held = te.copy()
        for name in ("base", "lgbm"):
            held[f"{name}_lo"], held[f"{name}_mid"], held[f"{name}_hi"] = pred[name].T
        plot_stints(held, plot_path)
    return {"split": races, "fit_races": fit_races, "tuning": tuning, "chosen": chosen, "test": test,
            "model": pred["_model"], "fit_laps": int(len(capped(fit_df))), "caps": age_caps(fit_df).to_dict()}


# ---------- laps remaining (the live rule, as a sanity check) ----------

def laps_remaining(model: dict, compound: str, age: float, track_temp: float) -> dict:
    """Roll forward from `age`; first future age whose delta > cliff, per quantile band."""
    ages = age + np.arange(1, cfg.MAX_FORECAST_LAPS + 1, dtype=float)
    n = len(ages)
    pred = predict_lgbm(model, [compound] * n, ages, [track_temp] * n)
    first = lambda col: next((i + 1 for i, d in enumerate(pred[:, col]) if d > cfg.CLIFF_DELTA_S),
                             cfg.MAX_FORECAST_LAPS)
    # Pessimistic (high delta) quantile gives the low laps estimate, and vice versa.
    return {"low": first(2), "mid": first(1), "high": first(0)}


# ---------- chart ----------

def pick_stints(df: pd.DataFrame, n: int = 4) -> list[tuple]:
    """Longest stint per compound, then the next longest overall, from different races."""
    keys = ["RaceId", "Driver", "Stint"]
    stints = df.groupby(keys).agg(Compound=("Compound", "first"), n=("Delta_s", "size")).reset_index()
    stints = stints.sort_values("n", ascending=False)
    chosen, races = [], set()
    for c in cfg.DRY_COMPOUNDS:
        row = stints[(stints["Compound"] == c) & ~stints["RaceId"].isin(races)].head(1)
        if len(row):
            chosen.append(tuple(row.iloc[0][keys]))
            races.add(row.iloc[0]["RaceId"])
    for _, row in stints.iterrows():
        if len(chosen) >= n:
            break
        if row["RaceId"] not in races:
            chosen.append(tuple(row[keys]))
            races.add(row["RaceId"])
    return chosen


def plot_stints(oof: pd.DataFrame, path: Path) -> None:
    """Small multiples: actual delta vs out-of-fold LightGBM band + baseline, per stint."""
    stints = pick_stints(oof)
    ncols = 2
    nrows = int(np.ceil(len(stints) / ncols))
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), sharey=True, facecolor=C_SURFACE)
    axes = np.atleast_1d(axes).ravel()
    indexed = oof.set_index(["RaceId", "Driver", "Stint"])

    for ax, key in zip(axes, stints):
        s = indexed.loc[key].sort_values("TyreLife")
        x = s["TyreLife"]
        ax.set_facecolor(C_SURFACE)
        ax.fill_between(x, s["lgbm_lo"], s["lgbm_hi"], color=C_LGBM, alpha=0.15, linewidth=0,
                        label="LightGBM 80% band")
        ax.plot(x, s["base_mid"], color=C_BASE, lw=2, ls="--", label="Baseline curve")
        ax.plot(x, s["lgbm_mid"], color=C_LGBM, lw=2, label="LightGBM median")
        ax.scatter(x, s["Delta_s"], s=22, color=C_ACTUAL, edgecolor=C_SURFACE, linewidth=0.8, zorder=3,
                   label="Actual lap")
        ax.axhline(cfg.CLIFF_DELTA_S, color=C_MUTED, lw=1, ls=":")
        ax.text(x.iloc[0], cfg.CLIFF_DELTA_S, f" cliff {cfg.CLIFF_DELTA_S:g} s", color=C_MUTED,
                va="bottom", fontsize=9)
        event = s["EventName"].iloc[0].replace(" Grand Prix", "")
        year = s["Year"].iloc[0]
        ax.set_title(f"{year} {event} · {key[1]} · {s['Compound'].iloc[0].title()} · stint {key[2]}",
                     loc="left", color=C_INK, fontsize=12)
        ax.grid(axis="y", color=C_GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(C_GRID)
        ax.tick_params(colors=C_MUTED)
        ax.set_xlabel("Tire age (laps)", color=C_MUTED)
    for ax in axes[len(stints):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("Lap-time delta vs. stint start (s)", color=C_MUTED)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=4, frameon=False, labelcolor=C_INK,
               bbox_to_anchor=(0.005, 0.96))
    fig.suptitle("Tire degradation: predicted vs. actual (held-out TEST races)", x=0.01, y=0.99,
                 ha="left", fontsize=14, color=C_INK, fontweight="bold")
    fig.text(0.01, 0.005, "Fuel-corrected lap times, FastF1 dry races. The model was fit on earlier races and "
             "never saw these test races; its settings were tuned on separate validation races.",
             color=C_MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.935))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=C_SURFACE)
    plt.close(fig)


# ---------- main ----------

def main() -> None:
    if not DATA_PATH.exists():
        sys.exit(f"{DATA_PATH} not found. Run fetch.py and prepare.py first.")
    df = pd.read_parquet(DATA_PATH)
    df = df[df["TrackTemp"].notna()].reset_index(drop=True)

    result = run(df, plot_path=PLOT_DIR / "degradation_stints.png")
    t = result["test"]
    print("\nTEST races (never used for fitting or tuning), target: lap-time delta, s")
    print(f"  {'model':<30} {'MAE':>7} {'80% coverage':>13} {'band width':>11}")
    print(f"  {'constant median':<30} {t['constant']['mae']:>7.3f} {'-':>13} {'-':>11}")
    for name, key in (("baseline (per-compound curve)", "baseline"), ("LightGBM quantile (shipped)", "lightgbm")):
        r = t[key]
        print(f"  {name:<30} {r['mae']:>7.3f} {r['coverage']:>13.1%} {r['width']:>10.2f}s")
    print(f"  LightGBM vs baseline MAE: {(t['lightgbm']['mae'] - t['baseline']['mae']) / t['baseline']['mae']:+.1%}")

    model = result["model"]
    bundle = {
        "version": 3,
        "compounds": list(cfg.DRY_COMPOUNDS),
        "quantiles": list(cfg.QUANTILES),
        "features": list(make_features([0.0], [0.0], model["use_temp"]).columns),
        "age_feature": "TyreLife: laps on the set incl. the current one (1 on a fresh set's first lap)",
        "baseline": model["baseline"],
        "max_age": model["max_age"],
        "lgbm": model["lgbm"],
        "cliff_delta_s": cfg.CLIFF_DELTA_S,
        "split": result["split"],               # race ids per split; the model was fit on train + val only
        "chosen_params": result["chosen"],
        "tuning": result["tuning"],
        "test": result["test"],
        "cv": result["test"],                   # kept for older readers of the bundle
        "n_laps": result["fit_laps"],
        "n_races": len(result["fit_races"]),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved {MODEL_PATH} (fit on {len(result['fit_races'])} train + validation races, "
          f"{result['fit_laps']} laps; chosen {result['chosen']})")
    for c, p in model["baseline"].items():
        kind = "quadratic" if p["coef"][0] else "linear"
        print(f"  baseline {c:<7} {kind:<9} coef {np.round(p['coef'], 4).tolist()}  "
              f"max age seen {model['max_age'][c]:.0f}")

    temp = float(df["TrackTemp"].median())
    print(f"\nSanity check, laps remaining on a fresh set (age 1) at {temp:.0f} C track:")
    for c in cfg.DRY_COMPOUNDS:
        print(f"  {c:<7} {laps_remaining(model, c, 1.0, temp)}")
    print(f"Saved {PLOT_DIR / 'degradation_stints.png'} (held-out test races)")


if __name__ == "__main__":
    main()
