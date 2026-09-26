"""Train / validation / test discipline for the laps model (training/train.py).

- splits are by whole race and chronological: every validation race is after every training race, and every
  test race is after every validation race
- no race, stint or lap appears in two splits
- the per-compound age cap and all hyper-parameter choices use training (and validation) data only; the test
  races are touched exactly once, for the final report
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import train  # noqa: E402


def synthetic_laps(n_years=2, rounds=10, drivers=4, laps_per_stint=18, seed=0) -> pd.DataFrame:
    """Tiny dataset with a known degradation slope per compound."""
    rng = np.random.default_rng(seed)
    rows = []
    slope = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03}
    for y in range(2024, 2024 + n_years):
        for r in range(1, rounds + 1):
            for d in range(drivers):
                for s, comp in enumerate(("SOFT", "MEDIUM", "HARD"), 1):
                    for age in range(1, laps_per_stint + 1):
                        rows.append({"RaceId": f"{y}_{r:02d}", "Year": y, "Round": r, "EventName": f"GP {r}",
                                     "Driver": f"D{d}", "Stint": s, "Compound": comp, "TyreLife": float(age),
                                     "TrackTemp": 30 + rng.normal(0, 3),
                                     "Delta_s": slope[comp] * age + rng.normal(0, 0.2)})
    return pd.DataFrame(rows)


def test_split_is_by_race_and_chronological():
    df = synthetic_laps()
    part = train.chronological_split(df, val_frac=0.2, test_frac=0.2)
    races = df.assign(part=part.values).groupby("RaceId")["part"].agg(set)
    assert (races.map(len) == 1).all()                                   # a race never straddles two splits
    order = df.drop_duplicates("RaceId").sort_values(["Year", "Round"])
    seq = order["RaceId"].map(races.map(lambda s: next(iter(s)))).tolist()
    assert seq == sorted(seq, key=["train", "val", "test"].index)        # train < val < test in time
    counts = pd.Series(seq).value_counts()
    assert counts["train"] == 12 and counts["val"] == 4 and counts["test"] == 4


def test_split_guard_rejects_any_overlap():
    df = synthetic_laps(n_years=1, rounds=6)
    part = train.chronological_split(df, val_frac=0.2, test_frac=0.2)
    train.assert_no_overlap(df, part)                                    # the real split passes
    leaked = part.copy()
    first_test = df.index[(part == "test").values][0]
    leaked.loc[first_test] = "train"                                     # one test lap leaks into training
    with pytest.raises(ValueError, match="overlap"):
        train.assert_no_overlap(df, leaked)


def test_age_caps_ignore_validation_and_test_laps():
    df = synthetic_laps()
    part = train.chronological_split(df, val_frac=0.2, test_frac=0.2)
    caps = train.age_caps(df[part == "train"])
    extreme = df.copy()
    extreme.loc[(part != "train").values, "TyreLife"] = 999.0            # absurd ages outside training
    assert train.age_caps(extreme[part == "train"]).equals(caps)


def test_pipeline_tunes_on_validation_and_never_fits_on_test(tmp_path):
    df = synthetic_laps()
    grid = [{"num_leaves": 4, "min_data_in_leaf": 20, "learning_rate": 0.1, "rounds": 30, "use_temp": False},
            {"num_leaves": 7, "min_data_in_leaf": 20, "learning_rate": 0.1, "rounds": 30, "use_temp": True}]
    result = train.run(df, grid=grid, val_frac=0.2, test_frac=0.2, plot_path=None)
    test_races = set(result["split"]["test"])
    assert test_races and not test_races & set(result["fit_races"])     # final model never saw a test race
    assert set(result["fit_races"]) == set(result["split"]["train"]) | set(result["split"]["val"])
    assert len(result["tuning"]) == len(grid)                           # every candidate scored on validation
    assert result["chosen"] == min(result["tuning"], key=lambda r: r["val_pinball"])["params"]
    assert {"baseline", "lightgbm", "constant"} <= set(result["test"])  # one final report on the test races
    assert 0.0 <= result["test"]["lightgbm"]["coverage"] <= 1.0
