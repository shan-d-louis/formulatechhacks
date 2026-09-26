"""Tyre Safety Atlas (Ollon track): data-driven insights from every stint 2018-2025.

- Per circuit and compound: degradation rate, how often stints reach the cliff, and a data-driven stint cap:
  the tyre age at which the Kaplan-Meier probability of having hit the cliff passes 10 %. Stints that
  ended in a planned stop are treated as censored, not as "no cliff".
- Degradation by compound and season, tyre failures vs stint length.

    python -m sidewall.data.build_atlas
"""
import json

import numpy as np
import pandas as pd

from sidewall import config
from sidewall.data.build_stints import OUT as STINTS

OUT = config.PROCESSED / "atlas.json"
COMPOUND_NAME = {0: "softest", 1: "middle", 2: "hardest"}


def kaplan_meier(durations: np.ndarray, events: np.ndarray, max_t: int = 70) -> np.ndarray:
    """S(t) for t = 0..max_t: probability of not yet having hit the event by tyre age t."""
    s, out = 1.0, []
    for t in range(max_t + 1):
        at_risk = np.sum(durations >= t)
        d = np.sum((durations == t) & events)
        if at_risk:
            s *= 1 - d / at_risk
        out.append(s)
    return np.array(out)


def stint_cap(S: np.ndarray, risk: float = 0.10) -> int | None:
    over = np.flatnonzero(S < 1 - risk)
    return int(over[0]) if len(over) else None


def build() -> dict:
    L = pd.read_parquet(STINTS)
    key = ["year", "round", "driver", "stint"]
    st = L[L["stint_usable"]].groupby(key).agg(
        circuit=("circuit", "first"), event=("event", "first"), compound=("compound", "first"),
        compound_rank=("compound_rank", "first"), slope=("deg_slope", "first"),
        end_life=("tyre_life", "max"), track_temp=("track_temp", "mean"))
    cl = L[L["cliff"]].groupby(key)["tyre_life"].min().rename("cliff_life")
    st = st.join(cl)
    st["had_cliff"] = st["cliff_life"].notna()
    st["duration"] = st["cliff_life"].fillna(st["end_life"]).astype(int)

    circuits = []
    for circ, g in st.groupby("circuit"):
        S = kaplan_meier(g["duration"].values, g["had_cliff"].values)
        by_comp = {}
        for rank, gc in g.groupby("compound_rank"):
            if len(gc) < 15:
                continue
            Sc = kaplan_meier(gc["duration"].values, gc["had_cliff"].values)
            by_comp[COMPOUND_NAME.get(int(rank), str(rank))] = {
                "stints": int(len(gc)), "deg_s_per_lap": round(float(gc["slope"].median()), 4),
                "cliff_rate": round(float(gc["had_cliff"].mean()), 3), "stint_cap_10pct": stint_cap(Sc)}
        circuits.append({
            "circuit": circ, "event": g["event"].iloc[-1], "stints": int(len(g)), "seasons": int(g.index.get_level_values(0).nunique()),
            "deg_s_per_lap": round(float(g["slope"].median()), 4),
            "cliff_rate": round(float(g["had_cliff"].mean()), 3),
            "median_cliff_life": None if not g["had_cliff"].any() else float(g.loc[g["had_cliff"], "cliff_life"].median()),
            "stint_cap_10pct": stint_cap(S), "survival": [round(float(x), 4) for x in S[:61]],
            "by_compound": by_comp,
        })
    circuits.sort(key=lambda c: (c["stint_cap_10pct"] is None, c["stint_cap_10pct"] or 999))

    # Degradation by compound family and season.
    deg = (st.groupby([st.index.get_level_values(0), "compound_rank"])["slope"].median()
           .unstack().round(4))
    deg_by_year = {int(y): {COMPOUND_NAME.get(int(k), str(k)): (None if pd.isna(v) else float(v)) for k, v in row.items()}
                   for y, row in deg.iterrows()}

    # Tyre failures vs stint age.
    fails = L[L["failure"]]
    all_life = L[L["compound"].notna()]["tyre_life"].dropna()
    bins = list(range(0, 61, 5))
    fail_hist = np.histogram(fails["tyre_life"].dropna(), bins=bins)[0]
    lap_hist = np.histogram(all_life, bins=bins)[0]
    failure_rate = [None if n == 0 else round(1000 * f / n, 3) for f, n in zip(fail_hist, lap_hist)]
    failures = fails[["year", "event", "driver", "lap", "compound", "tyre_life", "lap_time_s", "official_failure"]]

    qatar = next((c for c in circuits if "Lusail" in c["circuit"] or "Qatar" in c["event"]), None)
    return {
        "summary": {"stints": int(len(st)), "laps": int(len(L)), "races": int(L.groupby(["year", "round"]).ngroups),
                    "seasons": sorted(int(y) for y in L["year"].unique()),
                    "cliffs": int(st["had_cliff"].sum()), "failures": int(L["failure"].sum()),
                    "official_failures": int(L["official_failure"].sum())},
        "circuits": circuits,
        "deg_by_year": deg_by_year,
        "failure_rate_per_1000_laps_by_age": {"bins": bins, "rate": failure_rate,
                                              "failures": fail_hist.tolist(), "laps": lap_hist.tolist()},
        "failures": json.loads(failures.to_json(orient="records")),
        "qatar": None if qatar is None else {"fia_cap_2023": 18, "fia_cap_2025": 25,
                                             "model_cap_10pct": qatar["stint_cap_10pct"], "circuit": qatar["circuit"]},
    }


if __name__ == "__main__":
    atlas = build()
    OUT.write_text(json.dumps(atlas, indent=1, default=str))
    s = atlas["summary"]
    print(s)
    for c in atlas["circuits"][:10]:
        print(f'{c["circuit"]:<22} cap {c["stint_cap_10pct"]}  cliff rate {c["cliff_rate"]}  deg {c["deg_s_per_lap"]}')
    print("qatar:", atlas["qatar"])
