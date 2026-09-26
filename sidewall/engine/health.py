"""Tier C: fuse every model output into a per-tyre Tyre Health Index and an escalating pit call.

There are too few real failures to train a meta-model, so this is a transparent, monotone rule set:
each component is a 0..1 risk, combined as 1 - prod(1 - w_i * r_i). The component breakdown is the
explanation shown on the dashboard and read out in the radio call.
"""
from dataclasses import dataclass, field

WHEELS = ("fl", "fr", "rl", "rr")
WHEEL_NAMES = {"fl": "front-left", "fr": "front-right", "rl": "rear-left", "rr": "rear-right"}

WEIGHTS = {
    "cliff": 0.9,        # P(performance cliff within 3 laps)
    "failure": 1.0,      # per-lap tyre failure hazard, scaled
    "flat_spot": 0.8,
    "overheat": 0.6,
    "cold": 0.5,
    "pressure": 1.0,     # gas-mass loss (slow puncture)
    "abuse": 0.4,        # recent lock-up / wheelspin rate
}

CALLS = ["OK", "ADVISE", "BOX THIS LAP", "BOX NOW"]


@dataclass
class TyreState:
    components: dict = field(default_factory=dict)
    health: float = 100.0
    reasons: list = field(default_factory=list)


def tyre_health(components: dict) -> TyreState:
    risk_keep = 1.0
    for k, r in components.items():
        risk_keep *= 1.0 - WEIGHTS.get(k, 0.5) * min(max(r, 0.0), 1.0)
    health = 100.0 * risk_keep
    top = sorted(((WEIGHTS.get(k, 0.5) * v, k) for k, v in components.items()), reverse=True)
    return TyreState(components=components, health=round(health, 1),
                     reasons=[k for s, k in top if s >= 0.15][:3])


def pit_call(tyres: dict[str, TyreState], flags: dict) -> dict:
    """Escalating call with a human-readable reason.

    flags: deflation / slow_puncture / flat_spot (per wheel bool), cliff_alert and failure_alert (0/1/2)
    """
    worst = min(tyres, key=lambda w: tyres[w].health)
    h = tyres[worst].health
    level = 0
    why = []
    if h < 70:
        level = 1
    if h < 45:
        level = 2
    # Lap-level alerts: risk in the top 5 % (advise) / top 1 % (box) of all training laps.
    ca, fa = flags.get("cliff_alert", 0), flags.get("failure_alert", 0)
    if ca:
        level = max(level, ca)
        why.append("cliff risk " + ("top 1 %" if ca == 2 else "top 5 %") + " of laps")
    if fa:
        level = max(level, fa)
        why.append("failure risk " + ("top 1 %" if fa == 2 else "top 5 %") + " of laps")
    for w in WHEELS:
        if flags.get("slow_puncture", {}).get(w):
            level = max(level, 2)
            why.append(f"{WHEEL_NAMES[w]} losing air")
        if flags.get("flat_spot", {}).get(w):
            level = max(level, 2)
            why.append(f"{WHEEL_NAMES[w]} flat spot")
        if flags.get("deflation", {}).get(w):
            level = 3
            why.append(f"{WHEEL_NAMES[w]} deflation")
    for r in tyres[worst].reasons:
        why.append(f"{WHEEL_NAMES[worst]} {r.replace('_', ' ')}")
    # de-duplicate, keep order
    seen, reasons = set(), []
    for r in why:
        if r not in seen:
            seen.add(r)
            reasons.append(r)
    detail = ", ".join(reasons[:2])
    detail = (detail[0].upper() + detail[1:] + ".") if detail else ""
    radio = {
        0: "",
        1: f"Manage the {WHEEL_NAMES[worst]}. {detail}",
        2: f"Box, box. {WHEEL_NAMES[worst].capitalize()} tyre. {detail}",
        3: f"Box now, box now. Bring it in carefully. {detail}",
    }[level].strip()
    coach = flags.get("coach")
    if coach:
        # Preventive: sustained lock-up / wheelspin risk. Tell the driver what to change before it happens.
        name = "Lock-up" if coach["event"] == "lockup" else "Wheelspin"
        tag = f"{name} risk {100 * coach['p_avg']:.0f}% · {coach['top'].lower()}"
        reasons = [tag] + [r for r in reasons if r != tag]
        if level <= 1:
            level = 1
            radio = coach["advice"]
    return {"level": level, "call": CALLS[level], "worst_tyre": worst, "reasons": reasons[:4], "radio": radio,
            "coach": coach}
