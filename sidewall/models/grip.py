"""Grip budget: how much of the tyres' grip the driver is using right now.

A lock-up or wheelspin happens when demand exceeds grip. This module estimates both sides in g:
- available grip: base friction x downforce (grows with speed squared) x temperature window x pressure
  (the same tyre model the simulator uses, so on the live car it is exact and on replays it is an estimate)
- grip in use: the combined acceleration the tyres are transmitting (braking/traction and cornering together)
"""
import numpy as np

MU0_G = 1.7                 # mechanical grip of a warm, correctly inflated slick (g)
DOWNFORCE_REF_MS = 83.0     # speed at which downforce adds DOWNFORCE_GAIN x the mechanical grip
DOWNFORCE_GAIN = 2.5
TEMP_OPT_C, TEMP_WIDTH_C = 100.0, 38.0
PRESSURE_LOSS_PER_PSI = 0.04


def temperature_factor(surf_c):
    """Share of peak grip at this tread temperature (1.0 in the middle of the window, ~0.72 when far off it)."""
    return 0.72 + 0.28 * np.exp(-((np.asarray(surf_c, float) - TEMP_OPT_C) / TEMP_WIDTH_C) ** 2)


def pressure_factor(psi, psi_target):
    """Grip lost to a contact patch that is off its operating pressure (~4 % per psi)."""
    return np.maximum(0.75, 1.0 - PRESSURE_LOSS_PER_PSI * np.abs(np.asarray(psi, float) - np.asarray(psi_target, float)))


def available_g(speed_kmh, surf_c, psi, psi_target):
    v = np.asarray(speed_kmh, float) / 3.6
    downforce = 1.0 + (v / DOWNFORCE_REF_MS) ** 2 * DOWNFORCE_GAIN
    return MU0_G * downforce * temperature_factor(surf_c) * pressure_factor(psi, psi_target)


def budget(speed_kmh: float, ax_g: float, ay_g: float, surf_c: float, psi: float, psi_target: float,
           envelope_g: float | None = None) -> dict:
    """Grip in use vs grip available for one axle, plus how much the tyre condition is costing.

    envelope_g: the car's own demonstrated grip at this speed (its g-g-speed envelope so far). When given, it
    replaces the generic base-grip x downforce term, so a real car is judged against what it has shown it can do.
    """
    use = float(np.hypot(ax_g, ay_g))
    condition = float(temperature_factor(surf_c) * pressure_factor(psi, psi_target))
    if envelope_g:
        ideal = float(envelope_g) / TYPICAL_CONDITION
    else:
        ideal = float(available_g(speed_kmh, TEMP_OPT_C, psi_target, psi_target))
    avail = ideal * condition
    return {"use_g": round(use, 2), "avail_g": round(avail, 2), "pct": round(100 * use / max(avail, 0.1), 0),
            "condition_loss_pct": round(100 * (1 - condition), 0)}


TYPICAL_CONDITION = 0.97    # tyre condition factor behind a demonstrated envelope (tyres near their window)


class Envelope:
    """The car's demonstrated combined grip per 20 km/h speed band, from laps already driven (causal).
    Old peaks fade slowly (0.2 % per second) so the envelope follows the tyres as they wear."""

    BAND_KMH = 20.0

    def __init__(self, bands: int = 20, fade_per_s: float = 0.002):
        self.g = np.zeros(bands)
        self.fade = fade_per_s

    def update(self, dt: float, speed_kmh: float, use_g: float) -> None:
        self.g *= (1 - self.fade) ** dt
        b = min(int(speed_kmh // self.BAND_KMH), len(self.g) - 1)
        self.g[b] = max(self.g[b], use_g)

    def at(self, speed_kmh: float) -> float | None:
        """Envelope at this speed (interpolated from neighbouring bands), None until enough is known."""
        b = min(int(speed_kmh // self.BAND_KMH), len(self.g) - 1)
        vals = [self.g[i] for i in (b - 1, b, b + 1) if 0 <= i < len(self.g) and self.g[i] > 0.5]
        return float(max(vals)) if vals else None
