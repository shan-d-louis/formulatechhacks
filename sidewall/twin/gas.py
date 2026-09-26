"""Tyre gas physics: pressure from temperature, and the leak invariant.

At constant gas mass and volume, P_abs / T_abs is constant (ideal gas). Heat cycles change P and T together
but not their ratio; only a change in gas mass (a leak) does. So m_hat = P_abs / T_abs, normalised to its
value at the start of the stint, is a temperature-independent leak detector.
"""
import numpy as np

ATM_PSI = 14.696
KELVIN = 273.15


def hot_pressure(p_cold_psi: float, t_cold_c: float, t_gas_c):
    """Gauge pressure after the gas warms from t_cold_c to t_gas_c (no leak)."""
    return (p_cold_psi + ATM_PSI) * (np.asarray(t_gas_c) + KELVIN) / (t_cold_c + KELVIN) - ATM_PSI


def mass_index(p_psi, t_gas_c):
    """Gas-mass invariant, proportional to the moles of gas in the tyre."""
    return (np.asarray(p_psi) + ATM_PSI) / (np.asarray(t_gas_c) + KELVIN)


class LeakDetector:
    """One-sided CUSUM on the relative drop of the gas-mass index.

    `drift` is the tolerated noise per sample, `limit` the accumulated drop that triggers an alarm.
    A sudden deflation (gas escaping faster than `burst_frac_s` of the reference per second) alarms at once.
    Both use the temperature-compensated mass index, so heating and cooling can never trigger them.
    """

    def __init__(self, drift: float = 0.0015, limit: float = 0.02, burst_frac_s: float = 0.02):
        self.drift, self.limit, self.burst = drift, limit, burst_frac_s
        self.ref = None
        self.cusum = 0.0
        self.prev = None

    def update(self, t: float, p_psi: float, t_gas_c: float) -> dict:
        m = float(mass_index(p_psi, t_gas_c))
        if self.ref is None:
            self.ref = m
        loss = (self.ref - m) / self.ref                  # fraction of gas lost since the start
        self.cusum = max(0.0, self.cusum + loss - self.drift)
        rate = 0.0
        if self.prev is not None and t > self.prev[0]:
            rate = (self.prev[1] - m) / self.ref / (t - self.prev[0])
        self.prev = (t, m)
        return {"mass_loss_frac": loss, "cusum": self.cusum,
                "slow_puncture": self.cusum > self.limit, "deflation": rate > self.burst}
