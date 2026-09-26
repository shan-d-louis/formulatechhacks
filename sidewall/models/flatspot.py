"""Flat-spot detection. There is no labelled flat-spot data, so this is physics plus unsupervised logic.

Two complementary paths:

1. FlatSpotRisk: works on ANY stream, including public F1 data with no vibration channel. A flat spot is
   worn in while a wheel slides locked, so its depth scales with the sliding distance. We integrate the
   lock-up detector's probability x speed over each lock-up episode and attribute it to the likely wheel
   (the unloaded inside front in a corner, both fronts in a straight line).

2. OrderTracker: when a vibration signal and wheel speed exist (the phone sim, the bike rig, a real car's
   hub accelerometer), a flat spot shows up as a once-per-revolution (order 1) vibration. We demodulate the
   vibration at the wheel's own rotation angle, which isolates order 1 no matter how the speed changes, and
   alarm when its amplitude grows well above the wheel's pre-lock-up baseline.
"""
import numpy as np

WHEELS = ("fl", "fr", "rl", "rr")


class FlatSpotRisk:
    """Accumulates estimated locked-sliding distance per front tyre, one lock-up episode at a time.

    Each episode (consecutive frames with a detected lock-up) adds LOCK_S seconds of sliding at the episode's
    speed, weighted by how confidently it was detected above the alert threshold: a borderline detection
    adds almost nothing, an unmistakable one adds the full amount. Flat spots do not heal, so there is no decay.
    """

    LOCK_S = 0.3

    def __init__(self, severity_limit_m: float = 40.0, threshold: float = 0.5):
        # ~40 m of locked sliding (e.g. two long lock-ups from 250 km/h) leaves a flat spot worth pitting for.
        self.limit = severity_limit_m
        self.threshold = threshold
        self.severity = {w: 0.0 for w in WHEELS}
        self._episode = None     # (max p, speed at max, lateral g at max)

    def update(self, dt: float, p_lockup: float, speed_kmh: float, ay_g: float, detected: bool = True) -> dict:
        if detected:
            if self._episode is None or p_lockup > self._episode[0]:
                self._episode = (p_lockup, speed_kmh, ay_g)
        elif self._episode is not None:
            self._close_episode()
        return {w: {"slide_m": v, "risk": min(1.0, v / self.limit), "flat_spot": v >= self.limit}
                for w, v in self.severity.items()}

    def _close_episode(self):
        p, speed_kmh, ay_g = self._episode
        self._episode = None
        confidence = np.clip((p - self.threshold) / (1 - self.threshold), 0, 1)
        slide_m = confidence * self.LOCK_S * speed_kmh / 3.6
        if ay_g > 0.5:        # left-hand corner (car yawing left): the left tyres are the unloaded inside
            share = {"fl": 0.8, "fr": 0.2}
        elif ay_g < -0.5:
            share = {"fl": 0.2, "fr": 0.8}
        else:
            share = {"fl": 0.5, "fr": 0.5}
        for w, s in share.items():
            self.severity[w] += s * slide_m

    def reset(self, wheels=WHEELS):
        for w in wheels:
            self.severity[w] = 0.0


class OrderTracker:
    """Order-1 vibration amplitude of one wheel via demodulation at the wheel's rotation angle.

    Significance: for pure noise of std s over N samples, amp / (s * sqrt(2 / N)) follows a Rayleigh(1)
    distribution, so z > 5 has a per-window false-alarm probability of about 4e-6. We also require the
    amplitude to be well above the wheel's own baseline and the condition to persist for half a window.
    """

    def __init__(self, window_revs: float = 8.0, z_alarm: float = 5.0, alarm_ratio: float = 2.5):
        self.window = window_revs * 2 * np.pi
        self.z_alarm, self.ratio = z_alarm, alarm_ratio
        self.theta = 0.0
        self.buf: list[tuple[float, float]] = []   # (angle, vibration)
        self.baseline = None
        self._baseline_acc: list[float] = []
        self._hot_since = None

    def update(self, dt: float, omega_rad_s: float, vib: float) -> dict:
        return self.update_many(dt, omega_rad_s, [vib])

    def update_many(self, dt: float, omega_rad_s: float, vibs) -> dict:
        """Add several samples taken `dt` apart at a constant wheel speed, then evaluate once."""
        for v in vibs:
            self.theta += omega_rad_s * dt
            self.buf.append((self.theta, v))
        cut = 0
        while cut < len(self.buf) and self.theta - self.buf[cut][0] > self.window:
            cut += 1
        if cut:
            del self.buf[:cut]
        amp, z = self.amplitude()
        # Baseline: mean amplitude over the first two full windows.
        if self.baseline is None and self.theta > self.window:
            self._baseline_acc.append(amp)
            if self.theta > 3 * self.window:
                self.baseline = float(np.mean(self._baseline_acc))
        hot = self.baseline is not None and z > self.z_alarm and amp > self.ratio * self.baseline
        if hot and self._hot_since is None:
            self._hot_since = self.theta
        elif not hot:
            self._hot_since = None
        persistent = self._hot_since is not None and self.theta - self._hot_since > self.window / 2
        return {"order1_amp": amp, "z": z, "ratio": amp / self.baseline if self.baseline else 0.0,
                "flat_spot": bool(persistent)}

    def amplitude(self) -> tuple[float, float]:
        if len(self.buf) < 16:
            return 0.0, 0.0
        th, x = np.array(self.buf).T
        x = x - x.mean()
        # Fourier coefficient at exactly one cycle per revolution.
        amp = float(2 * np.hypot(np.mean(x * np.cos(th)), np.mean(x * np.sin(th))))
        noise = x.std() * np.sqrt(2.0 / len(x)) + 1e-12
        return amp, amp / noise

    def rebaseline(self):
        self.baseline = None
        self._baseline_acc.clear()
        self._hot_since = None
        self.buf.clear()
