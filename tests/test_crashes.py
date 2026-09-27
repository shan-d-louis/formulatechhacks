"""Crashes: the live car really stops, then is recovered; replays mark the real failures from the telemetry."""
import asyncio
import json

import pytest

from sidewall.sources.sim import TyreSim, track_profile


@pytest.fixture(scope="module")
def profile():
    return track_profile()


def test_tyre_that_lost_too_much_air_fails_and_the_car_stops(profile):
    sim = TyreSim(profile)
    for _ in range(200):
        sim.step(0.05)
    sim.gas_mass["rl"] = 0.55                     # a leak nobody boxed for
    r = sim.step(0.05)
    assert sim.crashed and sim.crashed["kind"] == "tyre_failure" and sim.crashed["wheel"] == "rl"
    assert r["truth"]["crash"]
    x0, s0 = r["x"], sim.s
    for _ in range(40):                           # two seconds later it has not moved
        r = sim.step(0.05)
    assert sim.v == 0 and sim.s == s0 and r["x"] == x0 and r["speed_kmh"] == 0


def test_too_fast_for_a_corner_puts_the_car_off(profile):
    sim = TyreSim(profile)
    sim.autopilot = False
    sim.script_kappa = 0.02                       # a tight virtual corner
    sim.v = 60.0                                  # far too fast for it
    sim.step(0.05)
    assert sim.crashed and sim.crashed["kind"] == "off"


def test_tidy_driving_never_crashes(profile):
    sim = TyreSim(profile)
    for _ in range(20 * 180):                     # three minutes of autopilot
        sim.step(0.05)
    assert sim.crashed is None


def test_live_session_announces_the_crash_then_recovers():
    from sidewall.engine.monitor import load_bundles
    from sidewall.sources.live import LiveSession, SIM_HZ, CRASH_HOLD_S

    async def main():
        msgs = []

        async def pub(ch, m):
            msgs.append((ch, m))

        live = LiveSession(load_bundles(), pub)
        for _ in range(100):
            await live._step_physics()
        live.start_scenario("lockup")
        live.sim.gas_mass["fl"] = 0.5
        await live._step_physics()
        crashes = [m for ch, m in msgs if m.get("type") == "crash"]
        assert {ch for ch, m in msgs if m.get("type") == "crash"} == {"pitwall", "driver"}
        assert crashes[0]["kind"] == "tyre_failure" and live.scenario is None
        for _ in range(int((CRASH_HOLD_S + 1) * SIM_HZ)):
            await live._step_physics()
        assert live.sim.crashed is None and live.sim.v > 0            # back out on fresh tyres
        assert any(m.get("type") == "recovered" for _, m in msgs)

    asyncio.run(main())


@pytest.mark.parametrize("key, kind, lap", [("baku2021", "crash", 46), ("silverstone2020", "failure", 52)])
def test_replays_mark_the_real_failure(key, kind, lap):
    from sidewall.sources import replay
    path = replay.REPLAY_CACHE / f"{key}.json"
    if not path.exists():
        pytest.skip("replay not built on this machine")
    frames = json.loads(path.read_text())["frames"]
    ev = replay.failure_event(frames, replay.SCENARIOS[key])
    assert ev and ev["kind"] == kind and ev["lap"] == lap          # Verstappen stopped; Hamilton limped on
    assert ev["speed_before_kmh"] > 200
