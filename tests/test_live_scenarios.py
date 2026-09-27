"""Driver-page scenario buttons: each one forces the live car into its hazard, and the pit wall sees it.

The scripts are simulator/scenarios.json (shared with the browser simulator and the backend tests).
"""
import asyncio

import pytest

from sidewall.sources.scenarios import ScenarioRunner, ScenarioWatch
from sidewall.sources.sim import TyreSim, track_profile, WHEELS
from sidewall.models.labels import HOT_SURFACE_C

DT = 0.05


@pytest.fixture(scope="module")
def profile():
    return track_profile()


def run_script(profile, key, warm_ticks=400):
    """Drive the script headless after some tidy autopilot laps; return per-step hazard counts and the sim."""
    sim = TyreSim(profile)
    for _ in range(warm_ticks):
        sim.step(DT)
    sim.autopilot = False
    run = ScenarioRunner(key, sim)
    per, t = {}, 0.0
    while not run.done and t < 200:
        step = run.index + 1
        run.tick(DT)
        truth = sim.step(DT)["truth"]
        t += DT
        d = per.setdefault(step, {"lockup": 0, "wheelspin": 0, "hot": 0})
        d["lockup"] += truth["lockup"]
        d["wheelspin"] += truth["wheelspin"]
        d["hot"] += max(sim.t_surf.values()) >= HOT_SURFACE_C
    return per, sim, run


def test_button_names_map_to_the_scripts():
    sim = TyreSim(track_profile())
    assert ScenarioRunner("pressure", sim).key == "puncture"
    assert ScenarioRunner("overheat", sim).key == "corner"
    with pytest.raises(KeyError):
        ScenarioRunner("nonsense", sim)


@pytest.mark.parametrize("warm_ticks", [0, 400, 900])
def test_lockup_only_in_the_last_braking_zone(profile, warm_ticks):
    per, _, run = run_script(profile, "lockup", warm_ticks)
    locked = {s for s, d in per.items() if d["lockup"]}
    assert locked == {6}, per                      # "Braking zone 3 of 3: far too late"
    assert not any(d["wheelspin"] > 2 for d in per.values()), per
    assert run.sim.script_kappa is None           # back on the real circuit afterwards


@pytest.mark.parametrize("warm_ticks", [0, 400, 900])
def test_wheelspin_only_on_the_standing_start_exit(profile, warm_ticks):
    per, _, _ = run_script(profile, "wheelspin", warm_ticks)
    spun = {s for s, d in per.items() if d["wheelspin"]}
    assert spun == {8}, per                        # "Hairpin exit 3 of 3: full throttle from almost a standstill"
    assert not any(d["lockup"] for d in per.values()), per


def test_overheating_in_the_long_corner(profile):
    per, _, _ = run_script(profile, "overheat")
    assert per[3]["hot"] > 0, per                  # "Long, tight right-hander: tires pushed past the limit"
    assert per[1]["hot"] == 0 and per[2]["hot"] == 0, per


def test_puncture_leaks_one_tyre_after_the_debris(profile):
    per, sim, run = run_script(profile, "pressure")
    leaking = [w for w in WHEELS if sim.leak_rate[w] > 0]
    assert len(leaking) == 1
    same_axle = [w for w in WHEELS if w != leaking[0] and w[0] == leaking[0][0]][0]
    assert sim.psi(same_axle) - sim.psi(leaking[0]) > 3.0


# ---------------------------------------------------------------- the watcher's verdict
def frame(t, p=0.0, on=False):
    tyre = {"components": {"overheat": 0.0}, "flags": {"slow_puncture": False, "deflation": False}}
    return {"t": t, "events": {"lockup": {"p": p, "on": on}}, "tyres": {w: tyre for w in WHEELS}}


def watch_with(warn_at, hazard_t, t0=0.0, end=20.0):
    w = ScenarioWatch("lockup", t0)
    w.hazard_t, w.end_t = hazard_t, end
    for i in range(int(end * 4)):
        t = i / 4
        w.on_frame(frame(t, 0.5 if t in warn_at else 0.05))
    return w.result()


def test_lead_counts_only_the_warning_that_runs_into_the_hazard():
    early = {2.0, 2.25}                            # an unrelated warning long before
    into = {9.0, 9.25, 9.5, 9.75, 10.0}            # continuous up to the hazard at 10.0
    r = watch_with(early | into, hazard_t=10.0)
    assert r["lead_s"] == 1.0 and r["approach_warnings"] == 1
    assert r["text"].startswith("Lightning Response warned 1.0 s before the lock-up")


def test_detection_after_the_hazard_is_reported_as_such():
    r = watch_with({10.5, 10.75}, hazard_t=10.0)
    assert r["lead_s"] == -0.5 and "detected the lock-up 0.5 s after" in r["text"]


def test_no_warning_is_reported_honestly():
    r = watch_with(set(), hazard_t=10.0)
    assert not r["warned"] and "did not flag" in r["text"]


def test_watch_waits_for_the_analysis_to_catch_up():
    w = ScenarioWatch("lockup", 0.0)
    w.hazard_t = 10.0
    assert not w.ready(30.0)                       # script still running
    w.end_t = 12.0
    assert not w.ready(11.0)                       # analysis behind the end of the script
    assert not w.ready(13.0)                       # no warning since the hazard yet: keep watching
    w.on_frame(frame(10.25, 0.6))
    assert w.ready(13.0)
    assert ScenarioWatch("lockup", 0.0).ready(0) is False


# ---------------------------------------------------------------- live session, with the trained models
@pytest.fixture(scope="module")
def bundles():
    from sidewall.engine.monitor import load_bundles
    return load_bundles()


def test_live_session_scenario_takes_the_pedals_and_reports(bundles):
    from sidewall.sources.live import LiveSession, SIM_HZ

    async def main():
        msgs = []

        async def pub(ch, m):
            msgs.append((ch, m))

        live = LiveSession(bundles, pub)
        for i in range(200):
            await live._step_physics()
        live.claim()
        live.start_scenario("lockup")
        live.set_input(1.0, 0.0, seq=1)            # the phone's pedals are ignored while the script drives
        assert live.sim.inputs["throttle"] == 0.0
        await live._publish_state(live.rows[-1])
        state = [m for ch, m in msgs if ch == "driver" and m.get("type") == "state"][-1]
        assert state["mode"] == "driver" and state["scenario"]["key"] == "lockup"
        i = 0
        while (live.scenario or live.watch) and i < 120 * SIM_HZ:
            await live._step_physics()
            i += 1
            if i % 10 == 0:
                await live._analyse()
        assert live.scenario is None and live.driver_active and not live.sim.autopilot   # phone keeps the wheel
        results = [m for ch, m in msgs if ch == "driver" and m.get("type") == "scenario_result"]
        assert len(results) == 1
        r = results[0]
        assert r["happened"] and r["warned"] and r["lead_s"] is not None and abs(r["lead_s"]) <= 1.0, r

    asyncio.run(main())


def test_claiming_the_wheel_cancels_a_scenario(bundles):
    from sidewall.sources.live import LiveSession

    async def pub(ch, m):
        pass

    live = LiveSession(bundles, pub)
    live.claim()
    live.start_scenario("overheat")
    assert live.sim.script_kappa is not None
    live.claim()                                   # phone reconnects / takes the wheel
    assert live.scenario is None and live.watch is None and live.sim.script_kappa is None
