"""Driver alerts from the next-second risk: warn = counteract, brace = imminent. Pushed to the phone and pit wall."""
import asyncio

import pytest

from sidewall.engine.monitor import load_bundles
from sidewall.sources.live import ALERT_HOLD_S, ALERT_P, LiveSession, SIM_HZ


@pytest.fixture(scope="module")
def bundles():
    return load_bundles()


async def _nop(ch, m):
    pass


def test_levels_rise_at_once_and_hold_before_falling(bundles):
    live = LiveSession(bundles, _nop)
    warn, brace = ALERT_P["lockup"]
    calm = {"lockup": (0.0, False), "wheelspin": (0.0, False)}
    assert live._update_alerts({**calm, "lockup": (warn, False)}) and live.alerts["lockup"]["level"] == 1
    assert live._update_alerts({**calm, "lockup": (brace, False)}) and live.alerts["lockup"]["level"] == 2
    live.sim.t += ALERT_HOLD_S / 2
    live._update_alerts(calm)
    assert live.alerts["lockup"]["level"] == 2                    # still inside the hold
    live.sim.t += ALERT_HOLD_S
    assert live._update_alerts(calm) and live.alerts["lockup"]["level"] == 0
    live._update_alerts({**calm, "wheelspin": (0.1, True)})       # already happening = brace
    assert live.alerts["wheelspin"]["level"] == 2
    msg = live._alert_msg()
    assert msg["alerts"]["wheelspin"]["text"].endswith("brace") and msg["alerts"]["lockup"]["text"] == ""


def test_no_alert_on_a_crashed_car(bundles):
    live = LiveSession(bundles, _nop)
    live.sim.crashed = {"kind": "off", "t": 0.0}
    live._update_alerts({"lockup": (0.99, True), "wheelspin": (0.99, True)})
    assert all(a["level"] == 0 for a in live.alerts.values())


def test_lockup_scenario_warns_the_driver_then_says_brace(bundles):
    async def main():
        msgs = []

        async def pub(ch, m):
            msgs.append((ch, m))

        live = LiveSession(bundles, pub)
        for i in range(20 * SIM_HZ):
            await live._step_physics()
            if i % 10 == 9:
                await live._analyse()
        msgs.clear()
        live.claim()
        live.start_scenario("lockup")
        i = 0
        while live.scenario and i < 90 * SIM_HZ:
            await live._step_physics()
            i += 1
            if i % 10 == 0:
                await live._analyse()
        levels = [m["alerts"]["lockup"]["level"] for ch, m in msgs if ch == "driver" and m["type"] == "driver_alert"]
        assert 1 in levels and 2 in levels                          # a warning on the approach, then brace
        assert levels.index(1) < levels.index(2)
        assert any(ch == "pitwall" and m["type"] == "driver_alert" for ch, m in msgs)

    asyncio.run(main())
