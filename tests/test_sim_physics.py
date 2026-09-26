"""The simulator's tyre physics must respond to the situation in the right direction and magnitude."""
import pytest

from sidewall.sources.sim import track_profile
from sidewall.sources.sim_check import drive, summary


@pytest.fixture(scope="module")
def profile():
    return track_profile()


@pytest.fixture(scope="module")
def warm(profile):
    """Three minutes of tidy driving from the blankets."""
    return drive(profile, "tidy", 180)


def test_tyres_warm_up_from_blankets_and_pressure_rises(warm):
    sim, d = warm
    assert d["surf_fl"].iloc[0] < 75                                  # starts at blanket temperature
    s = summary(d, 30)
    assert 85 < s.loc["fl", "surf_mean"] < 120                        # operating window after warm-up
    assert s.loc["fl", "psi"] > 23.5                                  # hot pressure above the 23.0 cold set-up


def test_safety_car_pace_cools_the_tread(profile):
    sim, fast = drive(profile, "tidy", 180)
    _, slow = drive(profile, "slow", 90, sim=sim)
    assert summary(slow, 30)["surf_mean"].mean() < summary(fast, 30)["surf_mean"].mean() - 8


def test_lockups_heat_the_fronts_and_wheelspin_heats_the_rears(profile):
    base_sim, _ = drive(profile, "tidy", 150)
    _, tidy = drive(profile, "tidy", 45, sim=base_sim)
    s_tidy = summary(tidy, 45)
    sim_b, _ = drive(profile, "tidy", 150)
    _, lb = drive(profile, "late_brake", 45, sim=sim_b)
    sim_f, _ = drive(profile, "tidy", 150)
    _, fl = drive(profile, "floor_it", 45, sim=sim_f)
    assert lb["lockup"].any() and fl["wheelspin"].any()
    assert summary(lb, 45).loc["fl", "surf_max"] > s_tidy.loc["fl", "surf_max"] + 10
    assert summary(fl, 45).loc["rl", "surf_mean"] > s_tidy.loc["rl", "surf_mean"] + 10
