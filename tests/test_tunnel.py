"""The tunnel watchdog: when to check the public address, when to replace the tunnel, and the public check itself."""
import pytest

from sidewall.server import tunnel as T


def st(**kw):
    s = {"started_at": 0.0, "url_at": 0.0, "last_check": -1e9}          # -1e9: never checked yet
    s.update(kw)
    return s


def test_a_stopped_cloudflared_is_restarted():
    assert T.decide(100.0, st(), proc_alive=False, url="https://a.trycloudflare.com") == "restart"


def test_no_address_yet_waits_then_restarts():
    assert T.decide(10.0, st(started_at=0.0), True, None) is None
    assert T.decide(T.NO_URL_TIMEOUT_S + 1, st(started_at=0.0), True, None) == "restart"


def test_a_fresh_address_gets_a_grace_period_then_regular_checks():
    url = "https://a.trycloudflare.com"
    assert T.decide(5.0, st(url_at=0.0), True, url) is None                          # DNS still spreading
    assert T.decide(T.GRACE_S + 1, st(url_at=0.0), True, url) == "check"
    later = T.GRACE_S + 5
    assert T.decide(later, st(url_at=0.0, last_check=later - 1), True, url) is None   # checked a second ago
    assert T.decide(later, st(url_at=0.0, last_check=later - T.CHECK_EVERY_S), True, url) == "check"


def test_a_name_missing_from_dns_fails_the_check(monkeypatch):
    monkeypatch.setattr(T, "resolve", lambda host, timeout=5.0: [])
    assert T.check_public("https://gone.trycloudflare.com") is False


def test_a_dns_error_fails_the_check(monkeypatch):
    def boom(host, timeout=5.0):
        raise OSError("offline")
    monkeypatch.setattr(T, "resolve", boom)
    assert T.check_public("https://x.trycloudflare.com") is False


def test_two_failed_checks_replace_the_tunnel(monkeypatch):
    """Drive the watchdog loop by hand: two failures in a row trigger one restart with a new address."""
    url = "https://old.trycloudflare.com"
    calls = {"restart": 0}

    class Alive:
        def poll(self):
            return None

    monkeypatch.setattr(T, "_proc", Alive())
    monkeypatch.setattr(T, "_url", url)
    monkeypatch.setattr(T, "check_public", lambda u, timeout=8.0: False)

    def fake_restart(reason):
        calls["restart"] += 1
        T._url = "https://new.trycloudflare.com"
        raise SystemExit                                           # stop the loop after the restart

    monkeypatch.setattr(T, "_restart", fake_restart)
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    clock = iter(range(1000, 100000, 40))                           # every tick 40 s later: always due a check
    monkeypatch.setattr(T.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(T, "state", {"started_at": 0.0, "url_at": 0.0, "last_check": 0.0, "fails": 0,
                                     "restarts": 0, "streak": 0, "last_ok": None, "reason": ""})
    monkeypatch.setattr(T, "_shutting_down", False)
    with pytest.raises(SystemExit):
        T._watchdog()
    assert calls["restart"] == 1 and T.state["fails"] == T.FAILS_TO_RESTART
