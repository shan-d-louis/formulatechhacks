"""Optional Cloudflare quick tunnel: a public https:// address for this laptop's server, kept alive by a watchdog.

Phones then open the driver page from any network (mobile data, eduroam, a different Wi-Fi), because they no longer
need to reach the laptop over the local network. `python -m sidewall.server.app --tunnel` starts `cloudflared`, reads
the https://<random>.trycloudflare.com address it prints, and the QR codes use it. The address changes every start.

Quick tunnels can die while `cloudflared` keeps running (seen after the laptop changed networks: the address vanished
from DNS and the QR code pointed nowhere). So a watchdog loads /health through the public address every 30 s, looking
the name up with Cloudflare's own DNS-over-HTTPS (a campus resolver can cache "no such name" for a fresh tunnel). After
2 failures in a row, or if `cloudflared` exits, it starts a new tunnel. Meanwhile public_url() is None, so the QR code
falls back to the local network, and the pit wall picks up the new address by itself. Repeated restarts back off.
"""
import atexit
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CHECK_EVERY_S = 30.0       # how often the public address is tested
GRACE_S = 20.0             # a brand-new address is left alone this long (DNS is still spreading)
FAILS_TO_RESTART = 2       # consecutive failed checks before the tunnel is replaced
NO_URL_TIMEOUT_S = 60.0    # cloudflared started but never printed an address
MAX_BACKOFF_S = 300.0      # longest wait between restarts when they keep failing (e.g. no internet)
DOH = "https://cloudflare-dns.com/dns-query"

_lock = threading.Lock()
_url: str | None = None
_proc: subprocess.Popen | None = None
_exe: str | None = None
_port: int = 8000
_shutting_down = False
state = {"started_at": 0.0, "url_at": 0.0, "last_check": 0.0, "fails": 0, "restarts": 0, "streak": 0,
         "last_ok": None, "reason": ""}


def find_cloudflared() -> str | None:
    """On PATH, or where winget and the Windows installer put it (a fresh install is not on PATH until a new shell)."""
    found = shutil.which("cloudflared")
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    for p in (local / "Microsoft" / "WinGet" / "Links" / "cloudflared.exe",
              Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
              Path(r"C:\Program Files\cloudflared\cloudflared.exe")):
        if p.exists():
            return str(p)
    for p in (local / "Microsoft" / "WinGet" / "Packages").glob("Cloudflare.cloudflared*/cloudflared*.exe"):
        return str(p)
    return None


def public_url() -> str | None:
    return _url


def status() -> dict:
    """For /api/lan: whether a tunnel is up and how the watchdog is doing."""
    return {"tunnel": _url is not None, "restarts": state["restarts"], "last_ok": state["last_ok"]}


# ------------------------------------------------------------------ checking the public address
def resolve(host: str, timeout: float = 5.0) -> list[str]:
    """IPv4 addresses for host from Cloudflare DNS-over-HTTPS (skips the local resolver's negative cache)."""
    q = urllib.parse.urlencode({"name": host, "type": "A"})
    req = urllib.request.Request(f"{DOH}?{q}", headers={"accept": "application/dns-json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]


def check_public(url: str, timeout: float = 8.0) -> bool:
    """True when GET <url>/health answers 200 through the tunnel. Connects to an address from resolve() while
    still presenting the tunnel's hostname (SNI and Host), exactly as a phone would."""
    host = urllib.parse.urlparse(url).hostname
    try:
        ips = resolve(host)
    except Exception:
        return False
    ctx = ssl.create_default_context()
    for ip in ips[:2]:
        try:
            sock = socket.create_connection((ip, 443), timeout=timeout)
            conn = http.client.HTTPSConnection(host, 443, timeout=timeout, context=ctx)
            conn.sock = ctx.wrap_socket(sock, server_hostname=host)
            conn.request("GET", "/health", headers={"Host": host, "User-Agent": "lightning-response-tunnel-check"})
            ok = conn.getresponse().status == 200
            conn.close()
            if ok:
                return True
        except Exception:
            continue
    return False


def decide(now: float, st: dict, proc_alive: bool, url: str | None) -> str | None:
    """What the watchdog should do now: 'restart', 'check' or None. Pure, so it can be tested."""
    if not proc_alive:
        return "restart"
    if url is None:
        return "restart" if now - st["started_at"] > NO_URL_TIMEOUT_S else None
    if now - st["url_at"] < GRACE_S or now - st["last_check"] < CHECK_EVERY_S:
        return None
    return "check"


# ------------------------------------------------------------------ process management
def start(port: int) -> bool:
    """Start the tunnel and its watchdog; public_url() is set once cloudflared reports the address (a few seconds)."""
    global _exe, _port
    _exe = find_cloudflared()
    if not _exe:
        print("Tunnel: cloudflared is not installed (winget install --id Cloudflare.cloudflared). Phones will use the local network.")
        return False
    _port = port
    _spawn()
    atexit.register(shutdown)
    threading.Thread(target=_watchdog, daemon=True).start()
    return True


def _spawn():
    global _proc, _url
    with _lock:
        _url = None
        _proc = subprocess.Popen([_exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{_port}"],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        state.update(started_at=time.monotonic(), url_at=0.0, last_check=0.0, fails=0)
        proc = _proc
    threading.Thread(target=_read, args=(proc,), daemon=True).start()


def _read(proc: subprocess.Popen):
    global _url
    for line in proc.stdout:                     # cloudflared logs the address once the tunnel is up
        m = URL_RE.search(line)
        if m and proc is _proc and not _url:
            with _lock:
                _url = m.group(0)
                state["url_at"] = time.monotonic()
            print(f"Tunnel ready: phones can open {_url}")
    if proc is _proc:                            # this tunnel's process ended: fall back to the local network
        _url = None


def _restart(reason: str):
    state["streak"] += 1
    wait = min(MAX_BACKOFF_S, 0.0 if state["streak"] == 1 else 5.0 * 2 ** (state["streak"] - 1))
    print(f"Tunnel: {reason}; starting a new one{f' in {wait:.0f} s' if wait else ''}. "
          "Phones use the local network until then.")
    stop()
    global _url
    _url = None
    state["reason"] = reason
    if wait:
        time.sleep(wait)
    if _shutting_down:
        return
    state["restarts"] += 1
    _spawn()


def _watchdog():
    while not _shutting_down:
        time.sleep(2.0)
        if _shutting_down:
            return
        try:
            proc = _proc
            action = decide(time.monotonic(), state, proc is not None and proc.poll() is None, _url)
            if action == "restart":
                _restart("cloudflared stopped" if proc is None or proc.poll() is not None else "no address from cloudflared")
            elif action == "check":
                url = _url
                ok = check_public(url)
                state["last_check"] = time.monotonic()
                if ok:
                    state["fails"], state["streak"], state["last_ok"] = 0, 0, time.time()
                else:
                    state["fails"] += 1
                    print(f"Tunnel: {url} did not answer ({state['fails']}/{FAILS_TO_RESTART}).")
                    if state["fails"] >= FAILS_TO_RESTART and url == _url:
                        _restart("the public address stopped working")
        except Exception as e:                   # the watchdog must never die
            print(f"Tunnel watchdog error: {e}")


def shutdown():
    """At exit: stop the tunnel for good (the watchdog must not start a new one)."""
    global _shutting_down
    _shutting_down = True
    stop()


def stop():
    proc = _proc
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
