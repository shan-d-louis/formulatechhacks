"""Optional Cloudflare quick tunnel: a public https:// address for this laptop's server.

Phones then open the driver page from any network (mobile data, eduroam, a different Wi-Fi), because they no longer
need to reach the laptop over the local network. `python -m sidewall.server.app --tunnel` starts `cloudflared`, reads
the https://<random>.trycloudflare.com address it prints, and the QR codes use it. The address changes every start.
"""
import atexit
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
_url: str | None = None
_proc: subprocess.Popen | None = None


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


def start(port: int) -> bool:
    """Start the tunnel in the background; public_url() is set once cloudflared reports it (a few seconds)."""
    global _proc
    exe = find_cloudflared()
    if not exe:
        print("Tunnel: cloudflared is not installed (winget install --id Cloudflare.cloudflared). Phones will use the local network.")
        return False
    _proc = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    atexit.register(stop)
    threading.Thread(target=_watch, args=(_proc,), daemon=True).start()
    return True


def _watch(proc: subprocess.Popen):
    global _url
    for line in proc.stdout:                     # cloudflared logs the address once the tunnel is up
        m = URL_RE.search(line)
        if m and not _url:
            _url = m.group(0)
            print(f"Tunnel ready: phones can open {_url}")
    _url = None                                  # cloudflared exited: fall back to the local network
    print("Tunnel closed: phones will use the local network.")


def stop():
    if _proc and _proc.poll() is None:
        _proc.terminate()
