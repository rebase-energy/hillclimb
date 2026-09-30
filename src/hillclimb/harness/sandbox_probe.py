"""What `hillclimb sandbox check` runs INSIDE the sandbox: it tries what a
hostile solution would try and reports, per attempt, whether it got through.

    python sandbox_probe.py '<json list of {id, kind, target}>'

prints one JSON object `{id: {"done": bool, "error": str | None, "seen": int | None}}`.
Stdlib only: it runs as a script under the engine's interpreter.
"""

from __future__ import annotations

import json
import os
import socket
import sys

TIMEOUT_S = 5


class Refused(Exception):
    """The proxy answered, and the answer was no."""


class NoRoute(Exception):
    """The proxy let it through and the host did not answer (no internet here)."""


def _through_proxy(target: str) -> None:
    address = os.environ["HTTPS_PROXY"].removeprefix("http://")
    host, _, port = address.rpartition(":")
    with socket.create_connection((host, int(port)), timeout=TIMEOUT_S) as sock:
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        head = b""
        while b"\r\n" not in head:
            chunk = sock.recv(1024)
            if not chunk:
                break
            head += chunk
    status = head.split(b"\r\n")[0].decode("latin-1")
    if " 200 " in status:
        return
    raise (NoRoute if " 502 " in status else Refused)(status)


def attempt(kind: str, target: str) -> int | None:
    if kind == "write":
        with open(target, "w") as sink:
            sink.write("written by hillclimb sandbox check\n")
    elif kind == "read":
        if os.path.isdir(target):
            return len(os.listdir(target))
        with open(target) as source:
            return len(source.read())
    elif kind == "connect":
        host, _, port = target.rpartition(":")
        socket.create_connection((host, int(port)), timeout=TIMEOUT_S).close()
    elif kind == "resolve":
        socket.getaddrinfo(target, 443)
    elif kind == "signal":
        os.kill(int(target), 0)
    elif kind == "proxy":
        _through_proxy(target)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return None


def main(argv: list[str]) -> int:
    report = {}
    for item in json.loads(argv[0]):
        try:
            seen = attempt(item["kind"], item["target"])
            report[item["id"]] = {"done": True, "error": None, "seen": seen}
        except Exception as exc:  # noqa: BLE001 — every refusal is an answer
            report[item["id"]] = {"done": False, "error": type(exc).__name__, "seen": None}
    json.dump(report, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
