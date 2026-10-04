"""The plain-CLI palette: the website's colours as hex, for everything rich
prints outside a Textual app (the CLI's voice, `--help`, the banner,
`hillclimb ps`).

A named ANSI colour ("cyan", "green") is drawn in whatever the terminal's
own palette says, so the same output is vivid in iTerm2 and pastel in a
stock Ghostty. Hex is the same everywhere; rich downgrades it to the nearest
256- or 16-colour entry on a terminal without truecolor.

Hex does not follow the terminal's theme, so there are two palettes and the
terminal's background picks one: `DARK` is theme.py's (`primary`, `success`,
`warning`, `error`) and graphview's vivid magenta; `LIGHT` is the same hues
darkened until they read as text on white, its cyan the website's light
`--cyan` (the README's light wordmark). The background is asked of the
terminal once, at import (`background()`); HILLCLIMB_BACKGROUND=light|dark
overrides the answer.

This module stays import-light so the CLI does not pay for Textual.
"""

from __future__ import annotations

import os
import re
import sys

DARK = {
    "cyan": "#2EE6E6", "green": "#22C55E", "yellow": "#EAB308",
    "red": "#EF4444", "magenta": "#E0218A",
}
LIGHT = {
    "cyan": "#08737A", "green": "#15803D", "yellow": "#A16207",
    "red": "#B91C1C", "magenta": "#BE185D",
}

# OSC 11 asks for the background colour; DA1 (`CSI c`) is the sentinel every
# terminal answers, so one that ignores OSC 11 is known by its DA1 reply
# arriving alone instead of by waiting out the timeout.
_QUERY = b"\x1b]11;?\x1b\\\x1b[c"
_OSC11_REPLY = re.compile(rb"\x1b\]11;rgba?:([0-9a-fA-F]+)/([0-9a-fA-F]+)/([0-9a-fA-F]+)")
_DA1_REPLY = re.compile(rb"\x1b\[\?[0-9;]*c")
QUERY_TIMEOUT_S = 0.5


def is_light(rgb: tuple[float, float, float]) -> bool:
    """Whether a background (channels 0..1) is light: its relative luminance
    is nearer white's than black's on the contrast scale."""
    return luminance(rgb) > 0.179  # where contrast with black and with white are equal


def luminance(rgb: tuple[float, float, float]) -> float:
    """WCAG relative luminance of an sRGB colour with channels 0..1."""
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def parse_osc11(reply: bytes) -> tuple[float, float, float] | None:
    """The colour in a terminal's OSC 11 reply (`rgb:RRRR/GGGG/BBBB`, one to
    four hex digits a channel) as channels 0..1, or None without one."""
    match = _OSC11_REPLY.search(reply)
    if match is None:
        return None
    return tuple(int(part, 16) / (16 ** len(part) - 1) for part in match.groups())


def query_background(fd: int, timeout_s: float = QUERY_TIMEOUT_S) -> tuple[float, float, float] | None:
    """Ask the terminal on `fd` for its background colour. None when it does
    not say (no OSC 11, no reply in time, not a terminal, no termios)."""
    try:
        import select
        import termios
        import time
        import tty

        saved = termios.tcgetattr(fd)
    except Exception:  # not a terminal, or not a platform with termios
        return None
    try:
        tty.setcbreak(fd)
        os.write(fd, _QUERY)
        reply = b""
        deadline = time.monotonic() + timeout_s
        while not _DA1_REPLY.search(reply):
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([fd], [], [], left)[0]:
                break
            reply += os.read(fd, 256)
        return parse_osc11(reply)
    except OSError:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSANOW, saved)


def colorfgbg(value: str) -> str | None:
    """`light` or `dark` from a COLORFGBG value ("15;0", "0;default;15"): its
    last field is the background's ANSI index, 7 and 15 being the whites."""
    last = value.rsplit(";", 1)[-1]
    if not last.isdigit():
        return None
    return "light" if int(last) in (7, 15) else "dark"


def background() -> str:
    """`light` or `dark`: HILLCLIMB_BACKGROUND if set, else what the terminal
    answers, else COLORFGBG, else dark. The terminal is only asked when
    hillclimb is talking to one that would be coloured."""
    forced = os.environ.get("HILLCLIMB_BACKGROUND", "").strip().lower()
    if forced in ("light", "dark"):
        return forced
    if "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb":
        try:
            interactive = sys.stdin.isatty() and sys.stdout.isatty()
        except (AttributeError, ValueError):  # a replaced or closed stream
            interactive = False
        if interactive:
            rgb = query_background(sys.stdin.fileno())
            if rgb is not None:
                return "light" if is_light(rgb) else "dark"
    return colorfgbg(os.environ.get("COLORFGBG", "")) or "dark"


BACKGROUND = background()
ANSI_TO_HEX = LIGHT if BACKGROUND == "light" else DARK

CYAN = ANSI_TO_HEX["cyan"]
GREEN = ANSI_TO_HEX["green"]
YELLOW = ANSI_TO_HEX["yellow"]
RED = ANSI_TO_HEX["red"]
MAGENTA = ANSI_TO_HEX["magenta"]


def pinned(style: str) -> str:
    """A rich style string with its named ANSI colours swapped for the
    palette's hex: "bold cyan" -> "bold #2EE6E6"."""
    return " ".join(ANSI_TO_HEX.get(word, word) for word in style.split())
