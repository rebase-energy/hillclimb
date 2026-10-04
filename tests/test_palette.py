"""The plain-CLI palette: two hex palettes and the background that picks one."""

from __future__ import annotations

import os
import pty
import threading

import pytest

from hillclimb.tui import palette
from hillclimb.tui.palette import DARK, LIGHT, colorfgbg, is_light, luminance, parse_osc11, query_background

BLACK, WHITE = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)


def _rgb(hex_color: str) -> tuple[float, float, float]:
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _contrast(a, b) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("colors, page", [(DARK, BLACK), (LIGHT, WHITE)])
def test_every_colour_reads_as_text_on_its_background(colors, page):
    for name, hex_color in colors.items():
        assert _contrast(_rgb(hex_color), page) >= 4.5, name


def test_light_cyan_is_the_websites_light_accent():
    assert LIGHT["cyan"].lower() == "#08737a"


def test_parse_osc11_takes_any_channel_width():
    assert parse_osc11(b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\") == WHITE
    assert parse_osc11(b"\x1b]11;rgb:00/00/00\x07") == BLACK
    assert parse_osc11(b"\x1b[?62;c") is None


def test_is_light_splits_page_colours():
    assert is_light(_rgb("#fbfcfc")) and is_light(_rgb("#fdf6e3"))
    assert not is_light(_rgb("#0b0d0e")) and not is_light(_rgb("#282c34"))


def test_colorfgbg_reads_the_last_field():
    assert colorfgbg("0;15") == "light"
    assert colorfgbg("0;default;7") == "light"
    assert colorfgbg("15;0") == "dark"
    assert colorfgbg("") is None and colorfgbg("15;default") is None


def _terminal(master: int, reply: bytes) -> threading.Thread:
    """Play the terminal on a pty's master side: wait for the query, answer."""
    def serve():
        seen = b""
        while b"\x1b[c" not in seen:
            seen += os.read(master, 256)
        os.write(master, reply)

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def test_query_background_reads_the_terminals_answer():
    master, slave = pty.openpty()
    try:
        _terminal(master, b"\x1b]11;rgb:fbfb/fcfc/fcfc\x1b\\\x1b[?62;c")
        rgb = query_background(slave, timeout_s=5)
        assert rgb is not None and is_light(rgb)
    finally:
        os.close(master), os.close(slave)


def test_query_background_stops_at_da1_without_osc11():
    master, slave = pty.openpty()
    try:
        _terminal(master, b"\x1b[?62;c")
        assert query_background(slave, timeout_s=5) is None  # returns on DA1, not the timeout
    finally:
        os.close(master), os.close(slave)


def test_query_background_is_none_off_a_terminal(tmp_path):
    with open(tmp_path / "f", "w") as handle:
        assert query_background(handle.fileno()) is None


def test_background_env_override_wins(monkeypatch):
    monkeypatch.setenv("COLORFGBG", "0;15")
    assert palette.background() == "light"
    monkeypatch.setenv("HILLCLIMB_BACKGROUND", "dark")
    assert palette.background() == "dark"
    monkeypatch.delenv("COLORFGBG")
    monkeypatch.delenv("HILLCLIMB_BACKGROUND")
    assert palette.background() == "dark"
