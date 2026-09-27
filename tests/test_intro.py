"""The intro's grabbable hold: the SGR mouse reader, the drag camera and a
frame rendered from an explicit camera (see `tui/intro.py`)."""

import re

from hillclimb.tui import intro


def _plain(frame: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", frame)


def test_sgr_reports_parse_to_press_drag_release_and_wheel():
    inp = intro._Input(fd=-1)
    inp.buffer += b"\x1b[<0;40;12M" + b"\x1b[<32;45;13M" + b"\x1b[<0;45;13m" + b"\x1b[<64;5;5M" + b"\x1b[<65;5;5M"
    assert inp.drain() == [
        ("press", 0, 40, 12),
        ("drag", 0, 45, 13),
        ("release", 0, 45, 13),
        ("wheel", 0, 5, 5),
        ("wheel", 1, 5, 5),
    ]
    assert not inp.buffer


def test_a_report_split_across_reads_waits_for_its_end():
    inp = intro._Input(fd=-1)
    inp.buffer += b"\x1b[<32;4"
    assert inp.drain() == []
    inp.buffer += b"5;13M\r"
    assert inp.drain() == [("drag", 0, 45, 13), "enter"]


def test_other_keys_are_swallowed_and_enter_is_enter():
    inp = intro._Input(fd=-1)
    inp.buffer += b"qx\n"
    assert inp.drain() == ["enter"]
    inp.buffer += b"\x1b[<garbage that never ends" + b"x" * 60
    assert inp.drain() == []
    assert not inp.buffer  # the junk is dropped, not kept forever


def test_drag_turns_with_the_cursor_and_tilt_stays_in_range():
    start = (40, 10, intro.END_YAW, 1.0)
    yaw, tilt = intro._drag_camera(start, 60, 10, width=80)
    assert yaw > intro.END_YAW  # dragging right turns the near face right
    assert tilt == 1.0
    yaw, tilt = intro._drag_camera(start, 40, 30, width=80)
    assert yaw == intro.END_YAW
    assert tilt == intro.TILT_RANGE[1]  # dragging far down tips the massif over: top view
    _, tilt = intro._drag_camera(start, 40, -30, width=80)
    assert tilt == intro.TILT_RANGE[0]  # dragging up lowers the eye to the floor
    assert intro._drag_camera(start, 40, 10, width=80) == (intro.END_YAW, 1.0)


def test_a_frame_from_an_explicit_camera_keeps_the_summit_star():
    samples = intro._surface_samples()
    paths = tuple(intro._ascent_path(x, y) for x, y in intro.CLIMBER_STARTS)
    end = intro._render(samples, paths, intro.TOTAL_FRAMES, 90, 30)
    same = intro._render(samples, paths, intro.TOTAL_FRAMES, 90, 30, yaw=intro.END_YAW, tilt=1.0)
    turned = intro._render(samples, paths, intro.TOTAL_FRAMES, 90, 30, yaw=intro.END_YAW + 1.0, tilt=0.6)
    assert same == end  # the hold starts exactly where the sweep ended
    assert turned != end
    assert "★" in _plain(turned) and "@" in _plain(turned)
