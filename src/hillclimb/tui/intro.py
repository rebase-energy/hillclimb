"""First-run intro: a rotating 3D hill, climbed live in the terminal.

Pure Python, stdlib only — the classic terminal-donut trick pointed at this
tool's own subject. A height field of Gaussian peaks is sampled once; every
frame the samples are yaw-rotated, projected onto the character grid through
a depth buffer, and shaded by how much light each patch of surface catches
(character ramp) and by altitude (cyan ramp, the CLI's palette). Two greedy
climbers walk their gradient-ascent paths as it turns: one reaches the
summit and the star, one tops out on a local optimum and stays there —
hillclimb's pitch, drawn instead of written. The logotype — the CLI
banner, downscaled into quadrant-glyph pixels — sweeps in last, a blank
band above the summit.

It plays once, the first time hillclimb runs in an interactive terminal,
then never again (marker file next to the user config). `hillclimb intro`
replays it; `hillclimb intro --reset` re-arms the first run; `hillclimb
--skip-intro` retires the auto-play without watching it. A non-TTY, CI,
HILLCLIMB_NO_INTRO, a dumb or tiny terminal, or shell completion all skip
it, Ctrl-C at any point skips the whole intro without killing the command
behind it, and Enter during the sweep jumps to its end screen (a second
Enter leaves it). Once the sweep has ended the mountain can be grabbed:
the hold turns on xterm mouse reporting and a drag (or the wheel) turns
and tilts the camera — a raw ANSI loop like the rest, no widget toolkit
behind it.

The palette is the product's: every altitude shade is derived from
theme.py's `--cyan`, the summit star is the theme's warning gold, the
summit climber the theme foreground.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import time

from hillclimb.project import user_config_path

# The terrain: (cx, cy, height, spread) per Gaussian. One dominant summit,
# one tempting local optimum for the second climber to get stuck on, and a
# low shoulder that keeps the ridge line from reading as two lone cones.
PEAKS = (
    (0.15, -0.10, 0.95, 0.35),
    (-0.75, 0.55, 0.48, 0.16),
    (0.55, 0.85, 0.25, 0.12),
)
GRID_EXTENT = 1.25
GRID_STEP = 0.024

# Where the two searches start: chosen so plain gradient ascent carries the
# first to the main summit and strands the second on the (-0.75, 0.55) bump —
# and, measured over every frame of the rotation sweep, so both climbers stay
# on the camera side of the ridge the whole way up.
CLIMBER_STARTS = ((-0.05, 1.15), (-1.00, 0.95))

CHARS = " .:-=+*#%@"  # sparse -> dense, indexed by lighting

# The mountain wears the product's palette. THEME_CYAN is theme.py's
# `--cyan` (#2EE6E6) mirrored as a literal because the intro is stdlib-only
# and must not import the textual-backed theme module — keep them in sync.
THEME_CYAN = (46, 230, 230)  # theme.py primary / the site's --cyan
THEME_FG = (232, 234, 236)  # theme.py foreground #E8EAEC — the summit climber
THEME_GOLD = (234, 179, 8)  # theme.py warning #EAB308 — the summit star


def _shade(t: float) -> tuple[int, int, int]:
    """Altitude ramp derived from the theme cyan: a near-black valley floor
    rising to the full accent, the top quarter blending on into a white
    snowcap — the single strongest "this is a mountain" cue a character
    grid can give."""
    r, g, b = THEME_CYAN
    if t < 0.75:
        k = 0.12 + 0.88 * t / 0.75
        return int(r * k), int(g * k), int(b * k)
    k = (t - 0.75) / 0.25
    return int(r + (255 - r) * k), int(g + (255 - g) * k), int(b + (255 - b) * k)


PALETTE = tuple(_shade(i / 7) for i in range(8))
TRAIL_COLORS = ((188, 192, 196), (120, 126, 131))
CLIMBER_COLORS = (THEME_FG, (150, 156, 162))
FLAG_COLOR = THEME_GOLD
WORDMARK = "h i l l c l i m b"  # fallback when the terminal is too small for the logotype
WORDMARK_COLOR = (127, 243, 243)  # theme.py accent #7FF3F3 — the logotype's pale cyan

# Light mostly from the side, not from above: flat ground lands mid-ramp,
# slopes facing the light go dense, slopes facing away go dark.
LIGHT = (-0.55, -0.60, 0.58)

TOTAL_FRAMES = 200
FRAME_S = 0.04
HOLD_S = 1.5  # non-interactive fallback hold; a TTY waits for Enter instead
CONTINUE_PROMPT = "Drag to look around · enter to continue"
CONTINUE_COLOR = (138, 144, 150)
# The camera at the end of the sweep (see `_render`), and the drag's reach:
# one full picture width of drag is DRAG_SWEEP radians of yaw; one row of
# drag is DRAG_TILT of the tilt factor, which scales the front-view
# elevation (1.0 = the sweep's own tilt) and is clamped so the peaks' back
# slopes never rise above the ground plane and hollow the massif.
END_YAW = 0.63
DRAG_SWEEP = 3.0
DRAG_TILT = 0.05
TILT_RANGE = (0.2, 1.4)
# xterm mouse reporting for the hold after the sweep: button-event tracking
# (press, release, motion while held) in SGR encoding (`\x1b[<b;x;yM|m`, no
# coordinate limit). Off again before the alternate screen closes.
MOUSE_ON = "\x1b[?1002h\x1b[?1006h"
MOUSE_OFF = "\x1b[?1002l\x1b[?1006l"
# Per-climber climb duration, as a fraction of the animation. The summit
# climber finishes early on purpose: its north-face route swings behind the
# ridge partway through the long rotation sweep, and by then it must already
# be sitting on the peak — the one spot that is visible from every angle.
CLIMB_PORTIONS = (0.40, 0.68)


def _xterm256(rgb: tuple[int, int, int]) -> int:
    """Nearest xterm-256 index, for terminals that don't advertise truecolor.
    Near-neutral colors snap to the gray ramp; the rest to the 6x6x6 cube."""
    r, g, b = rgb
    if abs(r - g) < 12 and abs(g - b) < 12 and abs(r - b) < 12:
        v = (r + g + b) // 3
        if v < 8:
            return 16
        if v > 238:
            return 231
        return 232 + (v - 8) // 10
    step = lambda v: 0 if v < 48 else 1 if v < 115 else (v - 35) // 40  # noqa: E731
    return 16 + 36 * step(r) + 6 * step(g) + step(b)


def _sgr_table() -> dict:
    """Color -> escape-sequence cache for one process. Truecolor when the
    terminal advertises it (COLORTERM), the closest 256-color index
    otherwise, so the theme cyan survives older terminals too."""
    truecolor = os.environ.get("COLORTERM", "") in ("truecolor", "24bit")
    table: dict = {}

    class _Table(dict):
        def __missing__(self, rgb):
            seq = (
                "\x1b[38;2;%d;%d;%dm" % rgb
                if truecolor
                else "\x1b[38;5;%dm" % _xterm256(rgb)
            )
            self[rgb] = seq
            return seq

    return _Table(table)


_LOGO_CACHE: list[str] | None = None


def _mini_logo() -> list[str]:
    """The CLI banner (tui.banner.BANNER_LINES) downscaled 2x2 into quadrant
    glyphs: every solid block cell is a pixel, the thin box-drawing shadow
    art is dropped — the wordmark and its rising-arrow mark land as a
    4-row pixel logotype."""
    global _LOGO_CACHE
    if _LOGO_CACHE is None:
        from hillclimb.tui.banner import BANNER_LINES

        quads = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"
        grid = [[ch == "█" for ch in line] for line in BANNER_LINES]
        w = max(len(row) for row in grid)
        for row in grid:
            row.extend([False] * (w - len(row)))
        if len(grid) % 2:
            grid.append([False] * w)
        if w % 2:
            for row in grid:
                row.append(False)
        lines = []
        for y in range(0, len(grid), 2):
            top, bottom = grid[y], grid[y + 1]
            lines.append(
                "".join(
                    quads[top[x] + 2 * top[x + 1] + 4 * bottom[x] + 8 * bottom[x + 1]]
                    for x in range(0, w, 2)
                ).rstrip()
            )
        while lines and not lines[-1]:
            lines.pop()
        _LOGO_CACHE = lines
    return _LOGO_CACHE


def _height_grad(x: float, y: float) -> tuple[float, float, float]:
    z = gx = gy = 0.0
    for cx, cy, amp, spread in PEAKS:
        e = amp * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / spread)
        z += e
        gx -= 2 * (x - cx) / spread * e
        gy -= 2 * (y - cy) / spread * e
    return z, gx, gy


def _surface_samples() -> list[tuple[float, float, float, float, float, float]]:
    """(x, y, height, unit surface normal) for every grid point — rotation
    and projection are per-frame, the terrain itself never changes."""
    samples = []
    n = int(2 * GRID_EXTENT / GRID_STEP) + 1
    for i in range(n):
        x = -GRID_EXTENT + i * GRID_STEP
        for j in range(n):
            y = -GRID_EXTENT + j * GRID_STEP
            z, gx, gy = _height_grad(x, y)
            inv = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
            samples.append((x, y, z, -gx * inv, -gy * inv, inv))
    return samples


def _ascent_path(x: float, y: float, step: float = 0.035) -> list[tuple[float, float, float]]:
    """Greedy gradient ascent — the search strategy, drawn on the hill it runs on.
    Stops when a step no longer gains height, i.e. at whichever peak is nearest."""
    path = [(x, y, _height_grad(x, y)[0])]
    for _ in range(300):
        z, gx, gy = _height_grad(x, y)
        norm = math.sqrt(gx * gx + gy * gy)
        if norm < 1e-4:
            break
        x += step * gx / norm
        y += step * gy / norm
        z_next = _height_grad(x, y)[0]
        if z_next <= z + 1e-5:
            break
        path.append((x, y, z_next))
    return path


def _render(
    samples: list[tuple[float, float, float, float, float, float]],
    paths: tuple[list[tuple[float, float, float]], ...],
    frame: int,
    width: int,
    height: int,
    margin_x: int = 0,
    margin_y: int = 0,
    *,
    yaw: float | None = None,
    tilt: float = 1.0,
) -> str:
    """One frame. The sweep derives the camera from `frame`; the hold after
    it passes `yaw`/`tilt` from the mouse drag instead, with `frame` at its
    last value so the climbers stay on their summits."""
    # Sweep from 3.65 down to 0.63 rad: along this whole arc the local-optimum
    # bump stays on the camera side of the massif, so both summits are in view
    # from the first frame instead of the small one appearing mid-rotation.
    # The end angle is where the two summits project to equal depth — the
    # line connecting them lies exactly perpendicular to the line of sight,
    # so the closing shot reads summit and local optimum side by side.
    if yaw is None:
        yaw = 3.65 - (3.65 - END_YAW) * frame / TOTAL_FRAMES
    cos_b, sin_b = math.cos(yaw), math.sin(yaw)
    kx = width / (2 * GRID_EXTENT) * 0.88  # small side margin: the climbers'
    # starting corners must stay on screen at every rotation angle
    ky = height / (2 * GRID_EXTENT) * 0.42 * tilt  # gentle tilt: an elevated front view
    kz = height * 0.48  # exaggerated relief, the way trail maps draw it
    logo = _mini_logo() if height >= 24 else None
    logo_w = max(map(len, logo)) if logo else 0
    if logo and logo_w + 2 > width:
        logo, logo_w = None, 0
    # With the logotype pinned to the top rows the whole scene drops a few
    # rows, so a blank band separates the logo from the summit star — the
    # star is the topmost thing the mountain ever draws.
    cx, cy = width / 2, height * 0.62 + (4 if logo else 0)
    lx, ly, lz = LIGHT
    ramp = len(CHARS) - 1

    chars = [" "] * (width * height)
    colors = [0] * (width * height)
    depth = [1e9] * (width * height)

    def project(x: float, y: float, z: float) -> tuple[int, int, float]:
        # Screen y falls with BOTH height and distance (-yr*ky - z*kz): a
        # camera in front of and above the terrain, far ground receding
        # upward toward the horizon. The sign of the yr term is what keeps
        # the massif watertight — with +yr*ky the viewpoint is effectively
        # under the ground plane, the back slopes splay out BELOW the peaks
        # where nothing covers them, and the mountain reads as a hollow
        # shell seen from inside. With -yr*ky every back-slope cell projects
        # into rows the near face also fills, so the depth test hides it.
        xr = x * cos_b - y * sin_b
        yr = x * sin_b + y * cos_b
        return int(cx + xr * kx), int(cy - yr * ky - z * kz), yr

    for x, y, z, nx, ny, nz in samples:
        sx, sy, yr = project(x, y, z)
        if not (0 <= sx < width and 0 <= sy < height):
            continue
        idx = sx + sy * width
        if yr >= depth[idx]:  # something nearer already owns this cell
            continue
        depth[idx] = yr
        # the normal only needs the same yaw rotation as the point
        nxr = nx * cos_b - ny * sin_b
        nyr = nx * sin_b + ny * cos_b
        lum = nxr * lx + nyr * ly + nz * lz
        # fade lighting with altitude: the flat valley floor drops to sparse
        # dots and the massif stays dense, so its silhouette reads at a glance
        lum = 0.1 + max(0.0, lum) * (0.45 + 0.55 * min(z, 1.0))
        # altitude floor: in this flattened oblique view the peaks' back
        # slopes are visible below their caps, and pure shadow would render
        # them as the same sparse dots as the far valley — the mountain then
        # reads as a hollow shell. Keep shadow-side flanks dense in
        # proportion to height so both summits stay solid from every angle,
        # while the flat valley (z ~ 0) keeps its sparse-dot texture.
        lum = max(lum, 0.1 + 0.65 * min(z, 1.0))
        # floor of 1, never 0: the shadow side must render as sparse dots,
        # not as blanks indistinguishable from the sky. Ceiling of ramp-1:
        # '@' is the climbers' character, the surface never wears it.
        chars[idx] = CHARS[max(1, min(ramp - 1, int(lum * (ramp + 1))))]
        # 9.2 rather than len(PALETTE): pulls the snowline below the summit
        # so the cap is a visible patch, not a single white cell
        colors[idx] = PALETTE[max(0, min(len(PALETTE) - 1, int(z * 9.2)))]

    def stamp(x: float, y: float, z: float, char: str, color: int, bias: float = 0.15) -> None:
        """Draw a marker slightly lifted and depth-biased so it sits on the
        surface instead of flickering into it. Climbers and the star pass a
        larger bias: near the summit the slope just in front projects into
        the same cell — and at low angles even the flat foreground can be
        over half a unit nearer — and the protagonists must win those fights."""
        sx, sy, yr = project(x, y, z + 0.05)
        if not (0 <= sx < width and 0 <= sy < height):
            return
        idx = sx + sy * width
        if yr - bias < depth[idx]:
            chars[idx] = char
            colors[idx] = color

    for path, portion, trail_color, climber_color in zip(
        paths, CLIMB_PORTIONS, TRAIL_COLORS, CLIMBER_COLORS
    ):
        progress = min(1.0, frame / (TOTAL_FRAMES * portion))
        at = int(progress * (len(path) - 1))
        for j in range(0, at, 2):
            px, py, pz = path[j]
            stamp(px, py, pz, "·", trail_color)
        px, py, pz = path[at]
        stamp(px, py, pz, "@", climber_color, bias=0.6)
    if frame >= TOTAL_FRAMES * CLIMB_PORTIONS[0]:  # arrival: the summit star
        px, py, pz = paths[0][-1]
        stamp(px, py, pz + 0.08, "★", FLAG_COLOR, bias=0.6)

    reveal = (frame - 0.72 * TOTAL_FRAMES) / (0.20 * TOTAL_FRAMES)
    if reveal > 0 and logo:
        # the logotype sweeps in left to right across its top rows
        shown_cols = max(0, min(logo_w, int(reveal * logo_w)))
        col = (width - logo_w) // 2
        for row, line in enumerate(logo):
            for offset, char in enumerate(line[:shown_cols]):
                if char != " ":
                    idx = col + offset + row * width
                    chars[idx] = char
                    colors[idx] = WORDMARK_COLOR
    elif reveal > 0:
        shown = WORDMARK[: max(0, min(len(WORDMARK), int(reveal * len(WORDMARK))))]
        col = (width - len(WORDMARK)) // 2
        for offset, char in enumerate(shown):
            if char != " ":
                idx = col + offset + 1 * width
                chars[idx] = char
                colors[idx] = WORDMARK_COLOR

    # One string per frame, colour codes only where the colour changes.
    sgr = _sgr_table()
    parts = ["\x1b[H", "\n" * margin_y]
    last_color = None
    pad = " " * margin_x
    for row in range(height):
        parts.append(pad)
        for colm in range(row * width, (row + 1) * width):
            char = chars[colm]
            if char == " ":
                parts.append(" ")
                continue
            color = colors[colm]
            if color != last_color:
                parts.append(sgr[color])
                last_color = color
            parts.append(char)
        parts.append("\x1b[K\n")
    return "".join(parts)


class _Input:
    """The cbreak stdin as events: `"enter"` (Enter, or EOF so a closed
    stdin can never hang the intro), or a mouse tuple `(kind, button, col,
    row)` parsed from an SGR report — `kind` is `"press"`, `"drag"`
    (motion with a button held), `"release"` or `"wheel"`. Any other key is
    swallowed without echoing into the picture. Bytes of a report that
    arrives split across reads are kept for the next poll."""

    def __init__(self, fd: int):
        self.fd = fd
        self.buffer = bytearray()

    def poll(self, timeout: float) -> list:
        """Wait up to `timeout` for input and return every event in it."""
        import select

        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return []
        data = os.read(self.fd, 256)
        if data == b"":
            return ["enter"]
        self.buffer += data
        return self.drain()

    def drain(self) -> list:
        """Every complete event at the front of the buffer, in order."""
        events: list = []
        buf = self.buffer
        while buf:
            if buf.startswith(b"\x1b[<"):
                end = _sgr_end(buf)
                if end is None:
                    if len(buf) > 64:  # not a report after all: drop the junk
                        del buf[:]
                    break  # wait for the rest of the report
                event = _parse_sgr(bytes(buf[3:end]), chr(buf[end]))
                del buf[: end + 1]
                if event is not None:
                    events.append(event)
                continue
            byte = buf[:1]
            del buf[:1]
            if byte in (b"\r", b"\n"):
                events.append("enter")
        return events


def _sgr_end(buf: bytearray) -> int | None:
    """Index of the `M`/`m` that ends the SGR report at the buffer's front."""
    for i in range(3, min(len(buf), 32)):
        if buf[i] in (0x4D, 0x6D):  # M, m
            return i
        if not (0x30 <= buf[i] <= 0x3B):  # digits and ';' only
            return None
    return None


def _parse_sgr(body: bytes, final: str):
    """`b;x;y` + `M`/`m` → a mouse event, or None for a report we ignore."""
    try:
        button, col, row = (int(v) for v in body.split(b";"))
    except ValueError:
        return None
    if button & 64:
        return ("wheel", button & 1, col, row)  # 0 = up, 1 = down
    if final == "m":
        return ("release", button & 3, col, row)
    if button & 32:
        return ("drag", button & 3, col, row)
    return ("press", button & 3, col, row)


def _drag_camera(
    start: tuple[int, int, float, float], col: int, row: int, width: int
) -> tuple[float, float]:
    """The camera for a drag from `start` (col, row, yaw, tilt) to (col,
    row): dragging right turns the near face right with the cursor,
    dragging down raises the eye, the way pulling the near edge toward you
    tips the massif over to show its top."""
    c0, r0, yaw0, tilt0 = start
    yaw = yaw0 + (col - c0) * DRAG_SWEEP / max(1, width)
    tilt = min(TILT_RANGE[1], max(TILT_RANGE[0], tilt0 + (row - r0) * DRAG_TILT))
    return yaw, tilt


def _play() -> None:
    term = shutil.get_terminal_size()
    width, height = min(term.columns, 96), min(term.lines - 1, 28)
    # The render is capped, the terminal may not be: center the picture in it.
    margin_x = max(0, (term.columns - width) // 2)
    margin_y = max(0, (term.lines - 1 - height) // 2)
    samples = _surface_samples()
    paths = tuple(_ascent_path(x, y) for x, y in CLIMBER_STARTS)
    out = sys.stdout
    # Windows has no termios/select on a console: the intro plays through
    # without input there (Ctrl-C still skips it)
    interactive = sys.stdin.isatty() and sys.platform != "win32"
    if sys.platform == "win32":
        os.system("")  # turns on the console's VT processing for the escapes below
    saved = None
    inp = None
    if interactive:
        # cbreak for the whole playback: Enter is read at any point, and ISIG
        # is kept so Ctrl-C still raises KeyboardInterrupt and skips it all.
        import termios
        import tty

        saved = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        inp = _Input(sys.stdin.fileno())
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")  # alternate screen, cursor hidden
    mouse = False
    try:
        for frame in range(TOTAL_FRAMES + 1):
            started = time.monotonic()
            out.write(_render(samples, paths, frame, width, height, margin_x, margin_y))
            out.flush()
            wait = max(0.0, FRAME_S - (time.monotonic() - started))
            if inp is not None:
                # The frame pacing doubles as the keypress poll: Enter at any
                # point skips the rest of the sweep and lands on the end
                # screen, where a second Enter drops into the CLI.
                if "enter" in inp.poll(wait):
                    out.write(_render(samples, paths, TOTAL_FRAMES, width, height, margin_x, margin_y))
                    break
            else:
                time.sleep(wait)
        # Rotation has stopped: hold the finished picture until the user
        # dismisses it (or briefly, when nobody is there to press Enter).
        # Meanwhile the mountain can be grabbed: a drag turns and tilts the
        # camera, the wheel turns it, and the last frame is redrawn at the
        # sweep's own pace.
        prompt_row = margin_y + height + 1
        prompt_col = margin_x + max(0, (width - len(CONTINUE_PROMPT)) // 2) + 1
        prompt = f"\x1b[{prompt_row};{prompt_col}H{_sgr_table()[CONTINUE_COLOR]}{CONTINUE_PROMPT}\x1b[0m"
        out.write(prompt)
        out.flush()
        if inp is None:
            time.sleep(HOLD_S)
            return
        out.write(MOUSE_ON)
        out.flush()
        mouse = True
        yaw, tilt = END_YAW, 1.0
        drag = None  # (col, row, yaw, tilt) at the press
        while True:
            shown = (yaw, tilt)
            for event in inp.poll(FRAME_S):
                if event == "enter":
                    return
                kind, button, col, row = event
                if kind == "press":
                    drag = (col, row, yaw, tilt)
                elif kind == "release":
                    drag = None
                elif kind == "drag" and drag is not None:
                    yaw, tilt = _drag_camera(drag, col, row, width)
                elif kind == "wheel":
                    yaw += DRAG_SWEEP / 30 * (1 if button else -1)
            if (yaw, tilt) != shown:
                out.write(_render(
                    samples, paths, TOTAL_FRAMES, width, height, margin_x, margin_y,
                    yaw=yaw, tilt=tilt,
                ))
                out.write(prompt)
                out.flush()
    finally:
        if mouse:
            out.write(MOUSE_OFF)
        if saved is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
        out.write("\x1b[0m\x1b[?25h\x1b[?1049l")  # terminal exactly as it was
        out.flush()


def play_intro() -> None:
    """Play the animation; Ctrl-C skips it without killing what follows."""
    try:
        _play()
    except KeyboardInterrupt:
        pass


def intro_marker_path():
    return user_config_path().with_name("intro_shown")


def mark_intro_shown() -> None:
    """Retire the auto-play for good; only `hillclimb intro` plays it after this."""
    try:
        path = intro_marker_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError:
        pass


def maybe_play_intro() -> None:
    """Play once ever, and only where it can look good: an interactive,
    reasonably sized, colour-capable terminal outside CI."""
    if os.environ.get("HILLCLIMB_NO_INTRO") or os.environ.get("CI"):
        return
    if os.environ.get("_HILLCLIMB_COMPLETE"):  # shell completion calls the CLI too
        return
    if os.name != "posix" or os.environ.get("TERM", "") in ("", "dumb"):
        return
    if not (sys.stdout.isatty() and sys.stdin.isatty()):
        return
    cols, lines = shutil.get_terminal_size()
    if cols < 60 or lines < 20:
        return
    if intro_marker_path().exists():
        return
    # Marked before playing: a crash mid-intro must not replay it forever.
    mark_intro_shown()
    play_intro()
