"""First-run intro: a rotating 3D hill, climbed live in the terminal.

Pure Python, stdlib only — the classic terminal-donut trick pointed at this
tool's own subject. A height field of Gaussian peaks is sampled once; every
frame the samples are yaw-rotated, projected onto the character grid through
a depth buffer, and shaded by how much light each patch of surface catches
(character ramp) and by altitude (cyan ramp, the CLI's palette). Two greedy
climbers walk their gradient-ascent paths as it turns: one reaches the
summit and the flag, one tops out on a local optimum and stays there —
hillclimb's pitch, drawn instead of written. The wordmark fades in last.

It plays once, the first time hillclimb runs in an interactive terminal,
then never again (marker file next to the user config). `hillclimb intro`
replays it; `hillclimb intro --reset` re-arms the first run; `hillclimb
--skip-intro` retires the auto-play without watching it. A non-TTY, CI,
HILLCLIMB_NO_INTRO, a dumb or tiny terminal, or shell completion all skip
it, and Ctrl-C skips the intro without killing the command behind it.
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
# 256-color teals by altitude, ending in white: the summit gets a snowcap,
# the single strongest "this is a mountain" cue a character grid can give.
PALETTE = (23, 30, 36, 43, 50, 87, 159, 231)
TRAIL_COLORS = (250, 240)
CLIMBER_COLORS = (231, 245)
FLAG_COLOR = 220
WORDMARK = "h i l l c l i m b"
WORDMARK_COLOR = 51

# Light mostly from the side, not from above: flat ground lands mid-ramp,
# slopes facing the light go dense, slopes facing away go dark.
LIGHT = (-0.55, -0.60, 0.58)

TOTAL_FRAMES = 200
FRAME_S = 0.04
HOLD_S = 1.5  # non-interactive fallback hold; a TTY waits for Enter instead
CONTINUE_PROMPT = "Press enter to continue"
CONTINUE_COLOR = 245
# Per-climber climb duration, as a fraction of the animation. The summit
# climber finishes early on purpose: its north-face route swings behind the
# ridge partway through the long rotation sweep, and by then it must already
# be sitting on the peak — the one spot that is visible from every angle.
CLIMB_PORTIONS = (0.40, 0.68)


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
) -> str:
    # Sweep from 3.65 down to 0.63 rad: along this whole arc the local-optimum
    # bump stays on the camera side of the massif, so both summits are in view
    # from the first frame instead of the small one appearing mid-rotation.
    # The end angle is where the two summits project to equal depth — the
    # line connecting them lies exactly perpendicular to the line of sight,
    # so the closing shot reads summit and local optimum side by side.
    yaw = 3.65 - 3.02 * frame / TOTAL_FRAMES
    cos_b, sin_b = math.cos(yaw), math.sin(yaw)
    kx = width / (2 * GRID_EXTENT) * 0.88  # small side margin: the climbers'
    # starting corners must stay on screen at every rotation angle
    ky = height / (2 * GRID_EXTENT) * 0.42  # gentle tilt: an elevated front view
    kz = height * 0.48  # exaggerated relief, the way trail maps draw it
    cx, cy = width / 2, height * 0.62
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
        surface instead of flickering into it. Climbers and the flag pass a
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
    if frame >= TOTAL_FRAMES * CLIMB_PORTIONS[0]:  # arrival: plant the flag
        px, py, pz = paths[0][-1]
        stamp(px, py, pz + 0.08, "▲", FLAG_COLOR, bias=0.6)

    reveal = (frame - 0.72 * TOTAL_FRAMES) / (0.20 * TOTAL_FRAMES)
    if reveal > 0:
        shown = WORDMARK[: max(0, min(len(WORDMARK), int(reveal * len(WORDMARK))))]
        col = (width - len(WORDMARK)) // 2
        for offset, char in enumerate(shown):
            if char != " ":
                idx = col + offset + 1 * width
                chars[idx] = char
                colors[idx] = WORDMARK_COLOR

    # One string per frame, colour codes only where the colour changes.
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
                parts.append(f"\x1b[38;5;{color}m")
                last_color = color
            parts.append(char)
        parts.append("\x1b[K\n")
    return "".join(parts)


def _wait_for_enter() -> None:
    """Block until Enter, without echoing keys into the picture. cbreak keeps
    ISIG, so Ctrl-C still raises KeyboardInterrupt and skips out."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while sys.stdin.read(1) not in ("\r", "\n", ""):
            pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def _play() -> None:
    term = shutil.get_terminal_size()
    width, height = min(term.columns, 96), min(term.lines - 1, 28)
    # The render is capped, the terminal may not be: center the picture in it.
    margin_x = max(0, (term.columns - width) // 2)
    margin_y = max(0, (term.lines - 1 - height) // 2)
    samples = _surface_samples()
    paths = tuple(_ascent_path(x, y) for x, y in CLIMBER_STARTS)
    out = sys.stdout
    out.write("\x1b[?1049h\x1b[?25l\x1b[2J")  # alternate screen, cursor hidden
    try:
        for frame in range(TOTAL_FRAMES + 1):
            started = time.monotonic()
            out.write(_render(samples, paths, frame, width, height, margin_x, margin_y))
            out.flush()
            time.sleep(max(0.0, FRAME_S - (time.monotonic() - started)))
        # Rotation has stopped: hold the finished picture until the user
        # dismisses it (or briefly, when nobody is there to press Enter).
        row = margin_y + height + 1
        col = margin_x + max(0, (width - len(CONTINUE_PROMPT)) // 2) + 1
        out.write(f"\x1b[{row};{col}H\x1b[38;5;{CONTINUE_COLOR}m{CONTINUE_PROMPT}\x1b[0m")
        out.flush()
        if sys.stdin.isatty():
            _wait_for_enter()
        else:
            time.sleep(HOLD_S)
    finally:
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
