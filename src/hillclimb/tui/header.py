"""The hillclimb TUI header: title, and a clock that names its zone.

Textual's header clock renders bare local time. Runs happen on machines in
other zones and journals carry UTC, so the clock says which zone it shows,
and clicking it (or `t`) picks another — persisted in the user's hillclimb
config dir so every TUI agrees. Mirrors the rebase toolkit's header.
"""

from __future__ import annotations

import json
from datetime import datetime, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Header, Input, OptionList, Static
from textual.widgets._header import HeaderTitle

from hillclimb.project import user_config_path

APP_TITLE = "hillclimb"
SYSTEM_ZONE = "system"


def tui_prefs_path() -> Path:
    return user_config_path().with_name("tui.json")


def load_display_timezone() -> ZoneInfo | None:
    """The saved zone, or None for the system's."""
    try:
        name = json.loads(tui_prefs_path().read_text()).get("timezone")
        return ZoneInfo(name) if name else None
    except (OSError, ValueError, KeyError, AttributeError):
        return None


def save_display_timezone(zone: ZoneInfo | None) -> None:
    path = tui_prefs_path()
    try:
        prefs = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        prefs = {}
    prefs["timezone"] = zone.key if zone is not None else None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(prefs, indent=2) + "\n")
    except OSError:
        pass


def display_tzinfo(app: App) -> tzinfo | None:
    zone = getattr(app, "display_timezone", None)
    return zone or datetime.now().astimezone().tzinfo


def clock_text(app: App) -> str:
    now = datetime.now(display_tzinfo(app))
    return f"{now:%H:%M:%S} {now.tzname() or ''}".rstrip()


class HillclimbClock(Static):
    """Header clock naming its zone, flush with the right edge; click to change."""

    DEFAULT_CSS = """
    HillclimbClock {
        dock: right;
        width: auto;
        height: 1;
        padding: 0;
        background: $panel;
        color: $foreground;
        content-align: right middle;
    }
    """

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def render(self) -> RenderResult:
        return Text(clock_text(self.app))

    async def on_click(self, event: events.Click) -> None:
        event.stop()
        await self.run_action("app.choose_timezone")


class HillclimbHeader(Header):
    """Textual's header with the hillclimb clock instead of the stock one."""

    def __init__(self) -> None:
        super().__init__(show_clock=False)

    def compose(self) -> ComposeResult:
        yield HeaderTitle()
        yield HillclimbClock()

    def _on_click(self, event: events.Click) -> None:
        return None  # no tall/short toggle


class TimezoneChoiceScreen(ModalScreen[str | None]):
    """Pick the zone the clock shows. `system` follows the machine."""

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("down", "highlight(1)", "next", show=False),
        Binding("up", "highlight(-1)", "previous", show=False),
    ]
    DEFAULT_CSS = """
    TimezoneChoiceScreen { align: center middle; background: $background 70%; }
    #timezone-dialog { width: 62; height: auto; padding: 1 2; background: $surface; border: solid $primary; }
    #timezone-title { color: $primary; text-style: bold; margin-bottom: 1; }
    #timezone-filter { background: $surface; border: solid $secondary; }
    #timezone-options { height: auto; max-height: 12; background: $surface; border: none; }
    #timezone-hint { color: $secondary; margin-top: 1; }
    """

    def __init__(self, *, current: str) -> None:
        super().__init__()
        self.current = current
        self.zones = [SYSTEM_ZONE, "UTC", *sorted(available_timezones() - {"UTC"})]

    def compose(self) -> ComposeResult:
        with Vertical(id="timezone-dialog"):
            yield Static(f"Show times in — currently {self.current}", id="timezone-title")
            yield Input(placeholder="filter, e.g. stockholm", id="timezone-filter")
            yield OptionList(*self.zones, id="timezone-options")
            yield Static("Enter picks the highlighted zone. Escape cancels.", id="timezone-hint")

    def on_mount(self) -> None:
        self.query_one("#timezone-filter", Input).focus()

    def matches(self, query: str) -> list[str]:
        needle = query.strip().casefold().replace(" ", "_")
        if not needle:
            return self.zones
        return [zone for zone in self.zones if needle in zone.casefold()]

    def on_input_changed(self, event: Input.Changed) -> None:
        options = self.query_one("#timezone-options", OptionList)
        options.clear_options()
        options.add_options(self.matches(event.value))
        if options.option_count:
            options.highlighted = 0

    def on_input_submitted(self, event: Input.Submitted) -> None:
        options = self.query_one("#timezone-options", OptionList)
        if options.option_count and options.highlighted is not None:
            self.dismiss(str(options.get_option_at_index(options.highlighted).prompt))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def action_highlight(self, delta: int) -> None:
        options = self.query_one("#timezone-options", OptionList)
        if not options.option_count:
            return
        current = options.highlighted if options.highlighted is not None else -1
        options.highlighted = max(0, min(options.option_count - 1, current + delta))

    def action_cancel(self) -> None:
        self.dismiss(None)


class TimezoneMixin:
    """For Apps: `display_timezone` + the `choose_timezone` action the clock calls."""

    TITLE = APP_TITLE
    ENABLE_COMMAND_PALETTE = False
    display_timezone: ZoneInfo | None = None

    def _init_timezone(self) -> None:
        self.display_timezone = load_display_timezone()

    def action_choose_timezone(self) -> None:
        current = self.display_timezone.key if self.display_timezone else SYSTEM_ZONE

        def chosen(choice: str | None) -> None:
            if choice is None:
                return
            self.display_timezone = None if choice == SYSTEM_ZONE else ZoneInfo(choice)
            save_display_timezone(self.display_timezone)
            for clock in self.query(HillclimbClock):
                clock.refresh()

        self.push_screen(TimezoneChoiceScreen(current=current), chosen)
