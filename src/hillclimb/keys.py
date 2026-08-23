"""The `?` keys panel: every command of the current screen, in a panel that
splits out from the right edge.

Footers only carry the few keys worth a permanent slot; everything else is
`show=False` and lives here. The panel reads the host screen's BINDINGS at
render time so it cannot drift from the keys that actually work, and a
screen can list pointer gestures that Textual bindings cannot express (the
graph's drag/scroll camera) in `POINTER_HELP`.
"""

from __future__ import annotations

from rich.table import Table
from rich.text import Text
from textual.binding import Binding
from textual.widgets import Static

KEYS_BINDING = Binding("question_mark", "toggle_keys", "keys")
QUIT_BINDINGS = [
    Binding("q", "app.quit", "quit"),
    # Textual maps ctrl+c to a "press ctrl+q to quit" hint; people expect it
    # to quit. Priority so it wins over any focused widget's own ctrl+c.
    Binding("ctrl+c", "app.quit", "quit", show=False, priority=True),
]


class KeysPanel(Static):
    """Every command of the screen that mounted it, gestures first."""

    COMPONENT_CLASSES = {"keys-panel--key", "keys-panel--description"}

    DEFAULT_CSS = """
    KeysPanel {
        split: right;
        width: 30;
        height: 1fr;
        padding: 1 2 0 1;
        border-left: vkey $foreground 30%;
        background: $surface;
    }
    KeysPanel > .keys-panel--key { color: $text-accent; text-style: bold; }
    KeysPanel > .keys-panel--description { color: $text-muted; }
    """

    # Textual key names that read badly on a key cap.
    KEY_NAMES = {
        "escape": "esc",
        "enter": "enter",
        "slash": "/",
        "question_mark": "?",
        "left_square_bracket": "[",
        "right_square_bracket": "]",
        "plus": "+",
        "minus": "-",
    }

    def pointer_rows(self) -> list[tuple[str, str]]:
        return list(getattr(self.screen, "POINTER_HELP", []))

    def rows(self) -> list[tuple[str, str]]:
        """(keys, what it does) for the whole screen — gestures, then the
        screen's keys, then the app's."""
        rows = self.pointer_rows()
        seen: set[str] = set()
        for owner in (self.screen, self.app):
            for binding in getattr(owner, "BINDINGS", []):
                if not isinstance(binding, Binding) or not binding.description:
                    continue
                keys = binding.key_display or " ".join(
                    self.KEY_NAMES.get(k, k) for k in binding.key.split(",")
                )
                if keys in seen:
                    continue
                seen.add(keys)
                rows.append((keys, binding.description))
        return rows

    def render(self) -> Table:
        key_style = self.get_component_rich_style("keys-panel--key")
        text_style = self.get_component_rich_style("keys-panel--description")
        table = Table(box=None, show_header=False, padding=(0, 1, 0, 0))
        table.add_column(justify="right")
        table.add_column()
        gestures = len(self.pointer_rows())
        for index, (keys, description) in enumerate(self.rows()):
            if gestures and index == gestures:  # gestures above, keys below
                table.add_row("", "")
            table.add_row(Text(keys, style=key_style), Text(description, style=text_style))
        return table


class KeysMixin:
    """For Screens: `?` toggles the KeysPanel. `split: right` reserves a
    column instead of overlaying, so the content shrinks and repaints (a
    kitty-graphics plot needs that)."""

    def action_toggle_keys(self) -> None:
        panel = self.query(KeysPanel)
        if panel:
            panel.remove()
        else:
            self.mount(KeysPanel())

    def keys_open(self) -> bool:
        return bool(self.query(KeysPanel))
