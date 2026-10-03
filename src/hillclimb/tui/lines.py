"""The one-line readouts over the TUI's plots (`chart`, `tree`,
`treeclimb`), wrapped between their items when a terminal is too narrow,
instead of cut off at its edge."""

from __future__ import annotations


def flow_items(items: list[str], dim_items: list[str], width: int, sep: str = " · ") -> str:
    """Rich markup items joined by `sep` on one line when they all fit in
    `width`. When they do not, `items` flow over as many lines as they need,
    broken between two items, never inside one, and `dim_items` (dimmed)
    start a line of their own, so a group is never split by a stray item.
    An item wider than `width` gets a line of its own (the label cuts it)."""
    from rich.text import Text

    def size(item: str) -> int:
        return Text.from_markup(item).cell_len

    def flow(group: list[str]) -> list[list[str]]:
        lines: list[list[str]] = [[]]
        used = 0
        for item in group:
            if lines[-1] and used + len(sep) + size(item) > width:
                lines.append([])
                used = 0
            used += (len(sep) if lines[-1] else 0) + size(item)
            lines[-1].append(item)
        return [line for line in lines if line]

    dimmed = [f"[dim]{item}[/]" for item in dim_items]
    every = [*items, *dimmed]
    if sum(map(size, every)) + len(sep) * (len(every) - 1) <= width:
        return sep.join(every)
    return "\n".join(sep.join(line) for line in [*flow(items), *flow(dimmed)])
