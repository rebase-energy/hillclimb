"""The words hillclimb uses for its concepts, in one place.

A word a reader meets in help text, a message or a screen label is a
product decision, and it can change: what the code calls an *engine* (the
background process that runs one search) the user is told is a *search*,
because that is the thing they started, watch and stop. Text that names a
concept takes the word from here instead of spelling it, so changing the
word back is one line in this file — not a hunt through every command.

    from hillclimb.terms import ENGINE
    say(f"No {ENGINE.pl} running.")            # an f-string
    @terms.doc                                  # a command's help (its docstring)
    def stop(): '''Stop a running {engine} now.'''

A docstring's `{engine}` / `{engines}` / `{an_engine}` / `{Engine}` /
`{Engines}` are filled in by `doc`; any other braces are left alone, so
JSON or f-string examples in a docstring are safe.

The code keeps its own names (`Engine`, `live_engines`): there the process
and the search record really are two things.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Term:
    word: str
    plural: str

    def __str__(self) -> str:
        return self.word

    @property
    def pl(self) -> str:
        return self.plural

    @property
    def a(self) -> str:
        """With its article: `a search`, `an engine`."""
        return f"{'an' if self.word[0] in 'aeiou' else 'a'} {self.word}"

    @property
    def cap(self) -> str:
        return self.word.capitalize()

    @property
    def cap_pl(self) -> str:
        return self.plural.capitalize()

    def form(self, count: int) -> str:
        """Singular or plural for `count`: `search`, `searches`."""
        return self.word if count == 1 else self.plural

    def n(self, count: int) -> str:
        """The word for `count` of them: `1 search`, `3 searches`."""
        return f"{count} {self.form(count)}"


# The background process that runs one search. Was "engine"/"engines".
ENGINE = Term("search", "searches")

TERMS = {"engine": ENGINE}


def role_label(role: str) -> str:
    """What a process role is called on screen. The code's role keys stay
    as they are (`engine`); only the label follows the term."""
    return ENGINE.word if role == "engine" else role


def fill(text: str) -> str:
    """`text` with each term's placeholders replaced by its current word."""
    for name, term in TERMS.items():
        for key, value in (
            (f"{{an_{name}}}", term.a),
            (f"{{{name}s}}", term.pl),
            (f"{{{name}}}", term.word),
            (f"{{{name.capitalize()}s}}", term.cap_pl),
            (f"{{{name.capitalize()}}}", term.cap),
        ):
            text = text.replace(key, value)
    return text


def doc(fn):
    """Decorator: fill the term placeholders in a function's docstring (a
    command's help). Goes under `@app.command()`, so the help Typer reads
    is already filled."""
    if fn.__doc__:
        fn.__doc__ = fill(fn.__doc__)
    return fn
