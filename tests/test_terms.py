"""The vocabulary module: one place decides what a concept is called."""

from __future__ import annotations

from hillclimb import terms
from hillclimb.terms import Term


def test_a_term_gives_every_form_a_sentence_needs():
    engine = Term("engine", "engines")
    assert (str(engine), engine.pl, engine.a, engine.cap, engine.cap_pl) == (
        "engine", "engines", "an engine", "Engine", "Engines"
    )
    assert Term("search", "searches").a == "a search"
    assert (engine.n(1), engine.n(3), engine.form(1), engine.form(0)) == ("1 engine", "3 engines", "engine", "engines")


def test_fill_replaces_only_term_placeholders(monkeypatch):
    monkeypatch.setitem(terms.TERMS, "engine", Term("engine", "engines"))
    text = 'Stop {an_engine}; {Engines} and {engines}; the {engine}. {Engine}! {"json": {other}}'
    assert terms.fill(text) == 'Stop an engine; Engines and engines; the engine. Engine! {"json": {other}}'


def test_doc_fills_a_docstring():
    @terms.doc
    def stop():
        """Stop a running {engine} now."""

    assert stop.__doc__ == f"Stop a running {terms.ENGINE} now."


def test_role_label_renames_only_the_search_process():
    assert terms.role_label("engine") == str(terms.ENGINE)
    assert terms.role_label("verifier") == "verifier"


def test_no_command_help_is_left_with_an_unfilled_placeholder():
    """A docstring with `{engine}` but no `@terms.doc` would print the braces."""
    import typer.main

    from hillclimb.cli import app

    def walk(command):
        yield command
        for sub in getattr(command, "commands", {}).values():
            yield from walk(sub)

    for command in walk(typer.main.get_command(app)):
        assert "{engine" not in (command.help or "").lower() and "{an_engine" not in (command.help or ""), command.name
