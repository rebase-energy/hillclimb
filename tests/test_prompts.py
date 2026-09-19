from tests.factories import trial as mk_trial
from hillclimb.operators.builtin import COMPLEXITY_CUES
from hillclimb.prompts.render import render
import pytest


def test_render_replaces_tokens():
    text = render(
        "contract_verifier",
        metric_name="accuracy",
        exec_timeout_min=30,
        runtime_pkgs="pandas, numpy",
        time_remaining="1h 30m",
        verifier_display="./problem/verifier.sh",
        problem_contract="solution.py must define answer().",
        interface_section="",
        params_section="",
        holdout_clause="",
        tools_clause="",
        network_note="Assume no internet access at execution time.",
    )
    assert "accuracy" in text
    assert "30 minutes" in text
    assert "./problem/verifier.sh" in text
    assert "{{" not in text

    with_holdout = render(
        "contract_verifier",
        metric_name="accuracy",
        exec_timeout_min=30,
        runtime_pkgs="pandas, numpy",
        time_remaining="1h 30m",
        verifier_display="./problem/verifier.sh",
        problem_contract="solution.py must define answer().",
        interface_section="",
        params_section="",
        tools_clause="",
        network_note="Assume no internet access at execution time.",
        holdout_clause=render("holdout_clause", metric_name="accuracy").rstrip(),
    )
    assert "Hidden holdout" in with_holdout
    assert "never see" in with_holdout
    assert "{{" not in with_holdout


def test_self_reported_contract_still_renders():
    """MLE-bench-shaped problems: the agent prints its own score and may
    write its own report."""
    text = render(
        "contract_submission",
        metric_name="accuracy",
        exec_timeout_min=30,
        runtime_pkgs="pandas, numpy",
        time_remaining="1h 30m",
        verifier_clause="- prints exactly one line `val_score: <float>` for accuracy",
        report_clause=render("report_clause").rstrip(),
        interface_section="",
        params_section="",
        holdout_clause="",
        tools_clause="",
        network_note="Assume no internet access at execution time.",
    )
    assert "val_score" in text
    assert "eval_result.json" in text
    assert "{{" not in text


def test_interface_section_renders_into_contract():
    text = render(
        "contract_verifier",
        metric_name="accuracy",
        exec_timeout_min=30,
        runtime_pkgs="pandas, numpy",
        time_remaining="1h 30m",
        verifier_display="./problem/verifier.sh",
        problem_contract="(see the problem description above)",
        interface_section="\n## Output interface (machine-checked)\n\nFile `submission.csv` (CSV)\n",
        params_section="",
        holdout_clause="",
        tools_clause="",
        network_note="Assume no internet access at execution time.",
    )
    assert "## Output interface (machine-checked)" in text
    assert "{{" not in text


def test_render_safe_with_braces():
    # problem descriptions contain literal {braces}; render must not choke
    text = render("draft", description="use {'a': 1} dicts", metric_name="x",
                  direction="higher is better", data_listing="- train.csv",
                  complexity_cue=COMPLEXITY_CUES["minimal"], prior_drafts="(none)",
                  contract="CONTRACT")
    assert "use {'a': 1} dicts" in text
    assert "CONTRACT" in text


def test_all_complexity_cues_distinct():
    assert len(set(COMPLEXITY_CUES.values())) == 3


def test_build_tree_marks_best_and_edges(tmp_path):
    from hillclimb.journal import Journal
    from hillclimb.candidate import Candidate
    from hillclimb.viz import build_tree

    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(Candidate(candidate_id="c001", operator="draft", status="passing",
                                       trials=[mk_trial(val_score=0.6)]))
    journal.candidate_result(Candidate(candidate_id="c002", operator="debug", parent_id="c001", status="buggy"))
    journal.candidate_result(Candidate(candidate_id="c003", operator="improve", parent_id="c001",
                                       status="passing", trials=[mk_trial(val_score=0.8)],
                                       summary="one change"))
    graph = build_tree(journal, higher_is_better=True)
    unq = lambda s: str(s).strip('"')
    nodes = {unq(n.get_name()): n for n in graph.get_nodes()}
    assert unq(nodes["c003"].get("fillcolor")) == "#fff59d"  # best = gold
    assert unq(nodes["c002"].get("fillcolor")) == "#ffcdd2"  # buggy = red
    edges = {(unq(e.get_source()), unq(e.get_destination())): e for e in graph.get_edges()}
    assert unq(edges[("c001", "c002")].get("style")) == "dashed"  # debug edge


# --- prompt overrides: <hillclimb dir>/prompts/<name>.md shadows the package ---


def test_override_dir_shadows_package_template_and_is_hashed(tmp_path, monkeypatch):
    from hillclimb.prompts import render as r

    baseline = r.templates_digest(None)
    assert baseline.overridden == [] and baseline == r.templates_digest(tmp_path / "missing")

    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "holdout_clause.md").write_text("custom clause for {{metric_name}}\n")
    (prompts / "brand_new.md").write_text("no tokens\n")
    digest = r.templates_digest(prompts)
    assert digest.overridden == ["holdout_clause"]
    assert digest.sha256 != baseline.sha256
    assert digest == r.templates_digest(prompts)  # deterministic
    assert "brand_new" in r.template_names(prompts)

    # not active until set: render still reads the package
    assert "custom clause" not in r.render("holdout_clause", metric_name="rmse")
    previous = r.set_override_dir(prompts)
    try:
        assert r.render("holdout_clause", metric_name="rmse") == "custom clause for rmse\n"
        assert r.template_path("draft") == r.TEMPLATE_DIR / "draft.md"  # untouched names fall through
        assert r.templates_digest().sha256 == digest.sha256  # active dir is the default
    finally:
        r.set_override_dir(previous)


def test_override_lint_rejects_unknown_tokens_and_empty_files(tmp_path):
    from hillclimb.prompts import render as r

    assert r.lint_overrides(None) == [] and r.lint_overrides(tmp_path / "missing") == []
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "improve.md").write_text("{{best_score}} {{made_up}} {{another}}\n")
    (prompts / "debug.md").write_text("   \n")
    (prompts / "draft.md").write_text("leaner: {{description}}\n")  # fewer tokens is fine
    (prompts / "custom_contract.md").write_text("{{anything}}\n")  # no package counterpart: free-form
    problems = r.lint_overrides(prompts)
    assert len(problems) == 2
    assert problems[0].startswith("debug.md: empty")
    assert "improve.md: unknown token(s) {{another}}, {{made_up}}" in problems[1]


def test_search_record_pins_the_climber_and_snapshots_it(config, tmp_path):
    """The prompts are part of the exploration process: they live in the
    climber, whose tree hash is the search's identity and whose files are
    snapshotted into the search dir — what the engine and a resume load."""
    from hillclimb.api import create_run, create_search
    from hillclimb.climber import load_climber, load_snapshot
    from hillclimb.problem import load_problem
    from hillclimb.run import RunMeta, load_search_meta
    from tests.test_cli import write_problem

    root = tmp_path / "problems"
    write_problem(root, "p")
    config.paths.problems_dir = root
    run_dir = create_run(config, RunMeta(run_id="r1", name="r1", kind="problem", target="p", problem_ids=["p"]))

    climber_dir = tmp_path / "mine"
    (climber_dir / "prompts").mkdir(parents=True)
    (climber_dir / "prompts" / "improve.md").write_text("tighter improve prompt: {{best_score}}\n\n{{contract}}\n")
    (climber_dir / "climber.yaml").write_text(
        "policy: hillclimb.policies.greedy:GreedyPolicy\nparams: {num_drafts: 1}\nprompts: prompts\n"
    )
    config.search.policy = str(climber_dir)
    search_dir = create_search(config, load_problem("p", config), run_dir, "r1", 60)
    meta = load_search_meta(search_dir)
    before = load_climber(str(climber_dir)).sha256
    assert (meta.climber, meta.climber_sha256) == (str(climber_dir), before)
    assert meta.climber_manifest["params"] == {"num_drafts": 1} and meta.hillclimb_version

    # the author keeps iterating on the live dir; the search keeps what it started with
    (climber_dir / "prompts" / "improve.md").write_text("a different prompt: {{best_score}}\n\n{{contract}}\n")
    assert load_climber(str(climber_dir)).sha256 != before
    snapshot = load_snapshot(search_dir, name="mine")
    assert snapshot.sha256 == before and snapshot.name == "mine"
    assert (snapshot.prompts_dir / "improve.md").read_text().startswith("tighter improve prompt")


def test_engine_refuses_to_start_on_a_broken_override(config, tmp_path):
    from hillclimb.api import _activate_prompt_overrides
    from hillclimb.prompts import render as r

    config.paths.prompts_dir = tmp_path / "prompts"
    config.paths.prompts_dir.mkdir()
    (config.paths.prompts_dir / "draft.md").write_text("{{typo_token}}\n")
    logged: list[str] = []
    previous = r.set_override_dir(None)
    try:
        with pytest.raises(ValueError, match="typo_token"):
            _activate_prompt_overrides(config, logged.append)
        assert r.override_dir() is None  # nothing activated
        (config.paths.prompts_dir / "draft.md").write_text("{{description}}\n")
        _activate_prompt_overrides(config, logged.append)
        assert r.override_dir() == config.paths.prompts_dir
        assert logged == [f"prompt overrides from {config.paths.prompts_dir}: draft"]
    finally:
        r.set_override_dir(previous)
