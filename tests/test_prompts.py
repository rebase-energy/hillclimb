from hillclimb.prompts.render import COMPLEXITY_CUES, render


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
    from hillclimb.candidate import Candidate, Trial
    from hillclimb.viz import build_tree

    journal = Journal(tmp_path / "j.jsonl")
    journal.candidate_result(Candidate(candidate_id="c001", operator="draft", status="ok",
                                       trials=[Trial(val_score=0.6)]))
    journal.candidate_result(Candidate(candidate_id="c002", operator="debug", parent_id="c001", status="buggy"))
    journal.candidate_result(Candidate(candidate_id="c003", operator="improve", parent_id="c001",
                                       status="ok", trials=[Trial(val_score=0.8)],
                                       summary="one change"))
    graph = build_tree(journal, higher_is_better=True)
    unq = lambda s: str(s).strip('"')
    nodes = {unq(n.get_name()): n for n in graph.get_nodes()}
    assert unq(nodes["c003"].get("fillcolor")) == "#fff59d"  # best = gold
    assert unq(nodes["c002"].get("fillcolor")) == "#ffcdd2"  # buggy = red
    edges = {(unq(e.get_source()), unq(e.get_destination())): e for e in graph.get_edges()}
    assert unq(edges[("c001", "c002")].get("style")) == "dashed"  # debug edge
