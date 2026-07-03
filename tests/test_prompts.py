from hillclimb.prompts.render import COMPLEXITY_CUES, render


def test_render_replaces_tokens():
    text = render(
        "contract",
        metric_name="accuracy",
        exec_timeout_min=30,
        runtime_pkgs="pandas, numpy",
        time_remaining="1h 30m",
    )
    assert "accuracy" in text
    assert "30 minutes" in text
    assert "{{" not in text


def test_render_safe_with_braces():
    # competition descriptions contain literal {braces}; render must not choke
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
    from hillclimb.node import Node
    from hillclimb.viz import build_tree

    journal = Journal(tmp_path / "j.jsonl")
    journal.node_result(Node(node_id="n001", operator="draft", status="ok", val_score=0.6))
    journal.node_result(Node(node_id="n002", operator="debug", parent_id="n001", status="buggy"))
    journal.node_result(Node(node_id="n003", operator="improve", parent_id="n001",
                             status="ok", val_score=0.8, summary="one change"))
    graph = build_tree(journal, lower_is_better=False)
    unq = lambda s: str(s).strip('"')
    nodes = {unq(n.get_name()): n for n in graph.get_nodes()}
    assert unq(nodes["n003"].get("fillcolor")) == "#fff59d"  # best = gold
    assert unq(nodes["n002"].get("fillcolor")) == "#ffcdd2"  # buggy = red
    edges = {(unq(e.get_source()), unq(e.get_destination())): e for e in graph.get_edges()}
    assert unq(edges[("n001", "n002")].get("style")) == "dashed"  # debug edge
