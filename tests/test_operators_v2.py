"""Operator scaffolds v2: retrieval-augmented draft and ablation-guided
improve — prompt injection and its config gates (operators:)."""

from __future__ import annotations

import sys
from pathlib import Path

from hillclimb.agents.fake import FakeAgent
from hillclimb.harness.budget import BudgetManager
from tests.conftest import local_executor
from hillclimb.harness.journal import Journal
from tests.harness_factory import SearchRig
from hillclimb.harness.dirs import create_search_dir
from tests.conftest import ok_script


def make_searcher(task, config, agent, max_candidates=10):
    search_dir = create_search_dir(config.paths.runs_dir, "test-run")
    journal = Journal(search_dir / "journal.jsonl")
    searcher = SearchRig(
        problem=task,
        config=config,
        journal=journal,
        agent=agent,
        executor=local_executor(),
        budget=BudgetManager(3600, stop_margin_s=1),
        search_dir=search_dir,
        max_candidates=max_candidates,
        log=lambda *_: None,
    )
    return searcher, journal, search_dir


# --- retrieval-augmented draft ---


def test_draft_prompt_carries_research_cue_by_default(task, config):
    searcher, _, _ = make_searcher(task, config, FakeAgent())
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "# Research first" in prompt
    assert "state of the art" in prompt
    assert "do NOT search for" in prompt  # competition-leakage guard
    # the cue restates the execution-time network rule for the solution script
    assert prompt.count("Assume no internet access at execution time.") >= 2
    assert "{{" not in prompt


def test_draft_research_cue_gated_off(task, config):
    config.climber.operators.setdefault("draft", {})["retrieval"] = False
    searcher, _, _ = make_searcher(task, config, FakeAgent())
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "# Research first" not in prompt
    assert "{{" not in prompt


def test_offline_agent_gets_no_research_cue(task, config):
    """An agent without internet cannot do the web research the cue asks
    for; the solution's own network rule still reaches the contract."""
    config.allow_internet_for_agents = False
    searcher, _, _ = make_searcher(task, config, FakeAgent())
    prompt = searcher.build_prompt("draft", None, "minimal")
    assert "# Research first" not in prompt
    assert "Assume no internet access at execution time." in prompt


# --- ablation-guided improve ---


def scored_target(task, config, agent):
    agent.queue(script=ok_script(0.6), notes="draft one\n")
    searcher, journal, _ = make_searcher(task, config, agent)
    target = searcher.run_operator("draft", None)
    assert target.status == "passing"
    return searcher, journal, target


def test_improve_prompt_carries_ablation_cue_by_default(task, config):
    searcher, _, target = scored_target(task, config, FakeAgent())
    prompt = searcher.build_prompt("improve", target, None)
    assert "Ablation study" in prompt
    assert "ablation.md" in prompt
    assert "Targeted refinement" in prompt
    # the one-change discipline survives the restructure
    assert "exactly ONE measurable change" in prompt
    assert "{{" not in prompt


def test_improve_ablation_cue_gated_off(task, config):
    config.climber.operators.setdefault("improve", {})["ablation"] = False
    searcher, _, target = scored_target(task, config, FakeAgent())
    prompt = searcher.build_prompt("improve", target, None)
    assert "Ablation study" not in prompt
    assert "exactly ONE measurable change" in prompt
    assert "{{" not in prompt


def test_improve_prompt_reuses_sibling_ablation(task, config):
    """An earlier improve attempt's ablation.md (analyzing the same target
    solution) is fed to the next improve of that target, newest first."""
    agent = FakeAgent()
    searcher, _, target = scored_target(task, config, agent)
    agent.queue(script=ok_script(0.7), notes="model: swap to gbm\n")
    child = searcher.run_operator("improve", target)
    Path(child.candidate_dir, "ablation.md").write_text(
        "- features: +0.04\n- model: +0.01\n- target: features\n"
    )
    prompt = searcher.build_prompt("improve", target, None)
    assert "Prior ablation findings" in prompt
    assert child.candidate_id in prompt
    assert "- features: +0.04" in prompt
    assert "do NOT redo them" in prompt


def test_prior_ablations_empty_without_files_or_when_gated(task, config):
    agent = FakeAgent()
    searcher, _, target = scored_target(task, config, agent)
    assert "Prior ablation findings" not in searcher.build_prompt("improve", target, None)
    agent.queue(script=ok_script(0.7), notes="tweak\n")
    child = searcher.run_operator("improve", target)
    Path(child.candidate_dir, "ablation.md").write_text("- model: +0.02\n")
    config.climber.operators.setdefault("improve", {})["ablation"] = False
    assert "Prior ablation findings" not in searcher.build_prompt("improve", target, None)
