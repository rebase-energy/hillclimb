"""`memory: files` — the built-in Memory — and `memory: none`.

`FilesMemory` is the YAML under `knowledge/`: a card per finished search,
the claims distilled into it, credit for the claims a search was shown,
playbooks, a skill library, and the graph that indexes them. Its steps are
the ones `Memory` names (base.py):

    retrieve   prior cards, then a consolidated playbook or graph-ranked
               claims, a proven solution for the first draft, and a learned
               draft-complexity offset for the policy
    live       the cards sibling searches of the same run have published so far
    publish    this search's card, for them
    record     the card, its claims (one agent pass, `routing.distill`),
               credit for the claims it was shown, a harvested skill, the graph

Every step is best effort: memory never fails a search. The settings are the
block's `memory_params` (`DEFAULTS` below); where the knowledge lives
(`learning.dir`), whether agents get the lookup tool (`learning.tool`) and the
distill pass's timeout (`learning.claims_timeout_s`) are the user's, in
hillclimb.yaml.

`NoMemory` reads and writes nothing across searches; a finished search still
gets its own card beside its artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from hillclimb.modules import refs
from hillclimb.modules.memory.base import DEFAULT_GRAPH, GraphModule, Memory, MemoryEnv, Retrieved

KIND = "memory"


def knowledge_dir_for(config) -> Path | None:
    """Where the knowledge lives: `learning.dir`, else `<hillclimb dir>/knowledge`;
    None when the user switched learning off or no dir is resolvable."""
    if not config.learning.enabled:
        return None
    if config.learning.dir is not None:
        return Path(config.learning.dir).absolute()
    if config.hillclimb_dir is not None:
        return config.hillclimb_dir / "knowledge"
    return None


def _search_card(env: MemoryEnv, journal, *, budget_s: int, cost_usd: float):
    from hillclimb.harness.run import search_ref
    from hillclimb.modules.memory.knowledge import distill_card

    return distill_card(
        journal,
        problem=env.problem,
        run_ref=search_ref(env.search_dir),
        target=env.target,
        budget_s=budget_s,
        cost_usd=cost_usd,
        selection=env.config.holdout.selection,
    )


def _keep_with_the_search(env: MemoryEnv, card) -> None:
    """A copy of the card beside the search's artifacts (synced for hosted runs)."""
    import yaml

    from hillclimb.modules.memory.knowledge import CARD_FILENAME

    (env.search_dir / CARD_FILENAME).write_text(
        yaml.safe_dump(card.model_dump(exclude_none=True), sort_keys=False)
    )


class NoMemory(Memory):
    """No cross-search memory: nothing is read, nothing is shared or kept."""

    name = "none"
    enabled = False

    def record(self, journal, *, budget_s: int, cost_usd: float) -> None:
        env = self.env
        try:
            _keep_with_the_search(env, _search_card(env, journal, budget_s=budget_s, cost_usd=cost_usd))
        except Exception as exc:  # noqa: BLE001
            env.log(f"learning: card distillation failed (search result unaffected): {exc}")


class FilesMemory(Memory):
    """Knowledge cards, claims, credit, playbooks and skills, as files under knowledge/."""

    name = "files"
    DEFAULTS = {
        "max_cards": 3,  # cards rendered into the prompt
        # live sharing: republish this search's card after every executed
        # candidate and read siblings' cards (runs/<run-id>/knowledge/), so
        # concurrent searches in one run learn from each other mid-flight
        "live": True,
        # opt-in policy bias: start the draft complexity schedule one step up
        # when past winners were never 'minimal'
        "complexity_prior": False,
        # semantic layer: one cheap agent pass after each finished search
        # distills typed claims into the card (claims.py). Routed via
        # `routing: distill:` (default model: haiku)
        "claims": True,
        # inject graph-ranked claims into operator prompts when
        # knowledge/graph.json exists (its own setting so A/B stays possible)
        "graph_retrieval": True,
        # credit assignment: injected claims share the search's outcome reward;
        # track records adjust retrieval confidence and retire failing claims
        "credit": True,
        # consolidated playbooks (knowledge/playbooks/<concept>.md) replace the
        # raw claims block in draft prompts when one matches the problem's
        # concepts; credit flows to the playbook's source claims
        "playbooks": True,
        # skill library: harvest scored winners into knowledge/skills/ and hand
        # the best match to the first draft as reference_solution.py
        "skills": True,
        # the module that indexes the knowledge as a graph: `knowledge-graph`
        # (the built-in), a .py file, or module:Class
        "graph": DEFAULT_GRAPH,
    }

    def agent_passes(self) -> tuple[str, ...]:
        return ("distill",) if bool(self.param("claims")) else ()

    def knowledge_dir(self) -> Path | None:
        return knowledge_dir_for(self.env.config)

    def graph_module(self) -> GraphModule:
        """The graph module `graph` names: the built-in by registry name, a
        local file (so it travels with the snapshot), or an importable class
        — keyed so that an edited file is a different builder and graph.json
        is rebuilt."""
        from hillclimb.modules.memory.graphs import graph_key

        ref = str(self.param("graph"))
        resolved = refs.resolve_ref(ref, "graph", scope=self.scope)
        module = resolved.target()
        if resolved.path is not None:
            # the file's NAME, not where it is: a snapshot's copy is the same builder
            attr = refs.split_file_ref(ref)[1]
            module.key = graph_key(resolved.path.name + (f":{attr}" if attr else ""), resolved.path)
        else:
            module.key = graph_key(ref)
        if not module.name:
            module.name = resolved.label  # type: ignore[misc]
        return module

    # --- before the search ---

    def retrieve(self, *, context: str | None = None) -> Retrieved:
        env = self.env
        text, offset = context, 0
        if context is None:
            text, offset, claim_ids = self.prior_experience()
            if claim_ids:
                from hillclimb.modules.memory.credit import record_injected_claims

                record_injected_claims(env.search_dir, claim_ids)
        reference, note = self._reference_solution()
        return Retrieved(
            text=text or "",
            reference=reference,
            reference_note=note,
            priors={"complexity_start": offset} if offset else {},
        )

    def prior_experience(self) -> tuple[str | None, int, list[str]]:
        """(prior-experience prompt section, draft-complexity offset, injected
        claim ids) from the knowledge cards. The claim ids feed credit
        assignment: whoever gets quoted in the prompt answers for the outcome.
        The claims come through this memory's graph module, so a climber that
        indexes memory differently is shown what it asked for."""
        from hillclimb.modules.memory.knowledge import (
            complexity_offset,
            load_cards,
            problem_family,
            render_prior_experience,
        )

        env, log, problem = self.env, self.env.log, self.env.problem
        knowledge_dir = self.knowledge_dir()
        if knowledge_dir is None:
            return None, 0, []
        family = problem_family(problem.problem_id, env.target)
        cards = load_cards(knowledge_dir, problem_id=problem.problem_id, family=family)
        if not cards:
            return None, 0, []
        log(f"learning: {len(cards)} prior search card(s) inform this search")
        offset = complexity_offset(cards) if self.param("complexity_prior") else 0
        text = render_prior_experience(cards, max_cards=int(self.param("max_cards")))
        claim_ids: list[str] = []
        if self.param("graph_retrieval"):
            # graph-walk retrieval: distilled claims for this family plus
            # cross-family claims that share a concept with the problem. When a
            # consolidated playbook covers the problem's concepts it REPLACES the
            # raw claim list (evolved prose beats retrieved facts), and credit
            # flows to the claims the playbook was built from.
            # Best effort — the cards block above never depends on the graph.
            try:
                from hillclimb.modules.memory.claims import problem_concepts, render_claims
                from hillclimb.modules.memory.graph import load_or_build_graph, node_to_claim

                kind = problem.runtime
                concepts = problem_concepts(kind, problem.metric_name)
                playbooks = []
                if self.param("playbooks"):
                    from hillclimb.modules.memory.consolidate import load_playbooks, render_playbooks

                    playbooks = load_playbooks(knowledge_dir, concepts)
                if playbooks:
                    log(
                        "learning: playbook(s) inform this search: "
                        + ", ".join(p.concept for p in playbooks)
                    )
                    text = f"{text}\n\n{render_playbooks(playbooks)}"
                    claim_ids = sorted({cid for p in playbooks for cid in p.source_claims})
                else:
                    module = self.graph_module()
                    nodes = module.retrieve(
                        load_or_build_graph(knowledge_dir, module=module),
                        family=family,
                        problem_id=problem.problem_id,
                        concepts=concepts,
                    )
                    claims_text = render_claims([node_to_claim(n) for n in nodes])
                    if claims_text:
                        log(f"learning: {len(nodes)} distilled claim(s) inform this search")
                        text = f"{text}\n\n{claims_text}"
                        claim_ids = [n.id.removeprefix("claim:") for n in nodes]
            except Exception as exc:  # noqa: BLE001
                log(f"learning: graph retrieval failed (prior cards unaffected): {exc}")
        return text, offset, claim_ids

    def _reference_solution(self) -> tuple[Path | None, str]:
        """A proven solution from the skill library, for the first draft."""
        env, log, problem = self.env, self.env.log, self.env.problem
        if not self.param("skills"):
            return None, ""
        knowledge_dir = self.knowledge_dir()
        if knowledge_dir is None:
            return None, ""
        try:
            from hillclimb.modules.memory.claims import problem_concepts
            from hillclimb.modules.memory.knowledge import problem_family
            from hillclimb.modules.memory.skills import SKILL_CODE_FILENAME, select_skill

            kind = problem.runtime
            match = select_skill(
                knowledge_dir,
                family=problem_family(problem.problem_id, env.target),
                concepts=problem_concepts(kind, problem.metric_name),
                higher_is_better=problem.higher_is_better,
            )
            if match is not None:
                skill, skill_dir = match
                score = f"{skill.score:g} {skill.metric}" if skill.score is not None else "unscored"
                note = f"scored {score} on {skill.problem_id}"
                log(f"learning: reference solution from {skill.run_ref} ({note})")
                return skill_dir / SKILL_CODE_FILENAME, note
        except Exception as exc:  # noqa: BLE001
            log(f"learning: skill selection failed (draft unaffected): {exc}")
        return None, ""

    # --- during the search: live sharing between the searches of one run ---

    def _live_run_dir(self) -> Path | None:
        """The run dir hosting the shared live-card folder; None when live
        sharing is off or the search dir is not in the runs/<run-id>/searches/
        layout (embedded and unit-test constructions)."""
        from hillclimb.harness.run import SEARCHES_DIRNAME

        env = self.env
        if env is None or not (env.config.learning.enabled and self.param("live")):
            return None
        if env.search_dir.parent.name != SEARCHES_DIRNAME:
            return None
        return env.search_dir.parents[1]

    def _family_target(self) -> str:
        """Problem target string for family grouping (same convention as the
        knowledge backfill: empty for non-emflow problems)."""
        problem = self.env.problem
        if problem.runtime == "emflow":
            return f"emflow://{problem.emflow_problem}"
        return ""

    def live(self) -> str:
        """Prompt section from sibling searches' live cards; "" when there are
        none. Polled fresh on every prompt build so late discoveries land in
        the very next operator."""
        run_dir = self._live_run_dir()
        if run_dir is None:
            return ""
        from hillclimb.modules.memory.knowledge import load_live_cards, problem_family, render_live_experience

        env = self.env
        try:
            cards = load_live_cards(
                run_dir,
                exclude_search_id=env.search_dir.name,
                family=problem_family(env.problem.problem_id, self._family_target()),
            )
            return render_live_experience(cards, max_cards=int(self.param("max_cards")))
        except Exception:  # noqa: BLE001
            return ""

    def publish(self, journal, *, budget_s: int, cost_usd: float) -> None:
        """Republish this search's knowledge card into the run-scoped live dir
        so concurrent sibling searches see discoveries mid-run. Best effort —
        sharing must never fail a search."""
        run_dir = self._live_run_dir()
        if run_dir is None:
            return
        from hillclimb.modules.memory.knowledge import distill_card, write_live_card

        env = self.env
        try:
            card = distill_card(
                journal,
                problem=env.problem,
                run_ref=f"{run_dir.name}/{env.search_dir.name}",
                target=self._family_target(),
                budget_s=budget_s,
                cost_usd=cost_usd,
                selection=env.config.holdout.selection,
            )
            write_live_card(run_dir, card, env.search_dir.name)
        except Exception as exc:  # noqa: BLE001
            env.log(f"  live card publish failed (search unaffected): {exc}")

    # --- after the search ---

    def record(self, journal, *, budget_s: int, cost_usd: float) -> None:
        """Best effort — learning must never fail a finished search."""
        from hillclimb.harness.run import SEARCHES_DIRNAME
        from hillclimb.modules.memory.knowledge import write_card, write_live_card

        env = self.env
        config, problem, search_dir, log = env.config, env.problem, env.search_dir, env.log
        try:
            card = _search_card(env, journal, budget_s=budget_s, cost_usd=cost_usd)
            knowledge_dir = self.knowledge_dir()
            if self.param("claims") and knowledge_dir is not None:
                # inner guard: a failed distill pass costs the claims, not the card
                try:
                    from hillclimb.modules.memory.claims import distill_claims

                    card.claims = distill_claims(
                        journal,
                        problem=problem,
                        card=card,
                        search_dir=search_dir,
                        knowledge_dir=knowledge_dir,
                        config=config,
                        log=log,
                        record_cost=True,  # this engine is the journal's writer
                    )
                    if card.claims:
                        log(f"learning: {len(card.claims)} claim(s) distilled")
                except Exception as exc:  # noqa: BLE001
                    log(f"learning: claims distillation failed (card unaffected): {exc}")
            # always keep a copy with the search artifacts (synced for hosted runs)
            _keep_with_the_search(env, card)
            if knowledge_dir is not None:
                path = write_card(knowledge_dir, card)
                log(f"learning: knowledge card written to {path}")
            if (
                config.learning.enabled
                and self.param("live")
                and search_dir.parent.name == SEARCHES_DIRNAME
            ):
                # final refresh of the run-scoped live card: siblings still
                # running loaded their static prior cards before this search
                # finished, so the live channel is how its result reaches them
                write_live_card(search_dir.parents[1], card, search_dir.name)
            if (
                self.param("credit")
                and self.param("graph_retrieval")
                and knowledge_dir is not None
            ):
                # credit assignment: the claims this search's drafts were shown
                # share its outcome (see credit.py for the reward definition)
                try:
                    from hillclimb.modules.memory.credit import (
                        CreditEvent,
                        read_injected_claims,
                        search_reward,
                        write_credit_event,
                    )
                    from hillclimb.modules.memory.knowledge import load_cards

                    claim_ids = read_injected_claims(search_dir)
                    if claim_ids:
                        prior_cards = [
                            c for c in load_cards(
                                knowledge_dir,
                                problem_id=card.problem_id,
                                family=card.family,
                            )
                            if c.run_ref != card.run_ref  # own card is already on disk
                        ]
                        reward, basis = search_reward(
                            journal, problem, prior_cards, selection=config.holdout.selection
                        )
                        write_credit_event(knowledge_dir, CreditEvent(
                            run_ref=card.run_ref,
                            problem_id=card.problem_id,
                            family=card.family,
                            claim_ids=claim_ids,
                            reward=reward,
                            basis=basis,
                        ))
                        log(
                            f"learning: credit {reward:g} ({basis}) recorded for "
                            f"{len(claim_ids)} injected claim(s)"
                        )
                except Exception as exc:  # noqa: BLE001
                    log(f"learning: credit assignment failed (card unaffected): {exc}")
            if self.param("skills") and knowledge_dir is not None:
                # procedural memory: a scored winner joins the skill library
                try:
                    from hillclimb.modules.memory.skills import harvest_skill

                    skill_dir = harvest_skill(
                        journal, problem=problem, card=card,
                        knowledge_dir=knowledge_dir,
                        selection=config.holdout.selection, log=log,
                    )
                    if skill_dir is not None:
                        log(f"learning: skill harvested -> {skill_dir}")
                except Exception as exc:  # noqa: BLE001
                    log(f"learning: skill harvest failed (card unaffected): {exc}")
            if knowledge_dir is not None:
                # keep the derived graph index fresh; cheap at this scale and
                # best-effort like everything else here
                try:
                    from hillclimb.modules.memory.graph import rebuild_graph

                    rebuild_graph(knowledge_dir, module=self.graph_module())
                except Exception as exc:  # noqa: BLE001
                    log(f"learning: graph rebuild failed (card unaffected): {exc}")
        except Exception as exc:  # noqa: BLE001
            log(f"learning: card distillation failed (search result unaffected): {exc}")


refs.register(KIND, "files", FilesMemory)
refs.register(KIND, "none", NoMemory)


def get_memory(
    name: str,
    params: Mapping | None = None,
    *,
    base_dir: Path | None = None,
    scope: refs.FileScope | None = None,
) -> Memory:
    """A memory instance, built with `params`. A setting it does not have is
    a `ClimberLoadError` naming the memory."""
    resolved = refs.resolve_ref(name, KIND, base_dir=base_dir, scope=scope)
    try:
        memory = refs.construct(resolved.target, {"params": dict(params or {})}, name)
    except ValueError as exc:
        raise refs.ClimberLoadError(f"memory: {name}: {exc}") from exc
    if not memory.name:
        memory.name = resolved.label  # type: ignore[misc]
    memory.scope = scope
    return memory
