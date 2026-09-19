"""Not a test: a generator for eyeballing `hillclimb tree2`.

Builds a synthetic archive-shaped search — like the Darwin Gödel Machine
figure: ~80 iterations, a few deep lineages, many dead ends — in a
throwaway hillclimb folder, then:

    uv run python tests/tree2_demo.py [/tmp/tree2-demo] [80]
    HILLCLIMB_DIR=/tmp/tree2-demo/hillclimb uv run hillclimb tree2

Seeded, so the same folder comes out every time. `tree` works on it too.
"""
from __future__ import annotations

import random
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for tests.factories

from hillclimb.candidate import Candidate
from hillclimb.journal import Journal
from hillclimb.run import RunMeta, SearchMeta, write_run_meta, write_search_meta
from hillclimb.status import SearchStatus, write_status
from tests.factories import trial as mk_trial

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tree2-demo")
n_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 80
rng = random.Random(7)
hc = root / "hillclimb"
hc.mkdir(parents=True, exist_ok=True)
(hc / "config.yaml").write_text("# tree2 demo hillclimb dir\n")
run_id, search_id = "20260915-demo", "archive-demo"
run_dir = hc / "runs" / run_id
search_dir = run_dir / "searches" / search_id
if run_dir.exists():
    shutil.rmtree(run_dir)
(search_dir / "candidates").mkdir(parents=True)
(search_dir / "best").mkdir()
write_run_meta(run_dir, RunMeta(run_id=run_id, name="tree2 demo", target="demo", problem_ids=[search_id]))
write_search_meta(search_dir, SearchMeta(
    search_id=search_id, run_id=run_id, problem="/tmp/problem", problem_id=search_id,
    backend="claude-code", model="sonnet", metric="score", higher_is_better=True, budget_s=3600,
))
journal = Journal(search_dir / "journal.jsonl")
t0 = datetime(2026, 9, 15, 9, 0, tzinfo=timezone.utc)

def stamp(i, offset=0):
    return (t0 + timedelta(minutes=3 * i + offset)).isoformat()

scores = {}  # id -> score of working candidates (parents are drawn from these)
def add(cid, operator, parent, score, status="passing", i=0):
    journal.candidate_result(Candidate(
        candidate_id=cid, operator=operator, parent_id=parent, status=status,
        trials=[mk_trial(val_score=score)] if score is not None else [],
        summary=f"{operator} of {parent}" if parent else operator,
        created_at=stamp(i), finished_at=stamp(i, 2),
    ))
    if score is not None:
        scores[cid] = score

add("c000", "baseline", None, 0.20, i=0)
for i in range(1, n_iter + 1):
    cid = f"c{i:03d}"
    # parent: mostly the good ones (score-weighted), sometimes anyone — an archive
    ids = list(scores)
    weights = [max(scores[c], 0.05) ** 3 for c in ids]
    parent = rng.choices(ids, weights)[0] if rng.random() < 0.8 else rng.choice(ids)
    roll = rng.random()
    if roll < 0.35:                      # no working solution
        add(cid, "improve", parent, None, status=rng.choice(["buggy", "buggy", "abandoned"]), i=i)
    else:
        base = scores[parent]
        delta = rng.gauss(0.0, 0.06) + (0.03 if roll > 0.9 else 0.0)
        add(cid, "debug" if roll < 0.45 else "improve", parent, round(min(0.95, max(0.02, base + delta)), 3), i=i)

write_status(search_dir, SearchStatus(search_id=search_id, run_id=run_id, state="done"))
best = max(scores, key=scores.get)
print(f"{n_iter + 1} candidates in {search_dir}\nbest {best} = {scores[best]}")
