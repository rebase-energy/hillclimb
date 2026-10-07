# Commands

Every `hillclimb` command, the live TUIs, and the local demo suite.

## Live TUIs

Open them in a second terminal while searches climb:

```bash
hillclimb watch candidates   # one search's candidates: drafting, debugging, improving
hillclimb watch              # all the searches side by side
hillclimb chart              # the hillclimb: best score so far across every search, every candidate a dot
                             # (several charts in the folder: a table first — enter opens one, esc or b comes back)
hillclimb graph              # the knowledge graph growing as searches finish
hillclimb tree               # one search's exploration tree: expanded vs discontinued lineages
hillclimb chart --detail     # the curve with that tree drawn on it (every scored candidate, parent edges)
```

`hillclimb watch` is the one to keep open beside your agent: runs → searches →
candidate trees, with an on-demand candidate detail panel for notes, scores,
lineage, output, and the timestamped operator stream when present. A
candidate still in flight gets a live console under its overview: the
coding agent's stream, then the verifier's stdout/stderr, appended as they are
written (`tail -f` style — it follows the end until you scroll up, and `f`
follows again). Drag the divider, use `+` / `-` to resize the detail
panel, or `m` to maximize it.

## Commands

Search-addressing commands take `<run-id>/<search-id>`, a bare `<run-id>` (when
the run has a single search), or `latest` (the default).

| command | what it does |
|---|---|
| `problem list` | the bundled example problems, with the best known value and who found it |
| `problem get <problem>` | copy a bundled problem into `problems/` (makes the current folder a hillclimb dir if there is none) and list its files (`fetch` is a deprecated alias) |
| `problem new <problem>` | start a problem of your own in `problems/<problem>/`: a small two-step problem that already runs (`run.py` calls the solution, `score.py` scores it) to edit into yours |
| `init [dir] [--datastore files\|sqlite]` | make the current folder (or `dir`) a hillclimb dir: `hillclimb.yaml`, an empty `problems/` and `runs/`, and the gitignore rules that commit each run's record but not its bulk; refuses a folder with its own `problems/` or `runs/` (`init hillclimb` keeps it in a subfolder); `--datastore sqlite` keeps the records of runs in one `store.sqlite` (candidates and `best/` stay in `runs/`) |
| `connect` | which coding agents this machine can run operators with, and who pays — each credential read through the same environment an operator gets |
| `connect <claude\|codex\|pi\|openrouter> [--auth ...] [--model ...] [--no-probe] [--default] [--local]` | run that coding agent's login, stage the credentials searches read, ping the route with one tool-free call, pin `agent`/`agent_auth` in `~/.config/hillclimb/config.yaml` (every folder; `--local` pins this folder's config.yaml as its override) |
| `verify <problem> [--repeat N] [--holdout] [--solution FILE]` | run a problem's verifier once, outside a search; `--repeat` measures the noise floor |
| `run <target> [--name ...] [--budget 2h] [--agent ...] [--model ...] [--climber REF] [--no-detach]` | start a run for one problem or a suite YAML — in the background, `hillclimb watch` follows it; `--no-detach` keeps it in the terminal |
| `run <problem> --parallel-searches N --parallel-agents M` | N independent searches (each in the background, one run) each running M coding agents at once |
| `run <problem> --parallel-replicates P` | of a trial's `--n-replicates` seeded runs, how many execute at once: 0 = all (default), 1 = one after another, required when the metric measures the machine |
| `run <problem> --solution-cpus C` | each run of a solution may use C cores, given to it as `$HILLCLIMB_CPUS` (default 1); see [Parallelism](concepts.md#parallelism) |
| `run <problem> --climber A --climber B [--experiment-set EXPERIMENT:KEY=VALUE]` | a mixed fleet: one search per climber, each tagged as an experiment |
| `resume [search] [--all] [--no-detach]` | continue a parked / stopped / crashed search on its remaining budget — detached, like `run` (`--no-detach` keeps it in the terminal); a dead coding agent login is offered a fresh one first |
| `status [search]` | search state + candidate tree (text) |
| `watch` | live TUI over runs, searches, and candidates |
| `chart` | live chart, one per problem: best score so far by tested-candidate count — one staircase across every run of the problem, every scored candidate a dot. With several runs `v` cycles the plain climb (default) → the climb with each run's dots in its own colour, named in the legend (`#2 15:08`) → a comparison, one line per run from its first candidate; `]`/`[` focus one run (the others fade); `d` overlays the focused run's tree; a study gets one line per experiment. A bare `chart` lists the folder's problems (and studies) first when there is more than one |
| `similarity map [search] [--single] [--metric M]` | (`v` from the default view) live 3D map: every candidate embedded by pairwise distance (behavioral by default; `m` cycles structural and blend), so nearby dots are alike — lineage edges, a gold best-so-far trail, hover reads distances, click dims everything outside a lineage, `space` replays the search growing; a study's experiment opens its whole run, coloured by experiment; a bare `similarity` is this view |
| `similarity [search]` = `similarity reference [search] [--single]` | the default view, the figure on hillclimb.sh, its three axes named in the plot (`behaviour →`, `code →`, `lineage →`) — live 3D cube: each candidate at behavioral / structural / lineage distance from the search's seed (or baseline; `c` toggles the champion); a study's experiment opens its whole run, coloured by experiment, `n`/`p` stepping through the run's problems; a problem's `fingerprint.py` defines the behavioral axis; `v` swaps between the two views |
| `graph` | the knowledge-graph TUI (same screen as `knowledge graph`) |
| `show [search] <candidate-id>` | everything about one candidate: scores, evaluation breakdown, diff vs parent, output |
| `plot [target] [candidate] [--problem P] [--out PNG] [--no-open]` | draw a solution with its problem's `plot.py` (matplotlib, run in the problem's runtime venv like its verifier) and open the PNG: with no argument the solution in this folder (what `summit` wrote), saved beside it as `solution.png`; else a folder of solution files, a search's best, or one of its candidates (saved under the machine cache); a problem without `plot.py` says so |
| `ps [--watch] [-n SECONDS]` | every process hillclimb owns on this machine — compute only, boxed like nvidia-smi and sized to the terminal: the machine at a glance (searches, coding agents against the slot cap, cpu, memory) above one process table where each engine heads its own tree (its row names the problem and hillclimb dir) — and so does a hillclimb command computing in a terminal (`verify`, `grade`, `run --no-detach`) — each process labelled by role (`agent`, `tool` = a command the coding agent runs itself, `mcp` = a server from your own Claude config, `verifier`, `solution` = what the verifier scores) with its command shortened to what tells it apart; columns that do not fit are dropped, nothing wraps; `orphan` marks searches whose hillclimb dir was deleted; `--watch` (`-w`) redraws it in place every second (`-n` sets the interval) until ctrl+c, folding MCP servers and sharing rows between searches so the frame stays on one screen. Search progress is `watch`'s |
| `top` | the control pane for every hillclimb process on this machine, from any folder: `ps`, live, in one table where each search heads its own process tree; keys act on the highlighted row — `s` stops its engine through the command queue (resumable), `g` gracefully, `k` kills that process and what it started (on a search row the whole search), `o`/`r` change the order of the engines. Search progress is `watch`'s |
| `stop [search] [--all] [--graceful]` | stop now: abort the operators in flight (their candidates are journaled abandoned), then park, resumable; `--graceful` lets them finish and be scored first; `--all` also reaps orphaned engines when no hillclimb dir is found |
| `kill [search] [--all] [--grace S]` | last resort for a search that ignores `stop`: SIGTERM it with its coding agents and verifiers, SIGKILL after `S` s (default 5); still resumable |
| `reset` | stop every search of THIS hillclimb dir and delete the dir — never another folder's; `--runs` deletes only what searches produced (`runs/`, the store, `knowledge/`) and keeps `hillclimb.yaml` and `problems/` |
| `prune <search> <candidate-id> [--reason ...]` | cut a candidate and its subtree from the search |
| `tree [search]` | live tree of one search, drawn like the Darwin Gödel Machine's archive: the candidate number inside each circle, fill = score (cyan ramp, bright = best; hollow = no working solution), ring = what the search did with it (white expanded — the spine the policy walked / none a scored, scored and left / red failed), star = best, bold white path = the best's lineage; circles are sized to the zoom so they never overlap, numbers appear as they grow; the legend toggles each stage, the best and the lineage (`1`-`5` or click) |
| `treeclimb [search]` | the `tree` tree on the left and the climb chart on the right — every scored candidate at (candidate number, score), the best-so-far staircase, and the lineage of the final best as a thick line, the same parent chain drawn bold in the tree; `j`/`k` scrub both panels together, click a node to ring its dot on the chart; the chart's legend sits in the corner the climb leaves empty and toggles its series (`6`-`9` or click), the hover readout keeps off it |
| `surface [search]` | live 3D fitness surface: the search's candidates on the problem's terrain (needs a `landscape.py` in the problem; `problems/fitness-landscape/` is the reference) |
| `summit [problem] [--to DIR] [--plot]` | copy the best solution found so far across every run of a problem into your hillclimb dir (its `solution.py`, `params.json` and output files); works mid-climb; `--plot` then draws it with the problem's `plot.py` into `solution.png` |
| `disconnect <claude\|codex\|pi\|openrouter> [--local]` | undo a connect on hillclimb's side: comment the pin out of the same config file, remove the staged homes under `~/.cache/hillclimb/`, drop an OpenRouter key from the `.env`; the coding agent's own login is never touched |
| `smoke [problem]` | one real coding agent call end-to-end (auth / contract check) |
| `climber list` | the climbers `run --climber` accepts: the presets and every folder or `.py` file under `climbers/` |
| `climber get [PRESET] [--name NAME] [--no-default]` | copy a preset out as `climbers/<name>/` — `climber.yaml`, `policy.py`, `prompts/` with the templates its operators render and a README of what fills them — and make it this folder's climber |
| `climber show [NAME]` | print a climber (a preset, a `.py` file, a climber folder, this folder's) as the block a run config takes |
| `climber new <name> [--from NAME]` | copy a policy's source into `climbers/<name>.py` and print the block that runs it |
| `climber check [--climber REF] [--set k=v] [--problem P] [--smoke] [--json]` | conformance check for a climber over the store's recorded journals; `--smoke` adds a dummy-coding-agent search |
| `store searches [--problem KEY]` / `store sync` | list what the record store holds / import the folder's searches into the configured store |
| `knowledge graph [--stats]` | interactive knowledge-graph TUI (or a text summary) |
| `knowledge rebuild` | force-rebuild the derived `knowledge/graph.json` index |
| `knowledge distill [search] [--backfill]` | run the LLM claims pass on a search / all cards |
| `knowledge backfill` | distill cards from every finished search that lacks one |
| `knowledge live [run]` | the live cards concurrent searches in a run are sharing |
| `paper add <pdf> [--problem <target>]` / `paper list` | distill a PDF into knowledge claims that seed future searches (one coding agent pass per paper, content-hash cached) |
| `knowledge consolidate [--dry-run]` | sleep phase: generalize claims + rewrite playbooks |
| `knowledge query "<terms>" [--json]` | read-only memory lookup (also available to coding agents) |
| `knowledge show <target>` | the prior-experience section a new search would get |
| `run <problem> --set key=value … [--study S --experiment E]` | any config setting, dotted; tag the search as one experiment of a study |
| `experiment run <spec> [--repeats N] [--budget B] [--parallel] [--max-concurrent N] [--run-id R --first-repeat K] [--dry-run]` | every experiment × problem × repeat of a study spec; `--max-concurrent` bounds how many run at once, `--run-id` appends repeats to a finished run |
| `experiment report [spec] [--problem X] [--control A] [--noise-floor F] [--json]` | compare a study's experiments on holdout; `--json` gives a meta-verifier the gaps and verdicts as data |

Exit code `2` from `run`/`resume` means the search parked or was stopped — resume it.

`hillclimb stop --all` ends a parallel run (the best solutions stay in `runs/`);
`hillclimb reset` ends it AND deletes what hillclimb made in this folder (`hillclimb.yaml`, `problems/`, `runs/`, …; nothing else) — only
searches pinned to that dir are killed, never another folder's. 3 searches x 3
coding agents is 9 coding agents, capped machine-wide by
`concurrency.machine_max_agents`; each search's log is under
`runs/<run-id>/logs/`.

## Local optimization demo suite

These problems are small, local, and require no download, so they are good for
exercising parallel searches in `hillclimb watch`:

| problem | objective |
|---|---|
| `circle-packing` | maximize total radius for 26 circles in a unit square |
| `heilbronn-11` | maximize the smallest triangle area among 11 points |
| `tsp-200` | minimize a 200-city Euclidean TSP tour |
| `labs-60` | minimize length-60 binary autocorrelation energy |

Start the suite, then open the TUI:

```bash
uv run hillclimb run problems/demo-suite.yaml --name "Optimization demo" --budget 10m
uv run hillclimb watch
```

