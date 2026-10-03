"""What an operator knows beyond its prompt: skills and standing instructions.

Two layers, written once in an agent-neutral form and rendered for whichever
coding agent runs:

  global  ~/.config/hillclimb/agent/   every hillclimb dir on this machine
  local   <hillclimb dir>/agent/       one problem family (committed with it)

Each layer may hold `AGENTS.md` (standing instructions) and `skills/<name>/`
(a folder with a `SKILL.md`, the format Claude Code and codex both read). A
local skill overrides a global one of the same name; instructions are joined,
global first. `agent_context:` in hillclimb.yaml can leave the global layer
out (`include_global: false`) or skip skills by name.

Before every operator call the layers are written into the candidate's
directory, where the coding agent looks for a project's own: Claude Code reads
`.claude/skills/` and `CLAUDE.md`, codex and pi `.agents/skills/` and
`AGENTS.md`. After the call, a skill the agent (or a plugin it runs) created
there is added to the local layer, so the next search of the family starts
with it; `hillclimb skills` lists and deletes them.
"""

from __future__ import annotations

import hashlib
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

AGENT_DIR = "agent"  # the layer's folder, in the hillclimb dir and in the user config dir
INSTRUCTIONS = "AGENTS.md"
SKILL_FILE = "SKILL.md"
# hillclimb's record of a skill, beside its SKILL.md: where it came from. Not
# in the SKILL.md front matter, which coding agents validate (codex rejects
# keys it does not know). No record = written by a person
RECORD_FILE = "skill.yaml"

# where each coding agent looks for a project's skills and instructions
LAYOUT = {
    "claude-code": (".claude/skills", "CLAUDE.md"),
    "codex": (".agents/skills", "AGENTS.md"),
    "pi": (".agents/skills", "AGENTS.md"),
}
# every place a coding agent may leave a skill it wrote in its working dir
SKILL_DIRS = tuple(sorted({skills for skills, _ in LAYOUT.values()}))

_harvest_lock = threading.Lock()


@dataclass(frozen=True)
class Skill:
    name: str
    path: Path  # the skill's folder
    layer: str  # "global" | "local"

    @property
    def record(self) -> dict:
        """Its skill.yaml, or {} for a skill a person wrote."""
        try:
            data = yaml.safe_load((self.path / RECORD_FILE).read_text())
        except (OSError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

    @property
    def sha256(self) -> str:
        return skill_sha256(self.path)

    @property
    def origin(self) -> str:
        """`manual` (a person wrote or kept it), `generated` (an operator or
        a plugin wrote it), or `generated, edited` (changed since)."""
        record = self.record
        if record.get("origin") != "generated":
            return "manual"
        return "generated" if record.get("sha256") in (None, self.sha256) else "generated, edited"

    @property
    def description(self) -> str:
        """The `description:` of its SKILL.md front matter (first line)."""
        try:
            text = (self.path / SKILL_FILE).read_text(errors="replace")
        except OSError:
            return ""
        if text.startswith("---"):
            for line in text.split("---", 2)[1].splitlines():
                if line.strip().startswith("description:"):
                    return line.split(":", 1)[1].strip().strip("\"'")
        return ""


def skill_sha256(folder: Path) -> str:
    """The skill's content: its SKILL.md (what every agent reads first)."""
    try:
        return hashlib.sha256((Path(folder) / SKILL_FILE).read_bytes()).hexdigest()
    except OSError:
        return ""


def global_dir() -> Path:
    from hillclimb.project import user_config_path

    return user_config_path().parent / AGENT_DIR


def local_dir(config) -> Path | None:
    hillclimb_dir = getattr(config, "hillclimb_dir", None)
    return Path(hillclimb_dir) / AGENT_DIR if hillclimb_dir is not None else None


def layers(config) -> list[tuple[str, Path]]:
    """The layers this config uses, global first."""
    settings = getattr(config, "agent_context", None)
    found = []
    if settings is None or settings.include_global:
        found.append(("global", global_dir()))
    local = local_dir(config)
    if local is not None:
        found.append(("local", local))
    return found


def skills(config) -> dict[str, Skill]:
    """Every skill the layers hold, by name: local over global, minus skipped."""
    settings = getattr(config, "agent_context", None)
    skipped = set(settings.skip) if settings is not None else set()
    found: dict[str, Skill] = {}
    for layer, root in layers(config):
        folder = root / "skills"
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_dir() and (path / SKILL_FILE).is_file() and path.name not in skipped:
                found[path.name] = Skill(path.name, path, layer)
    return found


def instructions(config) -> str:
    """The layers' standing instructions, global first."""
    parts = []
    for _, root in layers(config):
        path = root / INSTRUCTIONS
        if path.is_file():
            text = path.read_text(errors="replace").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def render(config, agent: str, candidate_dir: Path) -> dict:
    """Write the layers into `candidate_dir` the way `agent` reads a project's
    own, and return what was given, for the candidate's journal entry:
    `{"skills": [{name, layer, origin, sha256}], "instructions_sha256"}`.
    An agent with no layout (dummy, toy) gets nothing and {}. Copies, not
    links: what the agent changes there never reaches the layers behind
    hillclimb's back."""
    layout = LAYOUT.get(agent)
    if layout is None:
        return {}
    skills_dir, instructions_file = layout
    chosen = skills(config)
    if chosen:
        target = Path(candidate_dir) / skills_dir
        target.mkdir(parents=True, exist_ok=True)
        for skill in chosen.values():
            shutil.copytree(
                skill.path, target / skill.name, dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(RECORD_FILE),  # hillclimb's, not the agent's
            )
    text = instructions(config)
    if text:
        (Path(candidate_dir) / instructions_file).write_text(text + "\n")
    return {
        "skills": [
            {"name": s.name, "layer": s.layer, "origin": s.origin, "sha256": s.sha256}
            for s in sorted(chosen.values(), key=lambda skill: skill.name)
        ],
        "instructions_sha256": hashlib.sha256(text.encode()).hexdigest() if text else None,
    }


def harvest(
    config, candidate_dir: Path, *, homes: tuple[Path, ...] = (), created_by: dict | None = None, log=None
) -> list[str]:
    """Add the skills a call created to the local layer: any skill folder in
    the candidate's skill dirs (or in an operator home's `skills/`, where a
    plugin may write) that the layers do not hold yet. One that exists
    already is left as it is. Each gets a skill.yaml saying it was generated,
    by which call (`created_by`: run, search, candidate, agent, model) and
    what it said then. Returns the names added."""
    local = local_dir(config)
    if local is None:
        return []
    known = set(skills(config))
    sources = [Path(candidate_dir) / folder for folder in SKILL_DIRS]
    sources += [Path(home) / "skills" for home in homes]
    added = []
    with _harvest_lock:
        for source in sources:
            if not source.is_dir():
                continue
            for path in sorted(source.iterdir()):
                name = path.name
                if name.startswith(".") or not path.is_dir() or not (path / SKILL_FILE).is_file():
                    continue
                if name in known or name in added:
                    continue
                destination = local / "skills" / name
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(path, destination, ignore=shutil.ignore_patterns(RECORD_FILE))
                record = {
                    "origin": "generated",
                    "created_by": {**(created_by or {}), "found_in": _found_in(source, candidate_dir)},
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": skill_sha256(destination),
                }
                (destination / RECORD_FILE).write_text(yaml.safe_dump(record, sort_keys=False))
                added.append(name)
                if source.parent in [Path(home) for home in homes]:
                    shutil.rmtree(path, ignore_errors=True)  # moved: the next family starts clean
    if added and log is not None:
        log(f"  skills added to {local / 'skills'}: {', '.join(added)}")
    return added


def _found_in(source: Path, candidate_dir: Path) -> str:
    """Where the skill was written: a candidate's skill dir, or an operator home."""
    try:
        return str(Path(source).relative_to(candidate_dir))
    except ValueError:
        return "operator home"


def keep(skill: Skill) -> None:
    """Mark a generated skill as kept by a person: it is `manual` from now on,
    and its record says when and what it was generated as."""
    record = skill.record
    record.update(origin="manual", kept_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (skill.path / RECORD_FILE).write_text(yaml.safe_dump(record, sort_keys=False))
