"""The entity catalogue the parser is allowed to resolve against.

Built from the dataset at ingestion and enriched with the checked-in alias file.
Nothing outside this table can ever become a filter ID, which is what keeps free
text from reaching the database.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml

from pitchscanner.store.base import CatalogueEntry

KINDS = ("competition", "season", "team", "player")
#: Words that appear in so many club names that indexing them adds only noise.
_GENERIC_WORDS = frozenset(
    {"football", "club", "fc", "city", "town", "athletic", "sporting", "deportivo",
     "womens", "women", "national"}
)


_PUNCTUATION_AS_SPACE = frozenset("-.,;:!?()")


def normalise_aligned(text: str) -> tuple[str, list[int]]:
    """`normalise`, plus for each output character the index it came from in `text`.

    The filter parser uses the map to turn a phrase it found in the normalised
    text back into the exact substring the user typed.
    """

    chars: list[str] = []
    origin: list[int] = []
    for i, original in enumerate(text):
        for c in unicodedata.normalize("NFKD", original.lower()):
            if unicodedata.combining(c):
                continue
            chars.append(" " if c in _PUNCTUATION_AS_SPACE else c)
            origin.append(i)
    out: list[str] = []
    out_origin: list[int] = []
    pending_space = False
    for c, o in zip(chars, origin, strict=True):
        if c.isspace():
            pending_space = bool(out)
            continue
        if pending_space:
            out.append(" ")
            out_origin.append(o)
            pending_space = False
        out.append(c)
        out_origin.append(o)
    return "".join(out), out_origin


def normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace. One definition,
    used everywhere."""

    return normalise_aligned(text)[0]


#: Parsed YAML: nested lists and dicts of strings and ints. Typing it more
#: precisely than `Any` would only move the casts to every call site.
AliasFile = dict[str, Any]


@lru_cache(maxsize=1)
def load_alias_file() -> AliasFile:
    raw = resources.files("pitchscanner.filters").joinpath("aliases.yaml").read_text()
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise TypeError("aliases.yaml must contain a mapping at the top level")
    return loaded


@dataclass(slots=True)
class Entity:
    kind: str
    entity_id: int
    name: str
    parent_id: int | None
    frames: int
    aliases: tuple[str, ...] = ()


@dataclass(slots=True)
class Catalogue:
    entities: list[Entity] = field(default_factory=list)
    #: normalised alias -> list of entities that answer to it
    index: dict[str, list[Entity]] = field(default_factory=dict)

    @classmethod
    def from_entries(cls, entries: Iterable[CatalogueEntry]) -> Catalogue:
        alias_file = load_alias_file()
        extra_by_kind: dict[str, dict[str, list[str]]] = {
            "competition": alias_file.get("competitions") or {},
            "team": alias_file.get("teams") or {},
        }
        cat = cls()
        for e in entries:
            if e.kind not in KINDS:
                continue
            extras = extra_by_kind.get(e.kind, {})
            aliases = {normalise(e.name)}
            for key, values in extras.items():
                if normalise(str(key)) == normalise(e.name):
                    aliases.update(normalise(str(v)) for v in values)
            # Index the distinctive single words of a name too, so "Leverkusen"
            # and "Wirtz" resolve. Words shared by several entities ("United",
            # "Real") end up pointing at all of them on purpose: the parser
            # reports the ambiguity instead of guessing.
            if e.kind in ("team", "player"):
                for word in normalise(e.name).split():
                    if len(word) >= 4 and word not in _GENERIC_WORDS:
                        aliases.add(word)
            entity = Entity(
                kind=e.kind,
                entity_id=e.entity_id,
                name=e.name,
                parent_id=e.parent_id,
                frames=e.frames,
                aliases=tuple(sorted(aliases)),
            )
            cat.entities.append(entity)
            for alias in entity.aliases:
                cat.index.setdefault(alias, []).append(entity)
        return cat

    def of_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]

    def by_id(self, kind: str, entity_id: int) -> Entity | None:
        return next(
            (e for e in self.entities if e.kind == kind and e.entity_id == entity_id), None
        )

    def valid_ids(self, kind: str) -> set[int]:
        return {e.entity_id for e in self.entities if e.kind == kind}

    def keys(self) -> Sequence[str]:
        return tuple(self.index.keys())

    def covered_competitions(self) -> list[Entity]:
        return [e for e in self.of_kind("competition") if e.frames > 0]
