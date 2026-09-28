"""Rule-based filter parsing. Fast, deterministic and easy to test.

The text box and the dropdowns produce the same structured ``Filters``; free text
never reaches the database, because every ID is validated against the catalogue
built at ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import cast

from rapidfuzz import fuzz, process

from pitchscanner.filters.catalogue import Catalogue, Entity, load_alias_file, normalise
from pitchscanner.models import Chip, Filters, GameState, ParsedFilters, Role, Stage

MAX_NGRAM = 4
#: A one-letter slip in a seven-letter name ("Marocco") scores about 86 on
#: `fuzz.ratio`, so the cutoff sits just below that. Lower than this and short
#: club names start colliding; the phrase-set test guards both directions.
DEFAULT_THRESHOLD = 84
#: Words that carry grammar, not meaning. Never matched as entities, never
#: reported as "didn't understand".
STOPWORDS = frozenset(
    {
        "a", "an", "the", "in", "at", "on", "of", "for", "from", "to", "with", "when", "while",
        "during", "only", "just", "show", "me", "situations", "plays", "moments", "where", "and",
        "or", "is", "are", "was", "were", "their", "there", "all", "any", "team", "teams",
        "player", "players", "side", "game", "games", "match", "matches", "involving", "against",
        "facing", "vs", "versus", "attack", "attacks", "attacking", "defending", "possession",
        "ball", "rounds", "round", "stage", "stages", "minute", "minutes", "half", "last", "first",
        "second",
    }
)
_SEASON_RE = re.compile(r"\b(20\d{2})\s*[/\-]\s*(\d{2,4})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MINUTE_AFTER_RE = re.compile(r"\bafter (?:the )?(\d{1,3})(?:st|nd|rd|th)? ?(?:minute|min)?\b")
_MINUTE_BEFORE_RE = re.compile(r"\bbefore (?:the )?(\d{1,3})(?:st|nd|rd|th)? ?(?:minute|min)?\b")
_MINUTE_RANGE_RE = re.compile(r"\b(\d{1,3})\s*(?:-|to|until)\s*(\d{1,3})\s*(?:minute|min)?\b")
_SEMI_OR_QUARTER_RE = re.compile(r"\b(semi|quarter|last 16|round of 16)\b")


@dataclass(slots=True)
class _Match:
    start: int
    end: int
    entity: Entity
    score: float


class FilterParser:
    """Primary parser. ``parse`` never raises and never invents an ID."""

    def __init__(self, catalogue: Catalogue, *, threshold: int = DEFAULT_THRESHOLD) -> None:
        self.catalogue = catalogue
        self.threshold = threshold
        self.alias_file = load_alias_file()
        self._choices = list(catalogue.index.keys())

    # ---------------------------------------------------------------- entities
    def _phrase_matches(self, tokens: list[str]) -> list[_Match]:
        """Longest-first n-gram matching, so "manchester city" beats "city"."""

        found: list[_Match] = []
        used = [False] * len(tokens)
        for size in range(min(MAX_NGRAM, len(tokens)), 0, -1):
            for i in range(0, len(tokens) - size + 1):
                if any(used[i : i + size]):
                    continue
                phrase = " ".join(tokens[i : i + size])
                if size == 1 and phrase in STOPWORDS:
                    continue
                entities, score = self._lookup(phrase)
                if not entities:
                    continue
                for entity in entities:
                    found.append(_Match(i, i + size, entity, score))
                for j in range(i, i + size):
                    used[j] = True
        return found

    def _lookup(self, phrase: str) -> tuple[list[Entity], float]:
        exact = self.catalogue.index.get(phrase)
        if exact:
            return list(exact), 100.0
        if len(phrase) < 4:
            # Too short to fuzzy-match safely: "as" would hit half the dataset.
            return [], 0.0
        # `ratio`, not `WRatio`: the partial matching in WRatio makes a long
        # phrase score highly against any short alias it happens to contain, so
        # "rainy games in Madrid" resolved to Real Madrid and the word "rainy"
        # was silently swallowed instead of reported as not understood.
        hit = process.extractOne(
            phrase, self._choices, scorer=fuzz.ratio, score_cutoff=self.threshold
        )
        if hit is None:
            return [], 0.0
        return list(self.catalogue.index.get(hit[0], [])), float(hit[1])

    # ----------------------------------------------------------------- phrases
    def _game_state(self, text: str) -> tuple[GameState | None, str | None]:
        groups: dict[str, list[str]] = self.alias_file.get("game_state") or {}
        for state, phrases in groups.items():
            for phrase in sorted(phrases, key=len, reverse=True):
                if normalise(str(phrase)) in text:
                    return cast(GameState, state), normalise(str(phrase))
        return None, None

    def _stages(self, text: str) -> tuple[list[Stage], list[str]]:
        groups: dict[str, list[str]] = self.alias_file.get("stage") or {}
        stages: list[Stage] = []
        consumed: list[str] = []
        for stage, phrases in groups.items():
            for phrase in sorted(phrases, key=len, reverse=True):
                needle = normalise(str(phrase))
                if needle not in text:
                    continue
                if needle == "final" and _SEMI_OR_QUARTER_RE.search(text):
                    # "semi final" is a knockout round, not the final.
                    continue
                if stage not in stages:
                    stages.append(cast(Stage, stage))
                consumed.append(needle)
                break
        return stages, consumed

    def _minutes(self, text: str) -> tuple[tuple[int, int] | None, str | None]:
        time_patterns: list[tuple[str, int, int]] = self.alias_file.get("time") or []
        for phrase, lo, hi in time_patterns:
            if normalise(str(phrase)) in text:
                return (int(lo), int(hi)), normalise(str(phrase))
        if m := _MINUTE_RANGE_RE.search(text):
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                return (lo, min(hi, 120)), m.group(0)
        if m := _MINUTE_AFTER_RE.search(text):
            return (int(m.group(1)), 120), m.group(0)
        if m := _MINUTE_BEFORE_RE.search(text):
            return (0, int(m.group(1))), m.group(0)
        return None, None

    def _seasons(self, text: str, competition_ids: list[int]) -> tuple[list[int], list[str]]:
        """Resolve "2022/23" or "euro 2024" against the season names in the data."""

        wanted: list[str] = []
        consumed: list[str] = []
        if m := _SEASON_RE.search(text):
            tail = m.group(2)[-2:]
            wanted.append(f"{m.group(1)}/{tail}")
            consumed.append(m.group(0))
        for m in _YEAR_RE.finditer(text):
            wanted.append(m.group(1))
            consumed.append(m.group(0))

        ids: list[int] = []
        for season in self.catalogue.of_kind("season"):
            if competition_ids and season.parent_id not in competition_ids:
                continue
            name = normalise(season.name)
            for want in wanted:
                matched = normalise(want) == name or normalise(want) in name
                if matched and season.entity_id not in ids:
                    ids.append(season.entity_id)
        return ids, consumed

    # -------------------------------------------------------------------- main
    def parse(self, text: str) -> ParsedFilters:
        cleaned = normalise(text or "")
        if not cleaned:
            return ParsedFilters(
                filters=Filters(), chips=[], unmatched=[], warnings=[], source="none"
            )

        chips: list[Chip] = []
        warnings: list[str] = []
        consumed_text: list[str] = []

        game_state, gs_phrase = self._game_state(cleaned)
        if gs_phrase:
            consumed_text.append(gs_phrase)
        minute_range, minute_phrase = self._minutes(cleaned)
        if minute_phrase:
            consumed_text.append(minute_phrase)

        # Stages are matched only on what the time and score phrases did not
        # already claim, so "final ten minutes" is a time window and not the
        # final of a tournament.
        residual = cleaned
        for phrase in sorted(consumed_text, key=len, reverse=True):
            residual = residual.replace(phrase, " ")
        stages, stage_phrases = self._stages(residual)
        consumed_text.extend(stage_phrases)
        for phrase in sorted(stage_phrases, key=len, reverse=True):
            residual = residual.replace(phrase, " ")
        tokens = [t for t in residual.split() if t]

        matches = self._phrase_matches(tokens)
        matched_spans = {(m.start, m.end) for m in matches}

        competition_ids: list[int] = []
        team_ids: list[int] = []
        player_ids: list[int] = []
        team_role: Role = "any"
        player_role: Role = "any"

        # Group by span: one phrase, possibly several candidate entities.
        by_span: dict[tuple[int, int], list[_Match]] = {}
        for m in matches:
            by_span.setdefault((m.start, m.end), []).append(m)

        roles: dict[str, list[str]] = self.alias_file.get("role") or {}
        defending_markers = {normalise(p) for p in roles.get("defending", [])}
        seen_team_roles: dict[int, str] = {}

        for (start, end), group in sorted(by_span.items()):
            phrase = " ".join(tokens[start:end])
            distinct = {(g.entity.kind, g.entity.entity_id): g.entity for g in group}
            if len(distinct) > 1:
                # Ambiguous on purpose: filtering on a guess would quietly change
                # what the user asked for, so say so and let them pick.
                names = sorted(e.name for e in distinct.values())
                warnings.append(
                    f"'{phrase}' could mean {' or '.join(names[:4])}; pick one to filter on."
                )
                continue
            entity = next(iter(distinct.values()))
            if entity.kind == "competition" and entity.entity_id not in competition_ids:
                competition_ids.append(entity.entity_id)
                chips.append(
                    Chip(field="competition", label="Competition", value=entity.name,
                         ids=[entity.entity_id])
                )
                if entity.frames == 0:
                    warnings.append(
                        f"No positional data for {entity.name}; "
                        "see the dropdown for what is covered."
                    )
            elif entity.kind == "team":
                before = tokens[start - 1] if start > 0 else ""
                role_here = "defending" if before in defending_markers else "attacking"
                if entity.entity_id in team_ids:
                    if seen_team_roles.get(entity.entity_id, role_here) != role_here:
                        # "Spain vs Spain": both roles at once matches nothing, so
                        # keep the first reading and say what was dropped.
                        warnings.append(
                            f"{entity.name} was asked for as both the attacking and the "
                            "defending team; keeping the first."
                        )
                    continue
                team_ids.append(entity.entity_id)
                seen_team_roles[entity.entity_id] = role_here
                if role_here == "defending":
                    team_role = "defending"
                elif team_role == "any":
                    team_role = "attacking"
                chips.append(
                    Chip(field="team", label="Team", value=entity.name,
                         ids=[entity.entity_id])
                )
                if entity.frames == 0:
                    warnings.append(f"No positional data for {entity.name}.")
            elif entity.kind == "player" and entity.entity_id not in player_ids:
                player_ids.append(entity.entity_id)
                player_role = "attacking"
                chips.append(
                    Chip(field="player", label="Player", value=entity.name,
                         ids=[entity.entity_id])
                )

        season_ids, season_phrases = self._seasons(cleaned, competition_ids)
        for sid in season_ids:
            season = self.catalogue.by_id("season", sid)
            if season is not None:
                chips.append(
                    Chip(field="season", label="Season", value=season.name, ids=[sid])
                )

        if game_state:
            chips.append(Chip(field="game_state", label="Game state", value=game_state))
        for stage in stages:
            chips.append(Chip(field="stage", label="Stage", value=stage))
        if minute_range:
            chips.append(
                Chip(field="minute_range", label="Minutes",
                     value=f"{minute_range[0]}-{minute_range[1]}")
            )

        unmatched: list[str] = []
        for i, tok in enumerate(tokens):
            if tok in STOPWORDS or any(s <= i < e for s, e in matched_spans):
                continue
            if any(tok in normalise(p) for p in season_phrases):
                continue
            if tok.isdigit():
                continue
            unmatched.append(tok)

        filters = Filters(
            competition_ids=competition_ids,
            season_ids=season_ids,
            team_ids=team_ids,
            team_role=team_role,
            player_ids=player_ids,
            player_role=player_role,
            game_state=game_state or "any",
            minute_range=minute_range,
            stages=stages,
        )
        return ParsedFilters(
            filters=self.validate(filters),
            chips=chips,
            unmatched=sorted(set(unmatched)),
            warnings=sorted(set(warnings)),
            source="rules" if chips else "none",
        )

    def validate(self, filters: Filters) -> Filters:
        """Drop anything that is not in the catalogue. The last line of defence."""

        return filters.model_copy(
            update={
                "competition_ids": [
                    i
                    for i in filters.competition_ids
                    if i in self.catalogue.valid_ids("competition")
                ],
                "season_ids": [
                    i for i in filters.season_ids if i in self.catalogue.valid_ids("season")
                ],
                "team_ids": [i for i in filters.team_ids if i in self.catalogue.valid_ids("team")],
                "player_ids": [
                    i for i in filters.player_ids if i in self.catalogue.valid_ids("player")
                ],
            }
        )


def merge(explicit: Filters, parsed: Filters) -> Filters:
    """Explicit filters win on conflicts; parsed text fills the gaps."""

    data = parsed.model_dump()
    for field_name, value in explicit.model_dump().items():
        default = getattr(Filters(), field_name)
        if value != default:
            data[field_name] = value
    return Filters(**data)
