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

from pitchscanner.filters.catalogue import (
    Catalogue,
    Entity,
    load_alias_file,
    normalise,
    normalise_aligned,
)
from pitchscanner.models import Chip, Filters, GameState, ParsedFilters, Role, Stage, Venue

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
        "second", "defends", "attacks",
    }
)
#: Words after a team that say which side of the ball it is on.
_DEFENDING_AFTER = frozenset({"defending", "defends"})
_ATTACKING_AFTER = frozenset({"attacking", "attacks", "attack"})

_NUMBER_WORDS = {
    w: n
    for n, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    )
}
_NUMBER_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60})
_NUM = (
    r"(?:\d{1,3}|(?:twenty|thirty|forty|fifty|sixty)(?: (?:one|two|three|four|five|six|seven"
    r"|eight|nine))?|(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty))"
)
_ORD = r"(?:st|nd|rd|th)?"
#: Last N minutes counts back from the end of normal time plus a little stoppage,
#: the same convention as the "last 15 minutes" alias (75 to 98).
_LAST_N_RE = re.compile(rf"\b(?:last|final|closing) ({_NUM}) (?:minutes|mins)\b")
_FIRST_N_RE = re.compile(rf"\b(?:first|opening) ({_NUM}) (?:minutes|mins)\b")
_MINUTE_SPAN_RES = (
    re.compile(
        rf"\bbetween (?:the )?(?:minutes? )?({_NUM}){_ORD}(?: minutes?)? and (?:the )?"
        rf"({_NUM}){_ORD}(?: minutes?)?"
    ),
    re.compile(rf"\bfrom (?:the )?({_NUM}){_ORD} to (?:the )?({_NUM}){_ORD}(?: minutes?)?\b"),
    re.compile(rf"\bminutes? ({_NUM}){_ORD} (?:to |and |until )?({_NUM}){_ORD}\b"),
)
_VENUE_RE = re.compile(r"\b(away from home|on the road|away|at home|home)\b")
_BOX_RE = re.compile(r"\b(?:in |inside )?(?:the )?(?:penalty area|penalty box|box)\b")
_FINAL_THIRD_RE = re.compile(r"\b(?:in |into )?(?:the )?(?:final|attacking) third\b")
#: Zones are a 6 x 3 grid, ``row * 6 + col``, x running toward the target goal.
#: The final third is x >= 80, so columns 4 and 5 in every row.
FINAL_THIRD_ZONES = [4, 5, 10, 11, 16, 17]
#: The penalty area is x >= 102 and 18 <= y <= 62. The grid cannot say that: its
#: last column starts at x = 100, and the area straddles all three rows, so this
#: is the last column in every row. A little wider than the box, never narrower.
PENALTY_AREA_ZONES = [5, 11, 17]
_SEASON_RE = re.compile(r"\b(20\d{2})\s*[/\-]\s*(\d{2,4})\b")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_MINUTE_AFTER_RE = re.compile(r"\bafter (?:the )?(\d{1,3})(?:st|nd|rd|th)? ?(?:minute|min)?\b")
_MINUTE_BEFORE_RE = re.compile(r"\bbefore (?:the )?(\d{1,3})(?:st|nd|rd|th)? ?(?:minute|min)?\b")
_MINUTE_RANGE_RE = re.compile(r"\b(\d{1,3})\s*(?:-|to|until)\s*(\d{1,3})\s*(?:minute|min)?\b")
_SEMI_OR_QUARTER_RE = re.compile(r"\b(semi|quarter|last 16|round of 16)\b")


Span = tuple[int, int]


def _occurrences(text: str, needle: str) -> list[Span]:
    """Every place `needle` occurs in `text`, as (start, end) character offsets."""

    out: list[Span] = []
    if not needle:
        return out
    i = text.find(needle)
    while i != -1:
        out.append((i, i + len(needle)))
        i = text.find(needle, i + len(needle))
    return out


def _number(word: str) -> int:
    """A minute count written as digits or as words ("fifteen", "twenty five")."""

    if word.isdigit():
        return int(word)
    return sum(_NUMBER_WORDS[w] for w in word.split())


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
    def _game_state(self, text: str) -> tuple[GameState | None, list[Span]]:
        groups: dict[str, list[str]] = self.alias_file.get("game_state") or {}
        for state, phrases in groups.items():
            for phrase in sorted(phrases, key=len, reverse=True):
                if spans := _occurrences(text, normalise(str(phrase))):
                    return cast(GameState, state), spans
        return None, []

    def _stages(self, text: str) -> list[tuple[Stage, list[Span]]]:
        groups: dict[str, list[str]] = self.alias_file.get("stage") or {}
        hits: list[tuple[Stage, list[Span]]] = []
        for stage, phrases in groups.items():
            for phrase in sorted(phrases, key=len, reverse=True):
                needle = normalise(str(phrase))
                found = _occurrences(text, needle)
                if not found:
                    continue
                if needle == "final" and _SEMI_OR_QUARTER_RE.search(text):
                    # "semi final" is a knockout round, not the final.
                    continue
                hits.append((cast(Stage, stage), found))
                break
        return hits

    def _minutes(self, text: str) -> tuple[tuple[int, int] | None, Span | None]:
        time_patterns: list[tuple[str, int, int]] = self.alias_file.get("time") or []
        for phrase, lo, hi in time_patterns:
            if found := _occurrences(text, normalise(str(phrase))):
                return (int(lo), int(hi)), found[0]
        if m := _LAST_N_RE.search(text):
            return (max(90 - _number(m.group(1)), 0), 98), m.span()
        if m := _FIRST_N_RE.search(text):
            return (0, min(_number(m.group(1)), 120)), m.span()
        for pattern in _MINUTE_SPAN_RES:
            if (m := pattern.search(text)) and _number(m.group(1)) <= _number(m.group(2)):
                return (_number(m.group(1)), min(_number(m.group(2)), 120)), m.span()
        if m := _MINUTE_RANGE_RE.search(text):
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo <= hi:
                return (lo, min(hi, 120)), m.span()
        if m := _MINUTE_AFTER_RE.search(text):
            return (int(m.group(1)), 120), m.span()
        if m := _MINUTE_BEFORE_RE.search(text):
            return (0, int(m.group(1))), m.span()
        return None, None

    def _seasons(
        self, text: str, competition_ids: list[int]
    ) -> tuple[list[int], list[str], dict[int, Span]]:
        """Resolve "2022/23" or "euro 2024" against the season names in the data."""

        wanted: list[tuple[str, Span]] = []
        consumed: list[str] = []
        if m := _SEASON_RE.search(text):
            tail = m.group(2)[-2:]
            wanted.append((f"{m.group(1)}/{tail}", m.span()))
            consumed.append(m.group(0))
        for m in _YEAR_RE.finditer(text):
            wanted.append((m.group(1), m.span()))
            consumed.append(m.group(0))

        ids: list[int] = []
        spans: dict[int, Span] = {}
        for season in self.catalogue.of_kind("season"):
            if competition_ids and season.parent_id not in competition_ids:
                continue
            name = normalise(season.name)
            for want, span in wanted:
                matched = normalise(want) == name or normalise(want) in name
                if matched and season.entity_id not in ids:
                    ids.append(season.entity_id)
                    spans[season.entity_id] = span
        return ids, consumed, spans

    # -------------------------------------------------------------------- main
    def parse(self, text: str) -> ParsedFilters:
        cleaned, origin = normalise_aligned(text or "")
        if not cleaned:
            return ParsedFilters(
                filters=Filters(), chips=[], unmatched=[], warnings=[], source="none"
            )

        def original(span: Span) -> str:
            """The characters the user typed for a span of the normalised text."""

            return text[origin[span[0]] : origin[span[1] - 1] + 1]

        chips: list[Chip] = []
        warnings: list[str] = []
        # Consumed phrases are blanked out, not cut, so every offset stays valid.
        masked = list(cleaned)

        def consume(spans: list[Span]) -> None:
            for a, b in spans:
                masked[a:b] = " " * (b - a)

        game_state, gs_spans = self._game_state(cleaned)
        minute_range, minute_span = self._minutes(cleaned)
        consume(gs_spans)
        if minute_span:
            consume([minute_span])

        # Zones, venue and stages are matched only on what the time and score
        # phrases did not already claim, so "final ten minutes" is a time window
        # and "final third" is a place, not the final of a tournament.
        ball_zone_ids: list[int] = []
        zone_chip: tuple[str, Span] | None = None
        for pattern, label, zones in (
            (_FINAL_THIRD_RE, "final third", FINAL_THIRD_ZONES),
            (_BOX_RE, "penalty area", PENALTY_AREA_ZONES),
        ):
            if m := pattern.search("".join(masked)):
                ball_zone_ids = list(zones)
                zone_chip = (label, m.span())
                consume([m.span()])
                break
        venue: Venue = "any"
        venue_span: Span | None = None
        if m := _VENUE_RE.search("".join(masked)):
            venue = "away" if m.group(1) in ("away", "away from home", "on the road") else "home"
            venue_span = m.span()
            consume([venue_span])
        stage_hits = self._stages("".join(masked))
        stages = [stage for stage, _ in stage_hits]
        consume([span for _, spans in stage_hits for span in spans])

        residual = "".join(masked)
        token_spans = [m.span() for m in re.finditer(r"\S+", residual)]
        tokens = [residual[a:b] for a, b in token_spans]

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
        team_chips: dict[int, tuple[Chip, str]] = {}

        for (start, end), group in sorted(by_span.items()):
            phrase = " ".join(tokens[start:end])
            span_text = original((token_spans[start][0], token_spans[end - 1][1]))
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
                         ids=[entity.entity_id], span=span_text)
                )
                if entity.frames == 0:
                    warnings.append(
                        f"No positional data for {entity.name}; "
                        "see the dropdown for what is covered."
                    )
            elif entity.kind == "team":
                before = tokens[start - 1] if start > 0 else ""
                after = tokens[end] if end < len(tokens) else ""
                if before in defending_markers or after in _DEFENDING_AFTER:
                    role_here = "defending"
                else:
                    role_here = "attacking"
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
                chip = Chip(field="team", label="Team", value=entity.name,
                            ids=[entity.entity_id], span=span_text)
                team_chips[entity.entity_id] = (chip, entity.name)
                chips.append(chip)
                if entity.frames == 0:
                    warnings.append(f"No positional data for {entity.name}.")
            elif entity.kind == "player" and entity.entity_id not in player_ids:
                player_ids.append(entity.entity_id)
                player_role = "attacking"
                chips.append(
                    Chip(field="player", label="Player", value=entity.name,
                         ids=[entity.entity_id], span=span_text)
                )

        if team_ids:
            attacking = [t for t in team_ids if seen_team_roles[t] == "attacking"]
            defending = [t for t in team_ids if seen_team_roles[t] == "defending"]
            if attacking and defending:
                # One role covers every team, so "Germany against Spain" cannot be
                # both. Possession is the side that can be expressed; the
                # opponent is named in the warning instead of silently dropped.
                dropped = [team_chips[t] for t in defending]
                chips = [c for c in chips if all(c is not d[0] for d in dropped)]
                names = ", ".join(d[1] for d in dropped)
                kept = ", ".join(team_chips[t][1] for t in attacking)
                warnings.append(
                    f"Filtering on {kept} in possession; the opponent ({names}) "
                    "cannot be filtered separately."
                )
                team_ids = attacking
                team_role = "attacking"
            else:
                team_role = "defending" if defending else "attacking"

        season_ids, season_phrases, season_spans = self._seasons(cleaned, competition_ids)
        for sid in season_ids:
            season = self.catalogue.by_id("season", sid)
            if season is not None:
                chips.append(
                    Chip(field="season", label="Season", value=season.name, ids=[sid],
                         span=original(season_spans[sid]))
                )

        if venue_span is not None:
            chips.append(
                Chip(field="venue", label="Venue", value=venue, span=original(venue_span))
            )
        if zone_chip is not None:
            chips.append(
                Chip(field="ball_zone", label="Zone", value=zone_chip[0],
                     ids=list(ball_zone_ids), span=original(zone_chip[1]))
            )
        if game_state:
            chips.append(
                Chip(field="game_state", label="Game state", value=game_state,
                     span=original(gs_spans[0]))
            )
        for stage, stage_found in stage_hits:
            chips.append(
                Chip(field="stage", label="Stage", value=stage, span=original(stage_found[0]))
            )
        if minute_range and minute_span:
            chips.append(
                Chip(field="minute_range", label="Minutes",
                     value=f"{minute_range[0]}-{minute_range[1]}", span=original(minute_span))
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
            ball_zone_ids=ball_zone_ids,
            venue=venue,
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


def _same(a: object, b: object) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return sorted(a) == sorted(b)
    return a == b


def overridden(explicit: Filters, parsed: Filters) -> list[str]:
    """Fields where an explicit choice and the text asked for different things."""

    default = Filters()
    out: list[str] = []
    for name in Filters.model_fields:
        mine, theirs = getattr(explicit, name), getattr(parsed, name)
        if mine == getattr(default, name) or theirs == getattr(default, name):
            continue
        if not _same(mine, theirs):
            out.append(name)
    return out


def merge(explicit: Filters, parsed: Filters) -> Filters:
    """Explicit filters win on conflicts; parsed text fills the gaps."""

    data = parsed.model_dump()
    default = Filters()
    for field_name, value in explicit.model_dump().items():
        if value != getattr(default, field_name):
            data[field_name] = value
    # A role belongs to the teams (or players) it was read alongside. If the
    # selected teams replace the ones in the text, the text's role goes with them.
    conflicts = overridden(explicit, parsed)
    for ids, role in (("team_ids", "team_role"), ("player_ids", "player_role")):
        if ids in conflicts:
            data[role] = getattr(explicit, role)
    return Filters(**data)
