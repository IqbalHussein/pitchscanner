"""Rules first, LLM only when the rules find nothing."""

from __future__ import annotations

from pitchscanner.config import Settings, get_settings
from pitchscanner.filters.catalogue import Catalogue
from pitchscanner.filters.llm import LLMFilterParser
from pitchscanner.filters.parser import FilterParser, merge
from pitchscanner.models import Filters, ParsedFilters
from pitchscanner.store.base import Store


class FilterService:
    """Lazy on purpose: building the app must not require a reachable database.

    The catalogue is read once and then cached; `warm()` in the API's startup hook
    pays that cost before the first request instead of during it.
    """

    def __init__(self, store: Store, settings: Settings | None = None) -> None:
        self.cfg = settings or get_settings()
        self.store = store
        self._catalogue: Catalogue | None = None
        self._rules: FilterParser | None = None
        self._llm: LLMFilterParser | None = None

    @property
    def catalogue(self) -> Catalogue:
        if self._catalogue is None:
            self._catalogue = Catalogue.from_entries(self.store.catalogue())
        return self._catalogue

    @property
    def rules(self) -> FilterParser:
        if self._rules is None:
            self._rules = FilterParser(self.catalogue, threshold=self.cfg.fuzzy_threshold)
        return self._rules

    @property
    def llm(self) -> LLMFilterParser:
        if self._llm is None:
            self._llm = LLMFilterParser(self.catalogue, self.cfg)
        return self._llm

    def warm(self) -> int:
        """Load the catalogue now. Returns how many entities it holds."""

        return len(self.catalogue.entities)

    def parse(self, text: str) -> ParsedFilters:
        parsed = self.rules.parse(text)
        if parsed.filters.is_empty() and text.strip():
            fallback = self.llm.parse(text)
            if fallback is not None and not fallback.filters.is_empty():
                return fallback
        return parsed

    def resolve(self, explicit: Filters, text: str | None) -> tuple[Filters, ParsedFilters | None]:
        """Explicit filters win on conflicts; text fills in the rest."""

        if not text or not text.strip():
            return explicit, None
        parsed = self.parse(text)
        return merge(explicit, parsed.filters), parsed
