/**
 * Dropdowns for competition and season, a free-text box for everything else, and
 * chips showing what was understood.
 *
 * Both halves write to the same filter state, and the most recent change wins.
 */

import { useEffect, useMemo, useState } from 'react';
import { api } from '../lib/api';
import type { Filters, FiltersResponse, ParsedFilters } from '../lib/types';

const DEBOUNCE_MS = 300;

export interface FilterBarProps {
  coverage: FiltersResponse | null;
  filters: Filters;
  filterText: string;
  parsed: ParsedFilters | null;
  onFiltersChange: (next: Filters) => void;
  onFilterTextChange: (next: string) => void;
  onParsed: (parsed: ParsedFilters | null) => void;
}

export function FilterBar({
  coverage,
  filters,
  filterText,
  parsed,
  onFiltersChange,
  onFilterTextChange,
  onParsed,
}: FilterBarProps) {
  const [pending, setPending] = useState(false);

  // Parse as the user types, debounced. The parse endpoint is cheap and never
  // touches the database, so this is safe to call on most keystrokes.
  useEffect(() => {
    if (!filterText.trim()) {
      onParsed(null);
      return;
    }
    const controller = new AbortController();
    setPending(true);
    const timer = window.setTimeout(() => {
      api
        .parseFilters(filterText, controller.signal)
        .then((result) => {
          onParsed(result);
          // Text that names a competition or season updates the dropdowns, so the
          // two controls never disagree about what is selected.
          if (result.filters.competition_ids.length || result.filters.season_ids.length) {
            onFiltersChange({
              ...filters,
              competition_ids: result.filters.competition_ids.length
                ? result.filters.competition_ids
                : filters.competition_ids,
              season_ids: result.filters.season_ids.length
                ? result.filters.season_ids
                : filters.season_ids,
            });
          }
        })
        .catch(() => undefined)
        .finally(() => setPending(false));
    }, DEBOUNCE_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
      setPending(false);
    };
    // `filters` is deliberately excluded: including it would re-parse on every
    // dropdown change and fight the user for control of the selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterText]);

  const seasons = useMemo(() => {
    if (!coverage) return [];
    const competitions = filters.competition_ids.length
      ? coverage.competitions.filter((c) =>
          filters.competition_ids.includes(c.competition_id),
        )
      : coverage.competitions;
    return competitions.flatMap((c) =>
      c.seasons.map((s) => ({ ...s, competition: c.competition_name })),
    );
  }, [coverage, filters.competition_ids]);

  const multi = (event: React.ChangeEvent<HTMLSelectElement>): number[] =>
    Array.from(event.target.selectedOptions)
      .map((o) => Number(o.value))
      .filter((v) => Number.isFinite(v));

  const removeChip = (field: string, ids: number[]) => {
    const next = { ...filters };
    if (field === 'competition')
      next.competition_ids = filters.competition_ids.filter((id) => !ids.includes(id));
    if (field === 'season')
      next.season_ids = filters.season_ids.filter((id) => !ids.includes(id));
    if (field === 'team') next.team_ids = filters.team_ids.filter((id) => !ids.includes(id));
    if (field === 'player')
      next.player_ids = filters.player_ids.filter((id) => !ids.includes(id));
    if (field === 'game_state') next.game_state = 'any';
    if (field === 'minute_range') next.minute_range = null;
    if (field === 'stage') next.stages = [];
    onFiltersChange(next);
    // The chip came from the text box, so the text has to change too or the next
    // keystroke would bring it straight back.
    onParsed(parsed ? { ...parsed, chips: parsed.chips.filter((c) => c.field !== field) } : null);
  };

  return (
    <section className="filter-bar" aria-label="Filters">
      <div className="filter-row">
        <label className="field">
          <span>Competition</span>
          <select
            multiple
            size={3}
            value={filters.competition_ids.map(String)}
            onChange={(event) =>
              onFiltersChange({ ...filters, competition_ids: multi(event), season_ids: [] })
            }
            data-testid="competition-select"
          >
            {(coverage?.competitions ?? []).map((c) => (
              <option key={c.competition_id} value={c.competition_id}>
                {c.competition_name} · {c.matches} matches · {c.frames} frames
              </option>
            ))}
          </select>
        </label>

        <label className="field">
          <span>Season</span>
          <select
            multiple
            size={3}
            value={filters.season_ids.map(String)}
            onChange={(event) => onFiltersChange({ ...filters, season_ids: multi(event) })}
            data-testid="season-select"
          >
            {seasons.map((s) => (
              <option key={`${s.competition}-${s.season_id}`} value={s.season_id}>
                {s.season_name} · {s.frames} frames
              </option>
            ))}
          </select>
        </label>

        <label className="field grow">
          <span>Anything else</span>
          <input
            type="text"
            value={filterText}
            placeholder='e.g. "Leverkusen when losing in the last 15 minutes"'
            onChange={(event) => onFilterTextChange(event.target.value)}
            data-testid="filter-text"
            aria-describedby="filter-help"
          />
          <small id="filter-help">
            Teams, players, game state, time and stage. Competitions and seasons come from
            the dropdowns{pending ? ' · reading…' : ''}
          </small>
        </label>
      </div>

      {parsed && (parsed.chips.length > 0 || parsed.unmatched.length > 0 || parsed.warnings.length > 0) && (
        <div className="chips" data-testid="chips">
          {parsed.chips.map((chip) => (
            <button
              key={`${chip.field}-${chip.value}`}
              className="chip"
              onClick={() => removeChip(chip.field, chip.ids)}
              title="Remove this filter"
            >
              {chip.label}: {chip.value} <span aria-hidden>×</span>
            </button>
          ))}
          {parsed.unmatched.length > 0 && (
            <span className="chip muted" data-testid="unmatched">
              Didn&apos;t understand: {parsed.unmatched.map((w) => `'${w}'`).join(', ')}
            </span>
          )}
          {parsed.warnings.map((warning) => (
            <span key={warning} className="chip warning">
              {warning}
            </span>
          ))}
          {parsed.source === 'llm' && <span className="chip muted">read by the language model</span>}
        </div>
      )}
    </section>
  );
}
