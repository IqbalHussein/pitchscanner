/**
 * One page, three panels: sketch, results and stats, playback.
 *
 * The whole sketch lives in the URL, so any query is a shareable link, and the
 * search is debounced while dragging so a drag does not flood the API.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FilterBar } from './components/FilterBar';
import { Pitch } from './components/Pitch';
import { Playback } from './components/Playback';
import { Results } from './components/Results';
import { StatsPanel } from './components/StatsPanel';
import { api, ApiError } from './lib/api';
import { ms } from './lib/format';
import { PRESETS } from './lib/presets';
import {
  emptyFilters,
  type FiltersResponse,
  type ParsedFilters,
  type SearchResponse,
  type Sequence,
  type SituationDetail,
  type Side,
} from './lib/types';
import { decodeSketch, encodeSketch, nextId, type SketchState } from './lib/urlState';

const DEBOUNCE_MS = 150;

const initialState = (): SketchState => {
  const preset = PRESETS[0]!;
  const base: SketchState = {
    ball: preset.ball,
    players: preset.players.map((p) => ({ ...p, id: nextId() })),
    k: 25,
    encoding: 'E1',
    mirror: true,
    filterText: '',
    filters: emptyFilters(),
  };
  return { ...base, ...decodeSketch(window.location.search) };
};

export default function App() {
  const [state, setState] = useState<SketchState>(initialState);
  const [addSide, setAddSide] = useState<Side | 'ball'>('attacker');
  const [coverage, setCoverage] = useState<FiltersResponse | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [parsed, setParsed] = useState<ParsedFilters | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [overlay, setOverlay] = useState<SituationDetail | null>(null);
  const [detail, setDetail] = useState<SituationDetail | null>(null);
  const [sequence, setSequence] = useState<Sequence | null>(null);
  const [trail, setTrail] = useState<{ x: number; y: number }[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const inFlight = useRef<AbortController | null>(null);

  useEffect(() => {
    api.filters().then(setCoverage).catch(() => undefined);
  }, []);

  // --- the search itself --------------------------------------------------
  const runSearch = useCallback((current: SketchState) => {
    inFlight.current?.abort();
    const controller = new AbortController();
    inFlight.current = controller;
    setLoading(true);
    api
      .search(
        {
          ball: current.ball,
          attackers: current.players
            .filter((p) => p.side === 'attacker')
            .map(({ x, y, keeper }) => ({ x, y, keeper })),
          defenders: current.players
            .filter((p) => p.side === 'defender')
            .map(({ x, y, keeper }) => ({ x, y, keeper })),
          k: current.k,
          filters: current.filters,
          filter_text: current.filterText || null,
          options: { mirror: current.mirror, encoding: current.encoding },
        },
        controller.signal,
      )
      .then((result) => {
        setResponse(result);
        if (result.parsed_filters) setParsed(result.parsed_filters);
        setError(null);
      })
      .catch((cause: unknown) => {
        if (cause instanceof DOMException && cause.name === 'AbortError') return;
        setError(cause instanceof ApiError ? cause.message : 'Search failed. Try again.');
      })
      .finally(() => setLoading(false));
  }, []);

  // Debounced: dragging a player fires a search per frame otherwise.
  useEffect(() => {
    const hasPlayers = state.players.length > 0;
    if (!hasPlayers) {
      setResponse(null);
      return;
    }
    const timer = window.setTimeout(() => runSearch(state), DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [state, runSearch]);

  // The URL follows the sketch, so the address bar is always shareable.
  useEffect(() => {
    const query = encodeSketch(state);
    window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname);
  }, [state]);

  // --- hover overlay and selection ---------------------------------------
  useEffect(() => {
    if (!hovered) {
      setOverlay(null);
      return;
    }
    let cancelled = false;
    api
      .situation(hovered)
      .then((result) => {
        if (!cancelled) setOverlay(result);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [hovered]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      setSequence(null);
      setTrail(null);
      return;
    }
    Promise.all([api.situation(selected), api.sequence(selected)])
      .then(([situation, seq]) => {
        setDetail(situation);
        setSequence(seq);
      })
      .catch(() => undefined);
  }, [selected]);

  // --- editing ------------------------------------------------------------
  const update = (patch: Partial<SketchState>) => setState((prev) => ({ ...prev, ...patch }));

  const addPlayer = (x: number, y: number) => {
    if (addSide === 'ball') {
      update({ ball: { x, y } });
      return;
    }
    const side: Side = addSide;
    const sameSide = state.players.filter((p) => p.side === side).length;
    if (sameSide >= 11) return;
    update({ players: [...state.players, { id: nextId(), x, y, side }] });
  };

  const movePlayer = (id: string, x: number, y: number) =>
    setState((prev) => ({
      ...prev,
      players: prev.players.map((p) => (p.id === id ? { ...p, x, y } : p)),
    }));

  const removePlayer = (id: string) =>
    setState((prev) => ({ ...prev, players: prev.players.filter((p) => p.id !== id) }));

  const applyPreset = (presetId: string) => {
    const preset = PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSelected(null);
    update({
      ball: preset.ball,
      players: preset.players.map((p) => ({ ...p, id: nextId() })),
    });
  };

  const counts = useMemo(
    () => ({
      attackers: state.players.filter((p) => p.side === 'attacker').length,
      defenders: state.players.filter((p) => p.side === 'defender').length,
    }),
    [state.players],
  );

  const overlayProps = overlay
    ? {
        ball: overlay.ball,
        attackers: overlay.attackers,
        defenders: overlay.defenders,
        visibleArea: overlay.visible_area,
      }
    : null;

  return (
    <div className="app">
      <header className="masthead">
        <div>
          <h1>Match Situation Search</h1>
          <p className="muted">
            Sketch a situation. See the most similar real moments from professional matches, and
            what they led to.
          </p>
        </div>
        {coverage && (
          <p className="muted counts" data-testid="coverage">
            {coverage.total_frames.toLocaleString()} frames · {coverage.total_matches} matches ·{' '}
            {coverage.total_events.toLocaleString()} events
          </p>
        )}
      </header>

      <FilterBar
        coverage={coverage}
        filters={state.filters}
        filterText={state.filterText}
        parsed={parsed}
        onFiltersChange={(filters) => update({ filters })}
        onFilterTextChange={(filterText) => update({ filterText })}
        onParsed={setParsed}
      />

      <main className="panels">
        <section className="panel sketch" aria-label="Sketch">
          <div className="toolbar">
            <div className="segmented" role="group" aria-label="What to place">
              {(['attacker', 'defender', 'ball'] as const).map((mode) => (
                <button
                  key={mode}
                  className={addSide === mode ? 'active' : ''}
                  onClick={() => setAddSide(mode)}
                  data-testid={`mode-${mode}`}
                >
                  {mode === 'attacker'
                    ? `Attackers (${counts.attackers})`
                    : mode === 'defender'
                      ? `Defenders (${counts.defenders})`
                      : 'Ball'}
                </button>
              ))}
            </div>
            <div className="presets">
              {PRESETS.map((preset) => (
                <button
                  key={preset.id}
                  onClick={() => applyPreset(preset.id)}
                  title={preset.description}
                  data-testid={`preset-${preset.id}`}
                >
                  {preset.name}
                </button>
              ))}
              <button
                onClick={() => update({ players: [] })}
                className="ghost"
                data-testid="clear"
              >
                Clear
              </button>
            </div>
          </div>

          <Pitch
            ball={state.ball}
            players={state.players}
            addSide={addSide}
            overlay={overlayProps}
            trail={trail}
            onAdd={addPlayer}
            onMove={movePlayer}
            onMoveBall={(x, y) => update({ ball: { x, y } })}
            onRemove={removePlayer}
          />

          <div className="options">
            <label>
              Results
              <input
                type="number"
                min={1}
                max={100}
                value={state.k}
                onChange={(event) =>
                  update({ k: Math.min(100, Math.max(1, Number(event.target.value) || 1)) })
                }
                data-testid="k-input"
              />
            </label>
            <label>
              Encoding
              <select
                value={state.encoding}
                onChange={(event) => update({ encoding: event.target.value })}
                data-testid="encoding-select"
              >
                {(coverage?.encodings ?? ['E1']).map((encoding) => (
                  <option key={encoding} value={encoding}>
                    {encoding}
                  </option>
                ))}
              </select>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={state.mirror}
                onChange={(event) => update({ mirror: event.target.checked })}
              />
              Mirror left/right
            </label>
            <button
              className="ghost"
              onClick={() => navigator.clipboard?.writeText(window.location.href)}
            >
              Copy link
            </button>
          </div>
          <p className="muted hint">
            Click to place, drag to move, right-click or long-press to remove. Select a player
            and use the arrow keys to nudge it.
          </p>
        </section>

        <section className="panel side" aria-label="Results and statistics">
          {error && (
            <p className="notice error" role="alert" data-testid="error">
              {error}
            </p>
          )}
          {response && (
            <p className="muted timings" data-testid="timings">
              {response.results.length} results in {ms(response.timing_ms.total)} (encode{' '}
              {ms(response.timing_ms.encode)}, search {ms(response.timing_ms.search)}, aggregate{' '}
              {ms(response.timing_ms.aggregate)}) · {response.candidates_examined} candidates ·{' '}
              {response.encoding}
              {response.relaxed ? ' · exact scan of the filtered subset' : ''}
            </p>
          )}
          <StatsPanel stats={response?.stats ?? null} />
          <Results
            results={response?.results ?? []}
            selected={selected}
            loading={loading}
            onHover={setHovered}
            onSelect={setSelected}
          />
        </section>
      </main>

      <Playback detail={detail} sequence={sequence} onFrame={setTrail} />

      <footer className="footer muted">
        Data:{' '}
        <a href="https://github.com/statsbomb/open-data" rel="noreferrer noopener" target="_blank">
          StatsBomb Open Data
        </a>{' '}
        — free for non-commercial use, with thanks. Freeze frames only show players inside the
        broadcast camera view, and coverage is skewed towards major tournaments, so these
        statistics describe that sample rather than football in general.
      </footer>
    </div>
  );
}
