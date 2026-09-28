/** The result list: match, minute, similarity and outcome badges. */

import type { SearchResult } from '../lib/types';
import { clock, pct } from '../lib/format';

export interface ResultsProps {
  results: SearchResult[];
  selected: string | null;
  loading: boolean;
  onHover: (eventId: string | null) => void;
  onSelect: (eventId: string) => void;
}

export function Results({ results, selected, loading, onHover, onSelect }: ResultsProps) {
  if (!loading && results.length === 0) {
    return (
      <p className="empty">
        No situations matched. Try removing a filter, or move the ball closer to where the
        data actually is.
      </p>
    );
  }

  return (
    <ol className="results" data-testid="results" onMouseLeave={() => onHover(null)}>
      {results.map((result) => (
        <li key={result.event_id}>
          <button
            className={`result ${selected === result.event_id ? 'active' : ''}`}
            onMouseEnter={() => onHover(result.event_id)}
            onFocus={() => onHover(result.event_id)}
            onClick={() => onSelect(result.event_id)}
            data-testid="result"
          >
            <span className="result-main">
              <strong>{result.match}</strong>
              <span className="muted">
                {result.competition} {result.season} · {clock(result.minute, result.period)} ·{' '}
                {result.team} in possession
                {result.mirrored ? ' · mirrored' : ''}
              </span>
            </span>
            <span className="badges">
              {result.outcome?.shot && <span className="badge shot">shot</span>}
              {result.outcome?.goal && <span className="badge goal">goal</span>}
              {result.outcome?.turnover && <span className="badge turnover">turnover</span>}
              {result.outcome?.box_entry && <span className="badge box">box entry</span>}
              <span className="badge similarity" title={`distance ${result.distance.toFixed(4)}`}>
                {pct(result.similarity)}
              </span>
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}
