/**
 * The sketch lives in the URL, so every query is a shareable link.
 *
 * Encoding is compact and positional rather than JSON: `a86.20` is an attacker,
 * `d92.25` a defender, `k119.40` a keeper. A long JSON blob in the address bar
 * would be unreadable and would not survive being pasted into a chat window.
 */

import { emptyFilters, type Filters, type SketchPlayer } from './types';

export interface SketchState {
  ball: { x: number; y: number };
  players: SketchPlayer[];
  k: number;
  encoding: string;
  mirror: boolean;
  filterText: string;
  filters: Filters;
}

let counter = 0;
export const nextId = (): string => `p${++counter}`;

const round = (n: number) => Math.round(n * 10) / 10;

const encodePlayer = (p: SketchPlayer) =>
  `${p.keeper ? 'k' : p.side === 'attacker' ? 'a' : 'd'}${round(p.x)}.${round(p.y)}`;

export function encodeSketch(state: SketchState): string {
  const params = new URLSearchParams();
  params.set('b', `${round(state.ball.x)}.${round(state.ball.y)}`);
  if (state.players.length) params.set('p', state.players.map(encodePlayer).join('_'));
  if (state.k !== 25) params.set('k', String(state.k));
  if (state.encoding !== 'E1') params.set('e', state.encoding);
  if (!state.mirror) params.set('m', '0');
  if (state.filterText.trim()) params.set('q', state.filterText.trim());
  if (state.filters.competition_ids.length)
    params.set('c', state.filters.competition_ids.join(','));
  if (state.filters.season_ids.length) params.set('s', state.filters.season_ids.join(','));
  return params.toString();
}

const parsePair = (raw: string | null): { x: number; y: number } | null => {
  if (!raw) return null;
  const parts = raw.split('.');
  // "84.5.22.1" is x=84.5, y=22.1; "84.22" is x=84, y=22.
  if (parts.length === 2) {
    const x = Number(parts[0]);
    const y = Number(parts[1]);
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
  }
  if (parts.length === 4) {
    const x = Number(`${parts[0]}.${parts[1]}`);
    const y = Number(`${parts[2]}.${parts[3]}`);
    return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
  }
  return null;
};

const numbers = (raw: string | null): number[] =>
  (raw ?? '')
    .split(',')
    // Empty strings have to go before Number(): `Number('')` is 0, which would
    // turn a missing parameter into a filter on competition id 0 and silently
    // return no results at all.
    .map((v) => v.trim())
    .filter((v) => v.length > 0)
    .map((v) => Number(v))
    .filter((v) => Number.isFinite(v));

export function decodeSketch(search: string): Partial<SketchState> {
  const params = new URLSearchParams(search.startsWith('?') ? search.slice(1) : search);
  const out: Partial<SketchState> = {};

  const ball = parsePair(params.get('b'));
  if (ball) out.ball = ball;

  const raw = params.get('p');
  if (raw) {
    const players: SketchPlayer[] = [];
    for (const token of raw.split('_')) {
      const kind = token[0];
      const point = parsePair(token.slice(1));
      if (!point || !kind) continue;
      if (kind === 'a') players.push({ id: nextId(), side: 'attacker', ...point });
      else if (kind === 'd') players.push({ id: nextId(), side: 'defender', ...point });
      else if (kind === 'k')
        players.push({ id: nextId(), side: 'defender', keeper: true, ...point });
    }
    out.players = players;
  }

  const k = Number(params.get('k'));
  if (Number.isFinite(k) && k > 0) out.k = Math.min(100, Math.round(k));
  const encoding = params.get('e');
  if (encoding) out.encoding = encoding.toUpperCase();
  if (params.get('m') === '0') out.mirror = false;
  const q = params.get('q');
  if (q) out.filterText = q;

  const competitions = numbers(params.get('c'));
  const seasons = numbers(params.get('s'));
  if (competitions.length || seasons.length) {
    out.filters = {
      ...emptyFilters(),
      competition_ids: competitions,
      season_ids: seasons,
    };
  }
  return out;
}
