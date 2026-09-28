/**
 * Hand-written mirrors of the API models.
 *
 * `npm run gen:api` regenerates `api-types.ts` from the committed
 * `openapi.json`, and `contract.ts` asserts these types are assignable to the
 * generated ones, so a renamed field fails the build instead of failing at
 * runtime.
 */

export const PITCH_LENGTH = 120;
export const PITCH_WIDTH = 80;

export type Side = 'attacker' | 'defender';

export interface PointIn {
  x: number;
  y: number;
  keeper?: boolean;
}

export interface SketchPlayer extends PointIn {
  id: string;
  side: Side;
}

export type GameState = 'any' | 'winning' | 'drawing' | 'losing';
export type Role = 'any' | 'attacking' | 'defending';
export type Stage = 'group' | 'knockout' | 'final' | 'regular';

export interface Filters {
  competition_ids: number[];
  season_ids: number[];
  team_ids: number[];
  team_role: Role;
  player_ids: number[];
  player_role: Role;
  game_state: GameState;
  minute_range: [number, number] | null;
  stages: Stage[];
  ball_zone_ids: number[];
}

export const emptyFilters = (): Filters => ({
  competition_ids: [],
  season_ids: [],
  team_ids: [],
  team_role: 'any',
  player_ids: [],
  player_role: 'any',
  game_state: 'any',
  minute_range: null,
  stages: [],
  ball_zone_ids: [],
});

export interface Chip {
  field: string;
  label: string;
  value: string;
  ids: number[];
}

export interface ParsedFilters {
  filters: Filters;
  chips: Chip[];
  unmatched: string[];
  warnings: string[];
  source: 'rules' | 'llm' | 'none';
}

export interface Outcome {
  shot: boolean;
  xg_next: number;
  goal: boolean;
  turnover: boolean;
  box_entry: boolean;
  seconds_to_end: number;
}

export interface SearchResult {
  event_id: string;
  distance: number;
  similarity: number;
  match: string;
  match_id: number;
  competition: string;
  season: string;
  minute: number;
  period: number;
  possession_id: number;
  team: string;
  mirrored: boolean;
  outcome: Outcome | null;
}

export interface Interval {
  value: number;
  ci95: [number, number];
  vs_zone_baseline: number | null;
}

export interface Stats {
  n: number;
  n_possessions: number;
  shot_rate: Interval;
  xg_mean: Interval;
  turnover_rate: Interval;
  box_entry_rate: Interval;
  goal_rate: Interval;
  unweighted: Record<string, number>;
  zone_id: number;
  zone_label: string;
  baseline_n: number;
  low_confidence: boolean;
  notes: string[];
}

export interface Timing {
  encode: number;
  search: number;
  dedup: number;
  aggregate: number;
  total: number;
}

export interface SearchResponse {
  results: SearchResult[];
  stats: Stats | null;
  parsed_filters: ParsedFilters | null;
  filters_applied: Filters;
  encoding: string;
  candidates_examined: number;
  relaxed: boolean;
  timing_ms: Timing;
}

export interface SearchRequest {
  ball: PointIn;
  attackers: PointIn[];
  defenders: PointIn[];
  k: number;
  filters?: Filters;
  filter_text?: string | null;
  options?: { mirror?: boolean; encoding?: string; window_s?: number; dedup?: boolean };
}

export interface SeasonCoverage {
  season_id: number;
  season_name: string;
  matches: number;
  frames: number;
}

export interface CompetitionCoverage {
  competition_id: number;
  competition_name: string;
  matches: number;
  frames: number;
  seasons: SeasonCoverage[];
}

export interface FiltersResponse {
  competitions: CompetitionCoverage[];
  teams: { team_id: number; team_name: string; frames: number }[];
  stages: Stage[];
  game_states: GameState[];
  encodings: string[];
  total_frames: number;
  total_matches: number;
  total_events: number;
}

export interface SituationDetail {
  event_id: string;
  match_id: number;
  match: string;
  competition: string;
  season: string;
  minute: number;
  period: number;
  possession_id: number;
  team: string;
  ball: PointIn;
  attackers: PointIn[];
  defenders: PointIn[];
  visible_area: PointIn[] | null;
  outcome: Outcome | null;
}

export interface SequenceEvent {
  event_id: string;
  type: string;
  team: string;
  player: string | null;
  x: number | null;
  y: number | null;
  second_offset: number;
  xg: number | null;
  goal: boolean;
}

export interface Sequence {
  event_id: string;
  window_s: number;
  events: SequenceEvent[];
}
