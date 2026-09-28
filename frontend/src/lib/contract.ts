/**
 * Compile-time contract between the hand-written types and the OpenAPI schema.
 *
 * `npm run gen:api` regenerates `api-types.ts` from the API's own schema; these
 * assertions then fail `tsc` if a field is renamed, removed or retyped on either
 * side, so a breaking API change breaks the build instead of the app.
 *
 * Direction matters and both are checked:
 *   - a *response* the server produces has to satisfy the shape the UI reads;
 *   - a *request* the UI sends has to satisfy the shape the server accepts.
 *
 * There is no runtime code here on purpose.
 */

import type { components } from './api-types';
import type {
  Chip,
  FiltersResponse,
  Interval,
  Outcome,
  ParsedFilters,
  SearchRequest,
  SearchResponse,
  SearchResult,
  Sequence,
  SituationDetail,
  Stats,
} from './types';

type Schemas = components['schemas'];

/**
 * `Satisfies<Expected, Actual>` fails to compile unless `Actual` really is
 * assignable to `Expected`. The constraint does the work; the alias only exists
 * so each check reads as one line.
 */
type Satisfies<Expected, Actual extends Expected> = [Expected, Actual];

/**
 * Drop optionality, recursively, while preserving tuples.
 *
 * FastAPI emits a field that has a server-side default as optional even though a
 * response always carries it, so comparing raw shapes would fail on every such
 * field and say nothing about the contract. Normalising both sides keeps the
 * check on what actually matters: field names and value types.
 */
type DeepDefined<T> = T extends readonly unknown[]
  ? { [K in keyof T]: DeepDefined<T[K]> }
  : T extends object
    ? { [K in keyof T]-?: DeepDefined<Exclude<T[K], undefined>> }
    : T;

type Check<Expected, Actual extends DeepDefined<Expected>> = Satisfies<
  DeepDefined<Expected>,
  Actual
>;

// --- responses: the server must satisfy what the UI reads -----------------
export type SearchResponseContract = Check<SearchResponse, DeepDefined<Schemas['SearchResponse']>>;
export type SearchResultContract = Check<SearchResult, DeepDefined<Schemas['SearchResult']>>;
export type StatsContract = Check<Stats, DeepDefined<Schemas['Stats']>>;
export type IntervalContract = Check<Interval, DeepDefined<Schemas['Interval']>>;
export type OutcomeContract = Check<Outcome, DeepDefined<Schemas['Outcome']>>;
export type ParsedFiltersContract = Check<ParsedFilters, DeepDefined<Schemas['ParsedFilters']>>;
export type ChipContract = Check<Chip, DeepDefined<Schemas['Chip']>>;
export type FiltersResponseContract = Check<FiltersResponse, DeepDefined<Schemas['FiltersResponse']>>;
export type SituationContract = Check<SituationDetail, DeepDefined<Schemas['SituationDetail']>>;
export type SequenceContract = Check<Sequence, DeepDefined<Schemas['Sequence']>>;

// --- requests: what the UI sends must satisfy the server ------------------
export type SearchRequestContract = Check<Schemas['SearchRequest'], DeepDefined<SearchRequest>>;
