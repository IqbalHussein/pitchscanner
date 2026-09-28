/** Thin API client. One place that knows about URLs, errors and aborts. */

import type {
  FiltersResponse,
  ParsedFilters,
  SearchRequest,
  SearchResponse,
  Sequence,
  SituationDetail,
} from './types';

const BASE = import.meta.env.VITE_API_URL ?? '';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { 'content-type': 'application/json', ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new ApiError(
      response.status === 429
        ? 'Too many searches from this address. Give it a minute.'
        : `Request failed (${response.status}): ${detail.slice(0, 200)}`,
      response.status,
    );
  }
  return (await response.json()) as T;
}

export const api = {
  search: (body: SearchRequest, signal?: AbortSignal) =>
    request<SearchResponse>('/v1/search', {
      method: 'POST',
      body: JSON.stringify(body),
      signal,
    }),

  parseFilters: (text: string, signal?: AbortSignal) =>
    request<ParsedFilters>('/v1/filters/parse', {
      method: 'POST',
      body: JSON.stringify({ text }),
      signal,
    }),

  filters: () => request<FiltersResponse>('/v1/filters'),

  situation: (eventId: string) =>
    request<SituationDetail>(`/v1/situations/${encodeURIComponent(eventId)}`),

  sequence: (eventId: string) =>
    request<Sequence>(`/v1/situations/${encodeURIComponent(eventId)}/sequence`),
};
