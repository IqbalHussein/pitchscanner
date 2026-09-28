/** Formatting helpers, kept out of the components. */

export const pct = (value: number): string => `${(value * 100).toFixed(0)}%`;

export const xg = (value: number): string => value.toFixed(3).replace(/^0/, '');

export const ratio = (value: number | null): string =>
  value === null ? '—' : `${value.toFixed(2)}×`;

export const ms = (value: number): string =>
  value >= 10 ? `${value.toFixed(0)}ms` : `${value.toFixed(1)}ms`;

export const clock = (minute: number, period: number): string =>
  period > 2 ? `ET ${minute}'` : `${minute}'`;

/** "1.8× the average from this zone", or a plain reading when there is no baseline. */
export function baselineSentence(value: number | null, label: string): string {
  if (value === null) return `No baseline for the ${label} yet.`;
  if (value >= 1.15) return `${value.toFixed(1)}× the average from the ${label}.`;
  if (value <= 0.85) return `${value.toFixed(1)}× the average from the ${label} — below it.`;
  return `About average for the ${label}.`;
}
