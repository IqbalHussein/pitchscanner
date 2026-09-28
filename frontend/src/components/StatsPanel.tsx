/**
 * What these situations usually led to, with intervals drawn as bars.
 *
 * Every number carries its uncertainty and its zone baseline, because a shot
 * rate on its own tells you almost nothing.
 */

import type { Interval, Stats } from '../lib/types';
import { baselineSentence, pct, ratio, xg } from '../lib/format';

interface RowProps {
  label: string;
  interval: Interval;
  format: (value: number) => string;
  hint?: string;
}

function StatRow({ label, interval, format, hint }: RowProps) {
  const [lo, hi] = interval.ci95;
  // The bar's scale is the widest thing it has to show, so the interval is never
  // clipped and never looks narrower than it is.
  const scale = Math.max(hi, interval.value, 0.0001);
  const left = (lo / scale) * 100;
  const width = Math.max(1.5, ((hi - lo) / scale) * 100);
  const point = (interval.value / scale) * 100;

  return (
    <div className="stat" data-testid={`stat-${label.toLowerCase().replace(/\s/g, '-')}`}>
      <div className="stat-head">
        <span className="stat-label">{label}</span>
        <span className="stat-value">{format(interval.value)}</span>
        <span className="stat-ratio" title="versus the average for this pitch zone">
          {ratio(interval.vs_zone_baseline)}
        </span>
      </div>
      <div className="stat-bar" role="img" aria-label={`${label} ${format(interval.value)}, 95% interval ${format(lo)} to ${format(hi)}`}>
        <span className="stat-ci" style={{ left: `${left}%`, width: `${width}%` }} />
        <span className="stat-point" style={{ left: `${point}%` }} />
      </div>
      <div className="stat-foot muted">
        95%: {format(lo)} – {format(hi)}
        {hint ? ` · ${hint}` : ''}
      </div>
    </div>
  );
}

export function StatsPanel({ stats }: { stats: Stats | null }) {
  if (!stats) return null;

  return (
    <section
      className={`stats ${stats.low_confidence ? 'low-confidence' : ''}`}
      aria-label="Outcome statistics"
      data-testid="stats"
    >
      <header className="stats-head">
        <h2>What happened next</h2>
        <p className="muted">
          {stats.n_possessions} distinct possessions
          {' · '}
          <span title="Kish effective sample size: similarity weighting means the closest matches count for more, so the sample carries less information than its size suggests">
            {stats.effective_n.toFixed(1)} effective
          </span>
          {' · '}
          baseline from {stats.baseline_n} frames in the {stats.zone_label}
        </p>
      </header>

      {stats.low_confidence && (
        <p className="notice" data-testid="low-confidence">
          Fewer than 30 possessions matched, so these rates are indicative only. The intervals
          below show how little the sample pins down.
        </p>
      )}

      <StatRow label="Shot rate" interval={stats.shot_rate} format={pct} />
      <StatRow
        label="Average xG"
        interval={stats.xg_mean}
        format={xg}
        hint={baselineSentence(stats.xg_mean.vs_zone_baseline, stats.zone_label)}
      />
      <StatRow label="Turnover rate" interval={stats.turnover_rate} format={pct} />
      <StatRow label="Box entry rate" interval={stats.box_entry_rate} format={pct} />

      <details className="secondary">
        <summary>Goal rate (noisy at this sample size)</summary>
        <StatRow label="Goal rate" interval={stats.goal_rate} format={pct} />
        <p className="muted">
          Goals are rare, so a rate from a few dozen plays is mostly noise. Read the xG instead.
        </p>
      </details>

      <details className="secondary">
        <summary>Unweighted averages</summary>
        <ul className="plain">
          {Object.entries(stats.unweighted).map(([key, value]) => (
            <li key={key}>
              <span className="muted">{key.replace(/_/g, ' ')}</span>
              <span>{typeof value === 'number' ? value.toFixed(3) : value}</span>
            </li>
          ))}
        </ul>
      </details>

      {stats.notes.map((note) => (
        <p key={note} className="muted note">
          {note}
        </p>
      ))}
    </section>
  );
}
