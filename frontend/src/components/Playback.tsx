/**
 * Playback of what followed a selected frame.
 *
 * Freeze frames only exist at events, so the ball path is animated between event
 * locations and the players are drawn where they were at the start frame. Nothing
 * here invents player movement that the data does not contain.
 */

import { useEffect, useRef, useState } from 'react';
import type { Sequence, SituationDetail } from '../lib/types';

const STEP_MS = 900;

export interface PlaybackProps {
  detail: SituationDetail | null;
  sequence: Sequence | null;
  onFrame: (trail: { x: number; y: number }[] | null) => void;
}

export function Playback({ detail, sequence, onFrame }: PlaybackProps) {
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef<number | null>(null);

  const points = (sequence?.events ?? [])
    .filter((e) => e.x !== null && e.y !== null)
    .map((e) => ({ x: e.x as number, y: e.y as number }));

  useEffect(() => {
    setStep(points.length ? 1 : 0);
    setPlaying(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sequence?.event_id]);

  useEffect(() => {
    onFrame(points.length ? points.slice(0, Math.max(1, step)) : null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, sequence?.event_id]);

  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setStep((current) => {
        if (current >= points.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, STEP_MS);
    return () => {
      if (timer.current !== null) window.clearInterval(timer.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, points.length]);

  if (!detail) {
    return (
      <p className="empty">
        Pick a result to see the situation on the pitch and what followed it.
      </p>
    );
  }

  const events = sequence?.events ?? [];

  return (
    <section className="playback" aria-label="Playback" data-testid="playback">
      <header className="playback-head">
        <div>
          <strong>{detail.match}</strong>
          <span className="muted">
            {' '}
            {detail.competition} {detail.season} · {detail.minute}&apos; · {detail.team} in
            possession
          </span>
        </div>
        <div className="playback-controls">
          <button onClick={() => setPlaying((p) => !p)} disabled={points.length === 0}>
            {playing ? 'Pause' : 'Play'}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(1, points.length)}
            value={step}
            onChange={(event) => {
              setPlaying(false);
              setStep(Number(event.target.value));
            }}
            aria-label="Playback position"
          />
          <span className="muted">
            {step}/{points.length}
          </span>
        </div>
      </header>

      <ol className="sequence">
        {events.map((event, index) => (
          <li
            key={event.event_id}
            className={index < step ? 'played' : ''}
            onMouseEnter={() => setStep(index + 1)}
          >
            <span className="seq-time">+{event.second_offset.toFixed(1)}s</span>
            <span className="seq-type">{event.type}</span>
            <span className="muted">{event.player ?? event.team}</span>
            {event.xg !== null && <span className="badge shot">xG {event.xg.toFixed(2)}</span>}
            {event.goal && <span className="badge goal">goal</span>}
          </li>
        ))}
        {events.length === 0 && <li className="muted">No events followed inside the window.</li>}
      </ol>
    </section>
  );
}
