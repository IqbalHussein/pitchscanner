/**
 * The SVG pitch editor, in StatsBomb coordinates.
 *
 * Click to add, drag to move, right-click or long-press to delete, and arrow
 * keys to nudge a selected player. The SVG viewBox is the pitch itself, so
 * everything scales to phone width without recomputing a single coordinate.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import {
  PITCH_LENGTH,
  PITCH_WIDTH,
  type PointIn,
  type Side,
  type SketchPlayer,
} from '../lib/types';

const LONG_PRESS_MS = 500;

export interface PitchProps {
  ball: { x: number; y: number };
  players: SketchPlayer[];
  addSide: Side | 'ball';
  overlay?: {
    ball: PointIn;
    attackers: PointIn[];
    defenders: PointIn[];
    visibleArea: PointIn[] | null;
  } | null;
  trail?: { x: number; y: number }[] | null;
  onAdd: (x: number, y: number) => void;
  onMove: (id: string, x: number, y: number) => void;
  onMoveBall: (x: number, y: number) => void;
  onRemove: (id: string) => void;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function Pitch({
  ball,
  players,
  addSide,
  overlay,
  trail,
  onAdd,
  onMove,
  onMoveBall,
  onRemove,
}: PitchProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const longPress = useRef<number | null>(null);

  /** Screen coordinates to pitch coordinates, via the SVG's own matrix. */
  const toPitch = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return { x: 0, y: 0 };
    const point = svg.createSVGPoint();
    point.x = clientX;
    point.y = clientY;
    const matrix = svg.getScreenCTM();
    if (!matrix) return { x: 0, y: 0 };
    const local = point.matrixTransform(matrix.inverse());
    return {
      x: clamp(local.x, 0, PITCH_LENGTH),
      y: clamp(local.y, 0, PITCH_WIDTH),
    };
  }, []);

  useEffect(() => {
    if (!dragging) return;
    const onPointerMove = (event: PointerEvent) => {
      const { x, y } = toPitch(event.clientX, event.clientY);
      if (dragging === 'ball') onMoveBall(x, y);
      else onMove(dragging, x, y);
    };
    const stop = () => setDragging(null);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stop);
    window.addEventListener('pointercancel', stop);
    return () => {
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', stop);
      window.removeEventListener('pointercancel', stop);
    };
  }, [dragging, onMove, onMoveBall, toPitch]);

  const handleBackgroundClick = (event: React.MouseEvent<SVGRectElement>) => {
    const { x, y } = toPitch(event.clientX, event.clientY);
    if (addSide === 'ball') onMoveBall(x, y);
    else onAdd(x, y);
  };

  const startLongPress = (id: string) => {
    longPress.current = window.setTimeout(() => onRemove(id), LONG_PRESS_MS);
  };
  const cancelLongPress = () => {
    if (longPress.current !== null) {
      window.clearTimeout(longPress.current);
      longPress.current = null;
    }
  };

  const onKeyDown = (event: React.KeyboardEvent<SVGSVGElement>) => {
    if (!selected) return;
    const step = event.shiftKey ? 5 : 1;
    const target = players.find((p) => p.id === selected);
    if (!target) return;
    const moves: Record<string, [number, number]> = {
      ArrowLeft: [-step, 0],
      ArrowRight: [step, 0],
      ArrowUp: [0, -step],
      ArrowDown: [0, step],
    };
    if (event.key === 'Delete' || event.key === 'Backspace') {
      event.preventDefault();
      onRemove(selected);
      setSelected(null);
      return;
    }
    const move = moves[event.key];
    if (!move) return;
    event.preventDefault();
    onMove(
      selected,
      clamp(target.x + move[0], 0, PITCH_LENGTH),
      clamp(target.y + move[1], 0, PITCH_WIDTH),
    );
  };

  return (
    <svg
      ref={svgRef}
      className="pitch"
      viewBox={`-2 -2 ${PITCH_LENGTH + 4} ${PITCH_WIDTH + 4}`}
      role="application"
      aria-label="Pitch editor. Click to place players, drag to move, delete to remove."
      tabIndex={0}
      onKeyDown={onKeyDown}
      data-testid="pitch"
    >
      <rect
        x={0}
        y={0}
        width={PITCH_LENGTH}
        height={PITCH_WIDTH}
        className="pitch-grass"
        onClick={handleBackgroundClick}
        data-testid="pitch-surface"
      />

      {/* Markings. Attacking direction is always left to right. */}
      <g className="pitch-lines" pointerEvents="none">
        <rect x={0} y={0} width={PITCH_LENGTH} height={PITCH_WIDTH} />
        <line x1={60} y1={0} x2={60} y2={PITCH_WIDTH} />
        <circle cx={60} cy={40} r={10} />
        <circle cx={60} cy={40} r={0.6} className="pitch-dot" />
        <rect x={102} y={18} width={18} height={44} />
        <rect x={114} y={30} width={6} height={20} />
        <rect x={0} y={18} width={18} height={44} />
        <rect x={0} y={30} width={6} height={20} />
        <circle cx={108} cy={40} r={0.6} className="pitch-dot" />
        <circle cx={12} cy={40} r={0.6} className="pitch-dot" />
        <text x={110} y={6} className="pitch-hint">
          attacking →
        </text>
      </g>

      {/* A hovered result, drawn under the sketch. */}
      {overlay && (
        <g className="overlay" pointerEvents="none">
          {overlay.visibleArea && overlay.visibleArea.length > 2 && (
            <polygon
              className="overlay-area"
              points={overlay.visibleArea.map((p) => `${p.x},${p.y}`).join(' ')}
            />
          )}
          {overlay.attackers.map((p, i) => (
            <circle key={`oa${i}`} cx={p.x} cy={p.y} r={1.6} className="overlay-attacker" />
          ))}
          {overlay.defenders.map((p, i) => (
            <circle key={`od${i}`} cx={p.x} cy={p.y} r={1.6} className="overlay-defender" />
          ))}
          <circle cx={overlay.ball.x} cy={overlay.ball.y} r={1.1} className="overlay-ball" />
        </g>
      )}

      {/* Playback trail: the ball path after the selected frame. */}
      {trail && trail.length > 1 && (
        <g pointerEvents="none">
          <polyline className="trail" points={trail.map((p) => `${p.x},${p.y}`).join(' ')} />
          {trail.map((p, i) => (
            <circle key={`t${i}`} cx={p.x} cy={p.y} r={0.8} className="trail-dot" />
          ))}
        </g>
      )}

      {players.map((player) => (
        <g
          key={player.id}
          className={[
            'player',
            player.side === 'attacker' ? 'attacker' : 'defender',
            player.keeper ? 'keeper' : '',
            selected === player.id ? 'selected' : '',
          ]
            .join(' ')
            .trim()}
          transform={`translate(${player.x} ${player.y})`}
          onPointerDown={(event) => {
            event.preventDefault();
            setDragging(player.id);
            setSelected(player.id);
            startLongPress(player.id);
          }}
          onPointerUp={cancelLongPress}
          onPointerLeave={cancelLongPress}
          onContextMenu={(event) => {
            event.preventDefault();
            onRemove(player.id);
          }}
          data-testid={`player-${player.side}`}
        >
          <circle r={2.2} />
          {player.keeper && (
            <text className="player-label" y={0.9}>
              GK
            </text>
          )}
        </g>
      ))}

      <g
        className="ball"
        transform={`translate(${ball.x} ${ball.y})`}
        onPointerDown={(event) => {
          event.preventDefault();
          setDragging('ball');
        }}
        data-testid="ball"
      >
        <circle r={1.5} />
      </g>
    </svg>
  );
}
