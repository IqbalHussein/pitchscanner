/** Presets, so a first-time visitor has something to click immediately. */

import type { SketchPlayer } from './types';

export interface Preset {
  id: string;
  name: string;
  description: string;
  ball: { x: number; y: number };
  players: Omit<SketchPlayer, 'id'>[];
}

const p = (x: number, y: number, side: SketchPlayer['side'], keeper = false) => ({
  x,
  y,
  side,
  keeper,
});

export const PRESETS: Preset[] = [
  {
    id: '3v2-counter',
    name: '3v2 counter',
    description: 'Three attackers running at two defenders after a turnover.',
    ball: { x: 84, y: 22 },
    players: [
      p(86, 20, 'attacker'),
      p(95, 35, 'attacker'),
      p(78, 30, 'attacker'),
      p(92, 25, 'defender'),
      p(100, 30, 'defender'),
      p(119, 40, 'defender', true),
    ],
  },
  {
    id: 'high-press',
        name: 'High press',
    description: 'Ball trapped near the touchline with the press closing in.',
    ball: { x: 35, y: 12 },
    players: [
      p(37, 14, 'attacker'),
      p(45, 25, 'attacker'),
      p(30, 30, 'attacker'),
      p(33, 10, 'defender'),
      p(40, 18, 'defender'),
      p(28, 22, 'defender'),
      p(48, 14, 'defender'),
    ],
  },
  {
    id: 'low-block',
    name: 'Low block',
    description: 'Possession outside a packed penalty area.',
    ball: { x: 88, y: 40 },
    players: [
      p(90, 38, 'attacker'),
      p(96, 18, 'attacker'),
      p(96, 62, 'attacker'),
      p(104, 30, 'defender'),
      p(104, 40, 'defender'),
      p(104, 50, 'defender'),
      p(110, 34, 'defender'),
      p(110, 46, 'defender'),
      p(118, 40, 'defender', true),
    ],
  },
  {
    id: 'wide-overload',
    name: 'Wide overload',
    description: 'Two-on-one in the corner of the box.',
    ball: { x: 100, y: 8 },
    players: [
      p(102, 10, 'attacker'),
      p(108, 24, 'attacker'),
      p(104, 14, 'defender'),
      p(110, 30, 'defender'),
    ],
  },
];
