/**
 * The demo has to make sense to a recruiter in 30 seconds without instructions:
 * load a preset, get results and statistics, and share the link. That is exactly
 * what these tests check.
 */

import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.goto('/');
});

test('loads with a preset already searched', async ({ page }) => {
  await expect(page.getByRole('heading', { name: 'Match Situation Search' })).toBeVisible();
  await expect(page.getByTestId('pitch')).toBeVisible();
  await expect(page.getByTestId('result').first()).toBeVisible();
  await expect(page.getByTestId('stats')).toBeVisible();
  await expect(page.getByTestId('coverage')).toContainText('frames');
});

test('statistics show a value, an interval and a baseline ratio', async ({ page }) => {
  const shot = page.getByTestId('stat-shot-rate');
  await expect(shot).toBeVisible();
  await expect(shot).toContainText('%');
  await expect(shot).toContainText('95%:');
  await expect(page.getByTestId('stat-average-xg')).toBeVisible();
});

test('switching preset changes the results', async ({ page }) => {
  const first = await page.getByTestId('result').first().innerText();
  await page.getByTestId('preset-low-block').click();
  await expect(async () => {
    const next = await page.getByTestId('result').first().innerText();
    expect(next).not.toBe(first);
  }).toPass();
});

test('a text filter becomes a chip and narrows the search', async ({ page }) => {
  await page.getByTestId('filter-text').fill('when losing in the second half');
  await expect(page.getByTestId('chips')).toContainText('Game state: losing');
  await expect(page.getByTestId('chips')).toContainText('Minutes: 45-98');
  await expect(page.getByTestId('result').first()).toBeVisible();
});

test('unrecognised words are reported rather than ignored', async ({ page }) => {
  await page.getByTestId('filter-text').fill('rainy games');
  await expect(page.getByTestId('unmatched')).toContainText('rainy');
});

test('placing a player on the pitch triggers a new search', async ({ page }) => {
  const before = await page.getByTestId('player-attacker').count();
  await page.getByTestId('mode-attacker').click();
  await page.getByTestId('pitch-surface').click({ position: { x: 120, y: 60 } });
  await expect(page.getByTestId('player-attacker')).toHaveCount(before + 1);
  await expect(page.getByTestId('result').first()).toBeVisible();
});

test('the sketch is in the URL and reloads identically', async ({ page }) => {
  await page.getByTestId('preset-wide-overload').click();
  await expect(page.getByTestId('result').first()).toBeVisible();
  const url = page.url();
  expect(url).toContain('b=');
  expect(url).toContain('p=');

  const players = await page.getByTestId('player-attacker').count();
  await page.goto(url);
  await expect(page.getByTestId('player-attacker')).toHaveCount(players);
});

test('selecting a result opens playback of what followed', async ({ page }) => {
  await page.getByTestId('result').first().click();
  await expect(page.getByTestId('playback')).toBeVisible();
  await expect(page.getByTestId('playback')).toContainText('in possession');
});

test('the timing readout reports the stage breakdown', async ({ page }) => {
  await expect(page.getByTestId('timings')).toContainText('encode');
  await expect(page.getByTestId('timings')).toContainText('aggregate');
});

test('small samples are marked low confidence', async ({ page }) => {
  await page.getByTestId('k-input').fill('5');
  await expect(page.getByTestId('low-confidence')).toBeVisible();
});

test('the pitch scales to phone width without horizontal scrolling', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByTestId('pitch')).toBeVisible();
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
  );
  expect(overflow).toBe(false);
});

test('the footer credits StatsBomb', async ({ page }) => {
  await expect(page.getByRole('contentinfo')).toContainText('StatsBomb');
});
