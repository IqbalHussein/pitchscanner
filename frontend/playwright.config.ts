import { defineConfig, devices } from '@playwright/test';

/**
 * End-to-end tests run against the composed stack: the API serving the fixture
 * dataset and the built frontend. `webServer` starts both, so `npm run test:e2e`
 * works from a clean checkout and CI needs no extra orchestration.
 */
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 45_000,
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://127.0.0.1:4173',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile', use: { ...devices['Pixel 5'] } },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : [
        {
          command:
            'cd ../backend && PITCHSCANNER_STORE=memory python -m pitchscanner.cli --store memory serve --port 8000',
          url: 'http://127.0.0.1:8000/readyz',
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
        {
          command: 'npm run preview',
          url: 'http://127.0.0.1:4173',
          reuseExistingServer: !process.env.CI,
          timeout: 120_000,
        },
      ],
});
