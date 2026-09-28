import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const API = process.env.VITE_API_URL ?? 'http://127.0.0.1:8000';

// Both the dev server and `vite preview` proxy the API, so the app always talks
// to same-origin paths: no base URL is baked into the bundle, and the same build
// runs in dev, in the end-to-end tests and behind a reverse proxy in production.
const proxy = {
  '/v1': { target: API, changeOrigin: true },
  '/healthz': { target: API },
  '/readyz': { target: API },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: { outDir: 'dist', sourcemap: true },
});
