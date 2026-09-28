# Media

`demo.gif` and `grafana.png` are referenced by the root README and are not
checked in — they are recordings of a local run, and binaries that change every
time they are re-recorded do not belong in git history.

To record the demo GIF:

1. `make up` (or `make ingest && make serve && make web`).
2. Open the app, pick the "counter-attack" preset, drag two defenders, let the
   results and the statistics panel settle.
3. Capture the window and save it here as `demo.gif`, under about 5 MB.

To capture the dashboard screenshot:

1. `make up && make monitoring`.
2. Drive some traffic — a few dozen searches through the UI, or
   `make bench-quick` against the API — so the panels are not empty.
3. Open http://localhost:3000, the "Match Situation Search Engine" dashboard,
   and save the full dashboard here as `grafana.png`.
