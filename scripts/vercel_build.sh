#!/usr/bin/env bash
# buildCommand for Vercel: the frontend, then the data the Python function serves.
#
# The function reads api/_data (see api/index.py). What goes there:
#   * data/release.json pins a data release  -> download it, verify the sha256 and
#     stream-unpack it (scripts/data_release.py, standard library only).
#   * data/release.json is null              -> build a small array store from the
#     checked-in fixtures, so previews and a fresh fork run the same arrays code path
#     as production instead of a different one. Costs a venv and about 15 seconds.
# Files created here reach the function bundle through vercel.json "includeFiles".
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$PWD
DATA="$ROOT/api/_data"

echo "== frontend"
(cd frontend && npm run gen:api && npm run build)

echo "== data"
rm -rf "$DATA" "$DATA.partial"

# Newest interpreter that satisfies backend/pyproject.toml (>=3.12) first.
PY=""
for cand in python3.13 python3.12 python3; do
  if command -v "$cand" >/dev/null 2>&1; then PY=$cand; break; fi
done
[ -n "$PY" ] || { echo "no python3 on this build image" >&2; exit 1; }
echo "python: $($PY --version) at $(command -v "$PY")"

status=0
"$PY" -I scripts/data_release.py fetch data/release.json "$DATA" || status=$?

if [ "$status" -eq 10 ]; then
  echo "no data release pinned: building the fixture array store"
  "$PY" - <<'PYCHECK' || { echo "need python >= 3.12 to build the fixture store" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 12) else 1)
PYCHECK
  VENV="$(mktemp -d)/venv"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --disable-pip-version-check -r requirements.txt
  PYTHONPATH="$ROOT/backend" "$VENV/bin/python" -m pitchscanner.cli --store arrays ingest \
    --fixtures --encodings E1,E2 --out "$DATA" --reject-log "$(dirname "$VENV")/rejects.jsonl" \
    > "$(dirname "$VENV")/ingest.log"
  tail -n 25 "$(dirname "$VENV")/ingest.log"
elif [ "$status" -ne 0 ]; then
  echo "data release download failed (status $status)" >&2
  exit "$status"
fi

[ -f "$DATA/meta.json" ] || { echo "api/_data/meta.json is missing after the data step" >&2; exit 1; }
du -sh "$DATA" || true
echo "== done"
