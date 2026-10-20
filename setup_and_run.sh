#!/usr/bin/env bash
# One-step setup and run on macOS/Linux. Needs Python 3.10 or newer.
set -e
cd "$(dirname "$0")/backend"
[ -x .venv/bin/python ] || python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
python scripts/ensure_model.py
python -m pytest -q || echo "Some tests failed. See the output above."
export STATIC_DIR="$(cd ../frontend/dist-local && pwd)"
echo "Open http://localhost:8000 (Ctrl+C to stop)"
python -m uvicorn app.api:app --port 8000
