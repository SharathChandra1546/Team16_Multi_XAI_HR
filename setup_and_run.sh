#!/usr/bin/env bash
# Set up (if needed) and start the app on macOS/Linux. Needs Python 3.10 or newer.
#   ./setup_and_run.sh            set up and start on http://localhost:8000
#   ./setup_and_run.sh --test     also run the full test suite first (about 40 minutes)
#   ./setup_and_run.sh --config   choose the language model / API key, then start
set -e
cd "$(dirname "$0")/backend"
PY=python3
command -v python3 >/dev/null || PY=python
$PY -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' || { echo "Python 3.10 or newer is required."; exit 1; }
[ -x .venv/bin/python ] || $PY -m venv .venv
VPY=.venv/bin/python
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing Python packages (first time takes 3 to 10 minutes)..."
  $VPY -m pip install --upgrade pip -q --disable-pip-version-check
  $VPY -m pip install -r requirements.txt --disable-pip-version-check
  cp requirements.txt .venv/requirements.installed
fi
[ -f .env ] || cp .env.example .env
$VPY scripts/ensure_model.py
$VPY scripts/smoke_check.py
[ "$1" = "--test" ] && $VPY -m pytest -q
[ "$1" = "--config" ] && $VPY scripts/configure_env.py
$VPY scripts/configure_env.py --show
export STATIC_DIR="$(cd ../frontend/dist-local && pwd)"
echo "Open http://localhost:8000 (Ctrl+C to stop). Change the model with: ./setup_and_run.sh --config"
$VPY -m uvicorn app.api:app --host 127.0.0.1 --port 8000
