#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"
. .venv/bin/activate
python eval/run_all.py "$@"
