#!/usr/bin/env bash
# Bootstrap a venv, install dependencies, then hand over to app.py.
# Any arguments are passed straight through:  ./run.sh --port 8080 --reload
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
exec python app.py "$@"
