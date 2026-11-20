#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  lab_python="$PWD/.venv/bin/python"
elif [ -x ../../work/venv/bin/python ]; then
  lab_python="$PWD/../../work/venv/bin/python"
else
  lab_python=""
  for candidate in python3.11 python3.12 python3.13 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      lab_python="$candidate"
      break
    fi
  done
  if [ -z "$lab_python" ]; then
    echo 'Install Python 3.11 or newer, then run this launcher again.'
    read -k 1
    exit 1
  fi
  "$lab_python" -m venv .venv
  lab_python="$PWD/.venv/bin/python"
  "$lab_python" -m pip install --upgrade pip swig
  PATH="$PWD/.venv/bin:$PATH" "$lab_python" -m pip install -e .
fi
echo 'Open http://127.0.0.1:8765 in your browser. Keep this window open.'
"$lab_python" -m rocketlab.cli serve --runs runs
