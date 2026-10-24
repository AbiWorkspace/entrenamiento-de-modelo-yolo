#!/usr/bin/env bash
# etiquetar.sh — Lanzador rápido para el etiquetador manual de YOLO VBT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$HOME/.venv-rocm/bin/python" ]; then
    PY="$HOME/.venv-rocm/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PY="$SCRIPT_DIR/.venv/bin/python"
else
    PY="python3"
fi

exec "$PY" "$SCRIPT_DIR/etiquetar.py" "$@"
