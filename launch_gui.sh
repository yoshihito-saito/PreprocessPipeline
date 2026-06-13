#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python launch_gui.py "$@"
