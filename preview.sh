#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python src/pipeline.py --input "${1:-input/testclip_01.mov.mp4}" "${@:2}"
