#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo 'Install FFmpeg: brew install ffmpeg'; exit 1; }
command -v ffprobe >/dev/null
xcrun swift --version
python3 -m venv .venv
swift build --package-path coreml -c release
.venv/bin/python src/pipeline.py --check
