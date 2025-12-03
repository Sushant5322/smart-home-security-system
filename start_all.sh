#!/usr/bin/env bash
# start_all.sh — start UI and recognizer for presentation (macOS)

set -e
PROJECT_DIR="/Users/apple/smart-home-automation/smart-home-automation"
VENV="$PROJECT_DIR/venv"

cd "$PROJECT_DIR"

# Activate venv
source "$VENV/bin/activate"

# Start UI server in background
python -u ui/app.py 2>&1 | tee server_log.txt &
UI_PID=$!

sleep 1

# Open dashboard in browser
open "http://127.0.0.1:5000/"

# Start recognizer in background
python recognize_face.py 2>&1 | tee recognizer_log.txt &
REC_PID=$!

echo "UI PID: $UI_PID"
echo "Recognizer PID: $REC_PID"
echo "To stop: ./stop_all.sh"
