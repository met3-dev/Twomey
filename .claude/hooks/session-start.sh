#!/bin/bash
set -euo pipefail

# Only run in remote Claude Code environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "Alfred session start: installing dependencies..."
pip install -r "$CLAUDE_PROJECT_DIR/requirements.txt"
echo "Alfred session start: ready."
