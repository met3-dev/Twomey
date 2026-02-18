#!/bin/bash
set -euo pipefail

# Only run in remote Claude Code environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install dependencies here as the project grows.
# Examples:
#   npm install          (Node.js)
#   pip install -e .     (Python)
#   bundle install       (Ruby)
#   cargo build          (Rust)

echo "Session start hook: no dependencies to install yet."
