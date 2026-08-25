#!/bin/zsh
set -eu

SCRIPT_DIR=${0:A:h}
cd "$SCRIPT_DIR"

PORT=${PORT:-8766}
URL="http://127.0.0.1:${PORT}/"

if curl --fail --silent --show-error "$URL" >/dev/null 2>&1; then
  open "$URL"
  exit 0
fi

(
  for _ in {1..100}; do
    if curl --fail --silent "$URL" >/dev/null 2>&1; then
      open "$URL"
      exit 0
    fi
    sleep 0.1
  done
  print -u2 "Architecture Atlas did not become ready at $URL"
) &

print "Starting UniLab Architecture Atlas at $URL"
print "Keep this terminal window open; press Ctrl-C to stop it."
exec node auxiliary/atlas_app/serve_architecture.mjs
