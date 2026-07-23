#!/usr/bin/env bash
set -euo pipefail
: "${HERMES_HOME:=${HOME}/.hermes}"
TARGET="$HERMES_HOME/plugins/graphiti"
if [ -L "$TARGET" ]; then
  rm "$TARGET"
  echo "Removed $TARGET"
else
  echo "$TARGET is not a symlink; refusing to remove" >&2
  exit 1
fi
