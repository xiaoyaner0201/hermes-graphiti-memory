#!/usr/bin/env bash
set -euo pipefail
: "${HERMES_HOME:=${HOME}/.hermes}"
mkdir -p "$HERMES_HOME/plugins"
TARGET="$HERMES_HOME/plugins/graphiti"
SOURCE="$(pwd)"
if [ -e "$TARGET" ] && [ ! -L "$TARGET" ]; then
  cat >&2 <<EOF
Refusing to replace existing non-symlink plugin path: $TARGET

Back up or migrate the existing plugin directory first, for example:
  mv "$TARGET" "$TARGET.backup.$(date +%Y%m%d%H%M%S)"
Then re-run scripts/install.sh from this repository.
EOF
  exit 1
fi
ln -sfn "$SOURCE" "$TARGET"
echo "Installed graphiti memory plugin symlink at $TARGET -> $SOURCE"
echo "Set memory.provider=graphiti and configure GRAPHITI_MCP_URL before use."
