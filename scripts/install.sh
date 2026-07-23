#!/usr/bin/env bash
set -euo pipefail
: "${HERMES_HOME:=${HOME}/.hermes}"
mkdir -p "$HERMES_HOME/plugins"
ln -sfn "$(pwd)" "$HERMES_HOME/plugins/graphiti"
echo "Installed graphiti memory plugin symlink at $HERMES_HOME/plugins/graphiti"
echo "Set memory.provider=graphiti and configure GRAPHITI_MCP_URL before use."
