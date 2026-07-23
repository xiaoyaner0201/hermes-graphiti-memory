# hermes-graphiti-memory

Standalone Hermes Agent memory provider for a Graphiti MCP server. It exposes two model tools:

- `graphiti_memory`: direct Graphiti operations (`status`, `search_facts`, `search_nodes`, `get_episodes`, `add_episode`, failed-write spool/replay, and delete/get helpers).
- `recall`: unified explicit recall across Graphiti facts/nodes and Hermes `session_search`, with `sources`, `depth`, `budget`, `provenance`, `role_filter`, `session_id`, `hermes_home`, and `dispatch=sequential|concurrent`.

The implementation is based on the production Graphiti provider behavior, but is packaged as an isolated repo. It does not copy secrets and defaults to no URL unless configured.

## Install

```bash
# From this repo
python -m pip install -e .

# Or copy/symlink this directory as the active profile's user memory plugin
# Hermes discovers user-installed memory providers under $HERMES_HOME/plugins/<name>.
mkdir -p "$HERMES_HOME/plugins"
ln -s "$PWD" "$HERMES_HOME/plugins/graphiti"
hermes config set memory.provider graphiti
```

## Configuration

Use environment variables or `plugins.graphiti` in `config.yaml`:

```yaml
plugins:
  graphiti:
    url: https://graphiti.example.com/mcp
    group_id: xiaoyaner-core
    timeout: 180
    prefetch_mode: hybrid
    sync_prefetch_timeout: 2.5
    prefetch_limit: 6
```

Environment overrides:

- `GRAPHITI_MCP_URL`
- `GRAPHITI_GROUP_ID`

Secrets belong in `.env`; do not commit tokens or copied production config.

## Verification

Run deterministic tests:

```bash
python -m pytest -q
```

Optional read-only smoke against a configured production Graphiti MCP server:

```bash
GRAPHITI_MCP_URL=... python scripts/readonly_smoke.py
```

`readonly_smoke.py` only calls status/search operations and does not add test memories.
