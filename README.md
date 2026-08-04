# hermes-graphiti-memory

Standalone Hermes Agent memory provider for a Graphiti MCP server. It exposes two model tools:

- `graphiti_memory`: direct Graphiti operations (`status`, `search_facts`, `search_nodes`, `get_episodes`, `add_episode`, failed-write spool/replay, and delete/get helpers).
- `recall`: unified explicit recall across Graphiti facts/nodes and Hermes `session_search`, with `sources`, `depth`, `budget`, `provenance`, and `role_filter`. Session id and Hermes home are trusted runtime values injected by Hermes, not public tool arguments.

The implementation is based on the production Graphiti provider behavior, but is packaged as an isolated repo. It does not copy secrets and defaults to no URL unless configured.

## Compatibility

Version `0.1.x` supports the standalone memory-provider API in Hermes Agent
v0.20 / `v2026.8.3` (validated against the reviewed local-capability tree
`6eb7b0979ccc002c7adb4e9952ef6e1dd8da15ab`).

Hermes v0.20 has three distinct compatibility/health signals:

1. **Plugin registry:** Graphiti is an `exclusive` provider plugin, so a registry
   listing may correctly show `runtime_enabled=false`. This means the generic
   plugin runtime will not also run its hooks; it does **not** mean that the
   selected memory provider is inactive.
2. **Memory-provider activation:** `memory.provider: graphiti` is the setting
   that selects and initializes Graphiti for Hermes memory lifecycle calls.
   Provider discovery/availability is the relevant activation check.
3. **Backend end-to-end:** only a successful real MCP `get_status` plus a
   read-only search proves that Hermes can reach the configured Graphiti/Neo4j
   backend. Registry discovery or provider construction alone is not backend
   E2E proof.

## Install

```bash
# Install the standalone repository through Hermes
hermes plugins install xiaoyaner0201/hermes-graphiti-memory

# Or copy/symlink a checkout as the active profile's user memory plugin
# Hermes discovers user-installed memory providers under $HERMES_HOME/plugins/<name>.
mkdir -p "$HERMES_HOME/plugins"
ln -s "$PWD" "$HERMES_HOME/plugins/graphiti"
hermes config set memory.provider graphiti
```

The bundled `scripts/install.sh` creates the same symlink. If `$HERMES_HOME/plugins/graphiti` already exists as a real directory or regular file, the script refuses to replace it. Back up or migrate that existing plugin first, for example:

```bash
mv "$HERMES_HOME/plugins/graphiti" "$HERMES_HOME/plugins/graphiti.backup.$(date +%Y%m%d%H%M%S)"
bash scripts/install.sh
```

## Configuration

Use environment variables or `plugins.graphiti` in `config.yaml`:

```yaml
plugins:
  graphiti:
    url: https://graphiti.example.com/mcp
    group_id: xiaoyaner-core
    timeout: 180
    # Graphiti search may take tens of seconds. Async prefetch keeps the user
    # turn responsive and makes the warmed result available to the next turn.
    prefetch_mode: async
    sync_prefetch_timeout: 2.5
    prefetch_limit: 6
    auto_sync_turns: true
    session_end_episode: true
    retry_failed_on_start: true
    max_failed_replay_per_start: 20
```

The four lifecycle settings above enable normal runtime writes and bounded
startup replay. Set the write features deliberately for each deployment. The
read-only smoke overrides all four in memory (`auto_sync_turns=false`,
`session_end_episode=false`, `retry_failed_on_start=false`, replay limit `0`)
without changing `config.yaml`.

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

`readonly_smoke.py` only calls MCP `get_status` and `search_memory_facts`. It
does not call provider `initialize()`, `shutdown()`, sync/session-end hooks,
replay, queue, spool, or any mutation tool. In particular, it cannot replay
pending failed writes and does not alter the failed-write spool.
