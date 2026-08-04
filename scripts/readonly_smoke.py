#!/usr/bin/env python3
"""Read-only Graphiti MCP smoke for the standalone provider.

This probe deliberately bypasses the provider lifecycle. In particular, it does
not call ``initialize()`` because initialization starts the writer and may replay
failed writes from the spool.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from __init__ import GraphitiMemoryProvider, _load_config

_READ_ONLY_SETTINGS = {
    "auto_sync_turns": False,
    "session_end_episode": False,
    "retry_failed_on_start": False,
    "max_failed_replay_per_start": 0,
}


def create_readonly_provider(config: dict[str, Any] | None = None) -> GraphitiMemoryProvider:
    """Create a provider with every automatic write/replay path disabled."""
    root_config = copy.deepcopy(config if config is not None else _load_config())
    plugins = root_config.setdefault("plugins", {})
    graphiti = plugins.get("graphiti")
    if not isinstance(graphiti, dict) or not graphiti:
        memory = root_config.get("memory") or {}
        legacy = memory.get("graphiti") if isinstance(memory, dict) else {}
        graphiti = copy.deepcopy(legacy) if isinstance(legacy, dict) else {}
        plugins["graphiti"] = graphiti
    graphiti.update(_READ_ONLY_SETTINGS)
    return GraphitiMemoryProvider(root_config)


def run_readonly_smoke(
    provider: GraphitiMemoryProvider,
    *,
    query: str = "Hermes Graphiti",
) -> dict[str, Any]:
    """Call only Graphiti's status and fact-search MCP tools.

    The caller must pass a provider created for read-only use. No lifecycle,
    replay, sync, shutdown, queue, spool, or mutation method is invoked here.
    """
    status = json.loads(
        provider.handle_tool_call("graphiti_memory", {"action": "status"})
    )
    facts = json.loads(
        provider.handle_tool_call(
            "graphiti_memory",
            {"action": "search_facts", "query": query, "max_facts": 1},
        )
    )
    return {
        "status_success": "error" not in status,
        "status_keys": sorted(status.keys())[:10]
        if isinstance(status, dict)
        else [],
        "facts_count": len(facts.get("facts") or [])
        if isinstance(facts, dict)
        else 0,
    }


def main() -> int:
    provider = create_readonly_provider()
    if not provider.is_available():
        raise SystemExit(
            "provider unavailable: configure GRAPHITI_MCP_URL or "
            "plugins.graphiti.url and install mcp SDK"
        )
    print(json.dumps(run_readonly_smoke(provider), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
