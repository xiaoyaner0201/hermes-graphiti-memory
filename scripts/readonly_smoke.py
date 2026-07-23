#!/usr/bin/env python3
"""Read-only Graphiti MCP smoke for the standalone provider."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from __init__ import GraphitiMemoryProvider

override_url = os.environ.get("GRAPHITI_MCP_URL")
override_group = os.environ.get("GRAPHITI_GROUP_ID")
if override_url or override_group:
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": override_url or "", "group_id": override_group or "xiaoyaner-core"}}})
else:
    # Load profile-scoped Hermes config via hermes_constants/get_hermes_home.
    p = GraphitiMemoryProvider()
if not p.is_available():
    raise SystemExit("provider unavailable: configure GRAPHITI_MCP_URL or plugins.graphiti.url and install mcp SDK")
p.initialize("readonly-smoke", platform="cli", hermes_home=os.environ.get("HERMES_HOME", str(Path.home()/".hermes")))
try:
    status = json.loads(p.handle_tool_call("graphiti_memory", {"action": "status"}))
    facts = json.loads(p.handle_tool_call("graphiti_memory", {"action": "search_facts", "query": "Hermes Graphiti", "max_facts": 1}))
    print(json.dumps({
        "status_success": "error" not in status,
        "status_keys": sorted(status.keys())[:10] if isinstance(status, dict) else [],
        "facts_count": len(facts.get("facts") or []) if isinstance(facts, dict) else 0,
    }, ensure_ascii=False))
finally:
    p.shutdown()
