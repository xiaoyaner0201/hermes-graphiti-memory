from __future__ import annotations
import json
import sys
import types
from pathlib import Path

import pytest

from __init__ import GraphitiMemoryProvider, _ensure_mcp_url


def test_ensure_mcp_url_canonicalizes_trailing_slash():
    assert _ensure_mcp_url("https://example.com") == "https://example.com/mcp"
    assert _ensure_mcp_url("https://example.com/mcp/") == "https://example.com/mcp"
    assert _ensure_mcp_url("") == ""


def test_tool_schemas_expose_graphiti_memory_and_recall():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    schemas = {s["name"]: s for s in p.get_tool_schemas()}
    assert {"graphiti_memory", "recall"}.issubset(schemas)
    recall_props = schemas["recall"]["parameters"]["properties"]
    assert "sources" in recall_props
    assert "session_id" in recall_props
    assert "hermes_home" in recall_props
    assert "dispatch" in recall_props


def test_initialize_records_profile_scoped_hermes_home(tmp_path):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "retry_failed_on_start": False}}})
    p._call_tool_sync = lambda tool, args: {"ok": True}
    p.initialize("sid-1", platform="cli", hermes_home=str(tmp_path), user_id="u")
    try:
        assert p._hermes_home == str(tmp_path)
        assert p._session_id == "sid-1"
        assert p._user_id == "u"
    finally:
        p.shutdown()


def test_sync_turn_accepts_messages_and_enqueues_clean_episode():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "auto_sync_turns": True}}})
    queued = []
    p._enqueue_add_memory = lambda name, body, desc, uuid=None: queued.append((name, body, desc, uuid))
    p.sync_turn("hello <memory-context>secret</memory-context> world", "assistant reply", session_id="s", messages=[{"role": "user", "content": "x"}])
    assert queued
    assert "secret" not in queued[0][1]
    assert "session_id=s" in queued[0][2]


def test_graphiti_memory_search_facts_uses_group_and_limit():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "group_id": "core", "prefetch_limit": 6}}})
    calls = []
    p._call_tool_sync = lambda tool, args: calls.append((tool, args)) or {"facts": [{"fact": "ok"}]}
    out = json.loads(p.handle_tool_call("graphiti_memory", {"action": "search_facts", "query": "q", "max_facts": 2}))
    assert out["facts"][0]["fact"] == "ok"
    assert calls == [("search_memory_facts", {"query": "q", "group_ids": ["core"], "max_facts": 2})]


def test_recall_fuses_graph_and_session_search(monkeypatch):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "group_id": "core"}}})
    p._build_recall_context_with_timeout = lambda query, timeout: "Graphiti recalled long-term memory:\nFacts:\n- remembered"

    mod = types.ModuleType("tools.session_search_tool")
    def fake_session_search(**kwargs):
        assert kwargs["query"] == "project"
        assert kwargs["limit"] == 2
        assert kwargs["current_session_id"] == "sid"
        return json.dumps({"success": True, "data": [{"session_id": "abc"}]})
    mod.session_search = fake_session_search
    sys.modules["tools.session_search_tool"] = mod

    out = json.loads(p.handle_tool_call("recall", {
        "query": "project",
        "sources": ["graph", "session_fts"],
        "budget": "small",
        "limit": 5,
        "session_id": "sid",
    }, db=object()))
    assert out["success"] is True
    assert out["sources"] == ["graph", "session_fts"]
    assert "remembered" in out["results"]["graph"]
    assert out["results"]["sessions"]["success"] is True
    assert len(out["recall_key"]) == 16


def test_recall_concurrent_dispatch_runs_both_sources(monkeypatch):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    seen = []
    p._build_recall_context_with_timeout = lambda query, timeout: seen.append("graph") or "g"
    mod = types.ModuleType("tools.session_search_tool")
    mod.session_search = lambda **kwargs: seen.append("session") or json.dumps({"success": True})
    sys.modules["tools.session_search_tool"] = mod
    out = json.loads(p.handle_tool_call("recall", {"query": "q", "sources": ["graph", "session_summary"], "dispatch": "concurrent"}, db=object()))
    assert out["dispatch"] == "concurrent"
    assert sorted(seen) == ["graph", "session"]


def test_failed_write_spool_and_replay(tmp_path):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "failed_write_dir": str(tmp_path)}}})
    path = p._spool_failed_write({"name": "n", "group_id": "g"}, RuntimeError("boom"))
    assert path and path.exists()
    assert p._list_failed_writes()["count"] == 1
    p._call_tool_sync = lambda tool, args: {"ok": True}
    replay = p._replay_failed_writes()
    assert replay["succeeded"] == 1
    assert p._list_failed_writes()["count"] == 0


def test_queue_prefetch_does_not_guess_or_rewrite_query():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    captured = []
    p._build_recall_context = lambda q: captured.append(q) or "ctx"
    p.queue_prefetch("exact query", session_id="s")
    # Wait briefly for daemon thread.
    import time
    for _ in range(20):
        if captured:
            break
        time.sleep(0.01)
    assert captured == ["exact query"]
    with p._prefetch_lock:
        assert p._prefetch_cache["s"] == "ctx"
