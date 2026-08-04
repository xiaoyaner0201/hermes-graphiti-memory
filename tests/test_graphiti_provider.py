from __future__ import annotations
import json
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

from __init__ import GraphitiMemoryProvider, _ensure_mcp_url, _streamable_http_client_factory
from scripts import readonly_smoke


def test_ensure_mcp_url_canonicalizes_trailing_slash():
    assert _ensure_mcp_url("https://example.com") == "https://example.com/mcp"
    assert _ensure_mcp_url("https://example.com/mcp/") == "https://example.com/mcp"
    assert _ensure_mcp_url("") == ""


def test_installed_mcp_streamable_http_client_symbol_is_supported():
    factory, requires_http_client = _streamable_http_client_factory()
    assert callable(factory)
    assert isinstance(requires_http_client, bool)


def test_readonly_smoke_main_fails_when_status_probe_fails(monkeypatch, capsys):
    class Provider:
        def is_available(self):
            return True

    monkeypatch.setattr(readonly_smoke, "create_readonly_provider", Provider)
    monkeypatch.setattr(
        readonly_smoke,
        "run_readonly_smoke",
        lambda provider: {"status_success": False, "status_keys": ["error"], "facts_count": 0},
    )

    assert readonly_smoke.main() == 1
    assert '"status_success": false' in capsys.readouterr().out


def test_tool_schemas_expose_graphiti_memory_and_recall():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    schemas = {s["name"]: s for s in p.get_tool_schemas()}
    assert {"graphiti_memory", "recall"}.issubset(schemas)
    recall_props = schemas["recall"]["parameters"]["properties"]
    assert "sources" in recall_props
    assert "session_id" not in recall_props
    assert "hermes_home" not in recall_props
    assert "dispatch" not in recall_props


def test_recall_ignores_untrusted_session_id_and_hermes_home_args(monkeypatch, tmp_path):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "retry_failed_on_start": False}}})
    p._call_tool_sync = lambda tool, args: {"ok": True}
    p.initialize("trusted-sid", platform="cli", hermes_home=str(tmp_path / "trusted"))
    try:
        opened = []
        state_mod = types.ModuleType("hermes_state")

        class FakeSessionDB:
            def __init__(self, hermes_home=None):
                opened.append(hermes_home)

        setattr(state_mod, "SessionDB", FakeSessionDB)
        sys.modules["hermes_state"] = state_mod
        search_mod = types.ModuleType("tools.session_search_tool")

        def fake_session_search(**kwargs):
            assert kwargs["current_session_id"] == "trusted-sid"
            return json.dumps({"success": True, "data": []})

        setattr(search_mod, "session_search", fake_session_search)
        sys.modules["tools.session_search_tool"] = search_mod

        out = json.loads(p.handle_tool_call("recall", {
            "query": "project",
            "sources": ["session_fts"],
            "session_id": "attacker-sid",
            "hermes_home": str(tmp_path / "attacker"),
        }))
        assert out["success"] is True
        assert opened == [str(tmp_path / "trusted")]
    finally:
        p.shutdown()


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


def test_graphiti_memory_get_episodes_translates_to_current_mcp_schema():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "group_id": "core"}}})
    calls = []
    p._call_tool_sync = lambda tool, args: calls.append((tool, args)) or {"episodes": []}

    out = json.loads(p.handle_tool_call("graphiti_memory", {
        "action": "get_episodes",
        "group_id": "isolated",
        "last_n": 7,
    }))

    assert out == {"episodes": []}
    assert calls == [("get_episodes", {"group_ids": ["isolated"], "max_episodes": 7})]


def test_recall_fuses_graph_and_session_search(monkeypatch):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "group_id": "core"}}})
    p._graph_recall_for_tool = lambda query, depth, provenance: ("Graphiti recalled long-term memory:\nFacts:\n- remembered", {}, None)

    mod = types.ModuleType("tools.session_search_tool")
    def fake_session_search(**kwargs):
        assert kwargs["query"] == "project"
        assert kwargs["limit"] == 2
        assert kwargs["current_session_id"] == "sid"
        return json.dumps({"success": True, "data": [{"session_id": "abc"}]})
    setattr(mod, "session_search", fake_session_search)
    sys.modules["tools.session_search_tool"] = mod

    out = json.loads(p.handle_tool_call("recall", {
        "query": "project",
        "sources": ["graph", "session_fts"],
        "budget": "small",
        "limit": 5,
    }, db=object(), current_session_id="sid"))
    assert out["success"] is True
    assert out["sources"] == ["graph", "session_fts"]
    assert "remembered" in out["results"]["graph"]
    assert out["results"]["sessions"]["success"] is True
    assert len(out["recall_key"]) == 16


def test_recall_propagates_nested_session_failure(monkeypatch):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    mod = types.ModuleType("tools.session_search_tool")
    setattr(mod, "session_search", lambda **kwargs: json.dumps({
        "success": False,
        "error": "database unavailable",
    }))
    sys.modules["tools.session_search_tool"] = mod

    out = json.loads(p.handle_tool_call("recall", {
        "query": "project",
        "sources": ["session_fts"],
    }, db=object(), current_session_id="sid"))

    assert out["success"] is False
    assert out["errors"] == ["session recall failed: database unavailable"]


def test_recall_deep_graph_includes_facts_and_nodes(monkeypatch):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    calls = []

    async def fake_call(tool, args, *, timeout=None):
        calls.append((tool, args, timeout))
        if tool == "search_memory_facts":
            return {"facts": [{"uuid": "f1", "fact": "remembered fact", "valid_at": "2026-01-01"}]}
        if tool == "search_nodes":
            return {"nodes": [{"uuid": "n1", "name": "Node", "summary": "remembered node"}]}
        return {}

    p._call_tool = fake_call
    out = json.loads(p.handle_tool_call("recall", {"query": "q", "depth": "deep", "sources": ["graph"], "provenance": "ids"}, db=object()))
    assert out["success"] is True
    assert "remembered fact" in out["results"]["graph"]
    assert "remembered node" in out["results"]["graph"]
    assert out["provenance_details"]["graph"] == {"facts": ["f1"], "nodes": ["n1"]}
    assert {c[0] for c in calls} == {"search_memory_facts", "search_nodes"}


def test_recall_surfaces_structured_mcp_error_instead_of_silent_empty_success():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "timeout": 90}}})

    async def fake_call(tool, args, *, timeout=None):
        return {"error": "Fact search timed out after 57.0s"}

    p._call_tool = fake_call
    out = json.loads(p.handle_tool_call("recall", {
        "query": "project decision",
        "depth": "standard",
        "sources": ["graph"],
    }, db=object()))

    assert out["success"] is False
    assert out["results"]["graph"] == ""
    assert out["errors"] == ["graph recall failed: facts: Fact search timed out after 57.0s"]


def test_standard_explicit_recall_allows_observed_graphiti_latency():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {
        "url": "http://g/mcp",
        "timeout": 90,
        "sync_prefetch_timeout": 2.5,
    }}})
    calls = []

    async def fake_call(tool, args, *, timeout=None):
        calls.append((tool, args, timeout))
        return {"facts": [{"fact": "remembered"}]}

    p._call_tool = fake_call
    out = json.loads(p.handle_tool_call("recall", {
        "query": "project decision",
        "depth": "standard",
        "sources": ["graph"],
    }, db=object()))

    assert out["success"] is True
    assert calls[0][2] == 75.0
    assert calls[0][1]["timeout_seconds"] == 72.0


def test_background_prefetch_uses_full_provider_timeout():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {
        "url": "http://g/mcp",
        "timeout": 90,
        "sync_prefetch_timeout": 2.5,
    }}})
    calls = []
    p.recall_graph = lambda query, *, include_nodes=False, timeout=None: calls.append(
        (query, include_nodes, timeout)
    ) or "context"

    assert p._build_recall_context("project") == "context"
    assert calls == [("project", False, 90.0)]


def test_failed_write_spool_and_replay(tmp_path):
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "failed_write_dir": str(tmp_path)}}})
    path = p._spool_failed_write({"name": "n", "group_id": "g"}, RuntimeError("boom"))
    assert path and path.exists()
    assert p._list_failed_writes()["count"] == 1
    p._call_tool_sync = lambda tool, args: {"ok": True}
    replay = p._replay_failed_writes()
    assert replay["succeeded"] == 1
    assert p._list_failed_writes()["count"] == 0


def test_readonly_smoke_provider_disables_all_automatic_write_paths(tmp_path):
    p = readonly_smoke.create_readonly_provider({
        "plugins": {
            "graphiti": {
                "url": "http://g/mcp",
                "failed_write_dir": str(tmp_path),
                "auto_sync_turns": True,
                "session_end_episode": True,
                "retry_failed_on_start": True,
                "max_failed_replay_per_start": 20,
            }
        }
    })

    assert p._auto_sync is False
    assert p._session_end_enabled is False
    assert p._retry_failed_on_start is False
    assert p._max_failed_replay_per_start == 0


def test_readonly_smoke_preserves_legacy_memory_graphiti_connection_config():
    p = readonly_smoke.create_readonly_provider({
        "memory": {
            "graphiti": {
                "url": "http://legacy/mcp",
                "group_id": "legacy-core",
                "headers": {"Authorization": "redacted-test-value"},
            }
        }
    })

    assert p._url == "http://legacy/mcp"
    assert p._group_id == "legacy-core"
    assert p._headers == {"Authorization": "redacted-test-value"}
    assert p._retry_failed_on_start is False


def test_readonly_smoke_never_initializes_replays_writes_or_changes_spool(tmp_path):
    spool = tmp_path / "spool"
    spool.mkdir()
    pending = spool / "pending.json"
    pending.write_bytes(b'{"pending": true}\n')
    before = {path.relative_to(spool): path.read_bytes() for path in spool.rglob("*") if path.is_file()}

    p = readonly_smoke.create_readonly_provider({
        "plugins": {
            "graphiti": {
                "url": "http://g/mcp",
                "group_id": "core",
                "failed_write_dir": str(spool),
            }
        }
    })
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("read-only smoke invoked a lifecycle, replay, sync, or write path")

    p.initialize = forbidden
    p.shutdown = forbidden
    p._replay_failed_writes = forbidden
    p.sync_turn = forbidden
    p.on_session_end = forbidden
    p._enqueue_add_memory = forbidden
    p._spool_failed_write = forbidden

    def read_call(tool, args):
        calls.append((tool, args))
        if tool == "get_status":
            return {"status": "ok"}
        if tool == "search_memory_facts":
            return {"facts": [{"fact": "found"}]}
        raise AssertionError(f"unexpected MCP tool: {tool}")

    p._call_tool_sync = read_call
    result = readonly_smoke.run_readonly_smoke(p, query="Hermes Graphiti")
    after = {path.relative_to(spool): path.read_bytes() for path in spool.rglob("*") if path.is_file()}

    assert result["status_success"] is True
    assert result["facts_count"] == 1
    assert calls == [
        ("get_status", {}),
        ("search_memory_facts", {"query": "Hermes Graphiti", "group_ids": ["core"], "max_facts": 1}),
    ]
    assert after == before


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
        assert p._prefetch_cache["s"] == ("exact query", "ctx")


def test_late_prefetch_cannot_overwrite_newer_session_context():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    q1_started = threading.Event()
    release_q1 = threading.Event()

    def build(query):
        if query == "q1":
            q1_started.set()
            assert release_q1.wait(2)
            return "ctx1"
        return "ctx2"

    p._build_recall_context = build
    p.queue_prefetch("q1", session_id="s")
    assert q1_started.wait(1)
    p.queue_prefetch("q2", session_id="s")
    for _ in range(100):
        with p._prefetch_lock:
            if p._prefetch_cache.get("s") == ("q2", "ctx2"):
                break
        time.sleep(0.01)
    release_q1.set()
    for _ in range(100):
        with p._prefetch_lock:
            if not p._inflight_prefetch:
                break
        time.sleep(0.01)
    with p._prefetch_lock:
        assert p._prefetch_cache["s"] == ("q2", "ctx2")


def test_queue_prefetch_skips_query_already_cached_for_session():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp"}}})
    p._prefetch_cache["s"] = ("same query", "ctx")
    called = []
    p._build_recall_context = lambda query: called.append(query) or "duplicate"

    p.queue_prefetch("same query", session_id="s")
    time.sleep(0.05)

    assert called == []
    assert p._prefetch_cache["s"] == ("same query", "ctx")


def test_prefetch_cache_is_session_scoped_and_consumed_by_next_turn():
    p = GraphitiMemoryProvider({"plugins": {"graphiti": {"url": "http://g/mcp", "prefetch_mode": "async"}}})
    with p._prefetch_lock:
        p._prefetch_cache["s"] = ("old query", "old context")
        p._prefetch_cache["other"] = ("other query", "other context")
    queued = []
    p.queue_prefetch = lambda query, *, session_id="": queued.append((session_id, query))
    assert p.prefetch("new query", session_id="s") == "old context"
    assert queued == [("s", "new query")]
    assert "s" not in p._prefetch_cache
    assert p._prefetch_cache["other"] == ("other query", "other context")


def test_parse_mcp2_structured_error_preserves_failure():
    result = types.SimpleNamespace(
        content=[],
        structuredContent={"error": "backend timeout"},
        isError=True,
    )
    assert GraphitiMemoryProvider._parse_mcp_result(result) == {"error": "backend timeout"}


def test_parse_mcp2_structured_content_precedes_text_fallback():
    result = types.SimpleNamespace(
        content=[types.SimpleNamespace(text='{"facts": [{"fact": "old"}]}')],
        structuredContent={"facts": [{"fact": "new"}]},
        isError=False,
    )
    assert GraphitiMemoryProvider._parse_mcp_result(result) == {"facts": [{"fact": "new"}]}


def test_parse_mcp2_unwraps_single_result_envelope():
    result = types.SimpleNamespace(
        content=[],
        structuredContent={"result": {"message": "ok", "facts": [{"fact": "new"}]}},
        isError=False,
    )
    assert GraphitiMemoryProvider._parse_mcp_result(result) == {
        "message": "ok",
        "facts": [{"fact": "new"}],
    }


def test_install_refuses_existing_non_symlink_plugin_dir(tmp_path):
    hermes_home = tmp_path / "hermes"
    existing = hermes_home / "plugins" / "graphiti"
    existing.mkdir(parents=True)
    proc = subprocess.run(
        ["bash", "scripts/install.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env={"HOME": str(tmp_path), "HERMES_HOME": str(hermes_home)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "Refusing to replace existing non-symlink plugin path" in proc.stderr
    assert existing.is_dir()
