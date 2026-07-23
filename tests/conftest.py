from __future__ import annotations
import sys, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

agent = types.ModuleType("agent")
memory_provider_mod = types.ModuleType("agent.memory_provider")
class MemoryProvider:
    @property
    def name(self): raise NotImplementedError
    def is_available(self): raise NotImplementedError
    def initialize(self, session_id, **kwargs): raise NotImplementedError
    def system_prompt_block(self): return ""
    def prefetch(self, query, *, session_id=""): return ""
    def queue_prefetch(self, query, *, session_id=""): pass
    def sync_turn(self, user_content, assistant_content, *, session_id="", messages=None): pass
    def get_tool_schemas(self): return []
    def handle_tool_call(self, tool_name, args, **kwargs): raise NotImplementedError
    def shutdown(self): pass
memory_provider_mod.MemoryProvider = MemoryProvider
sys.modules.setdefault("agent", agent)
sys.modules.setdefault("agent.memory_provider", memory_provider_mod)

tools = types.ModuleType("tools")
registry_mod = types.ModuleType("tools.registry")
def tool_error(message, success=False):
    import json
    return json.dumps({"success": success, "error": str(message)}, ensure_ascii=False)
registry_mod.tool_error = tool_error
sys.modules.setdefault("tools", tools)
sys.modules.setdefault("tools.registry", registry_mod)
