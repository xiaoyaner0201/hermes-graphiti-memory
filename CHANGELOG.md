# Changelog

## 0.1.5 - 2026-08-05

- Validate the standalone provider against Hermes v0.20.0 / `v2026.8.3`.
- Support MCP SDK 1.x and 2.x Streamable HTTP transports.
- Preserve structured MCP errors and unwrap MCP 2 `structuredContent` result envelopes.
- Use latency-aware explicit recall budgets and asynchronous automatic prefetch.
- Migrate `get_episodes` calls to `group_ids` and `max_episodes`.
- Fence per-session prefetch generations so late old workers cannot replace newer context.
- Propagate nested session recall failures into top-level unified recall status.
- Document that async prefetch is a delayed session pipeline, not a query-keyed cache.

## 0.1.0 - 2026-07-31

- Initial standalone Graphiti memory provider and unified recall tool.