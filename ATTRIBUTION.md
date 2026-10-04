# Attribution

This project (Axodex) is a derivative of [GitNexus](https://github.com/abhigyanpatwari/GitNexus),
originally created by **Abhigyan Patwari**.

We are deeply grateful to Abhigyan for building the zero-server code
intelligence engine that forms the foundation of Axodex. His work on
the tree-sitter grammar vendoring, the knowledge-graph ingestion
pipeline, the DuckDB FTS integration, and the multi-language call-
extraction architecture is exceptional engineering that saved us
months of development.

## What we changed

- Renamed `gitnexus` → `axodex` across the entire codebase
- Repackaged as `@fraziym/axodex` on npm
- Added 12 new tree-sitter languages (Bash, SQL, Dockerfile, YAML, TOML,
  JSON, HTML, CSS, Scala, Elixir, Lua, GraphQL)
- Integrated with AXONIZ as a peer dependency
- Added FRAZIYM versioning (V00.01.001-beta-02)
- Added model-training/ directory for fine-tuning small specialized
  models for AXOVB and AXOTEST

## What we kept

- The original PolyForm-Noncommercial-1.0.0 license
- The full tree-sitter grammar vendoring architecture
- The ingestion pipeline (call extractors, import resolvers, scope
  resolution, type extraction)
- The MCP server and CLI
- The web UI (axodex-web)
- The test fixtures and benchmarks

## Original license

The original GitNexus license (PolyForm-Noncommercial-1.0.0) is
preserved. See [LICENSE](./LICENSE) for the full text.

---

Thank you, Abhigyan. Your work is the backbone of Fraziym's code
intelligence stack.

— Akik Faraji, Fraziym Tech & AI
