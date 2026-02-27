# KV Connector PR Draft Packet

This folder captures PR-ready slices for the KV connector workstream.

## Branch inventory

Base mirror (fork):
- `codex/spyre-kv-base-pr759`
- [compare link](https://github.com/toddllm/vllm-spyre/tree/codex/spyre-kv-base-pr759)

Slice 1 (bridge lifecycle):
- `codex/spyre-kv-slice1-bridge`
- base: `codex/spyre-kv-base-pr759`
- compare: <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-slice1-bridge>
- doc: `slice1-bridge.md`

Slice 2 (connector core + metadata + reuse correctness):
- `codex/spyre-kv-slice2-core`
- base: `codex/spyre-kv-slice1-bridge` (stacked)
- compare: <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice1-bridge...codex/spyre-kv-slice2-core>
- doc: `slice2-core.md`

Slice 3 (runtime hardening + async-ready + 2-process tests):
- `codex/spyre-kv-slice3-runtime`
- base: `codex/spyre-kv-slice2-core` (stacked)
- compare: <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice2-core...codex/spyre-kv-slice3-runtime>
- doc: `slice3-runtime.md`

Slice 4 (metrics adapter):
- `codex/spyre-kv-slice4-metrics`
- base: `codex/spyre-kv-slice3-runtime` (stacked)
- compare: <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-slice3-runtime...codex/spyre-kv-slice4-metrics>
- doc: `slice4-metrics.md`

Combined branch for validation:
- `codex/spyre-kv-combined` (currently same code as slice4)
- base mirror compare:
  <https://github.com/toddllm/vllm-spyre/compare/codex/spyre-kv-base-pr759...codex/spyre-kv-combined>
- doc: `combined-testing.md`

Planning/integration branch (notes + complete history):
- `codex/spyre-kv-connector`
- contains internal notes and slicing docs.

## Quick strategy

If opening PRs immediately:
1. Open Slice 1 first.
2. Open Slice 2 targeting Slice 1.
3. Open Slice 3 targeting Slice 2.
4. Open Slice 4 targeting Slice 3.

If opening against upstream main after merges:
1. Rebase each slice onto latest agreed base.
2. Retarget PR base to upstream branch used by maintainers.
3. Keep internal docs (`docs/internal/`) out of upstream PR payload unless requested.
