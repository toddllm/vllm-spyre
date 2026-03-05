# Appendix: Open Assumptions to Challenge

These are the assumptions worth reviewing before implementation expands.

## Core assumptions

- source identity must be stronger than destination block ID alone
- the stable export/import seam should be region handles, not raw addresses
- shared stable pages should be immutable or protected by explicit COW rules
- pages need valid extent/dirtiness metadata, not just identity
- runtime should own residency tracking
- sync/lifetime semantics are part of the architecture, not optional transport detail
- transport should optimize batched movement, not page-at-a-time micro-transfers
- direct physical-address assumptions are not the durable long-term abstraction

## Questions still open

- what is the exact minimum identity tuple?
- what is the right `prefix_key` or equivalent reusable-key concept?
- who owns refcounts in the concrete implementation?
- what is the minimum handle contract needed for export/import?
- what is the right old-stack hook for returning KV after prefill?
- what is the correct transfer granularity under realistic workloads?
