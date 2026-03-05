# Appendix: Capabilities and Region-Handle Contract

## Purpose

This appendix captures what the execution/runtime layer must provide for the KV
runtime model to be implementable.

The key point is that the durable contract should be framed in terms of region
handles or exportable regions, not raw addresses.

## Minimal conceptual contract

```text
prepare_export(logical_pages) -> ExportRegionSet
start_export(ExportRegionSet) -> TransferToken
wait_export(TransferToken) -> Result
release_export(ExportRegionSet)

prepare_import(destination_pages) -> ImportRegionSet
start_import(ImportRegionSet, payload) -> TransferToken
wait_import(TransferToken) -> Result
release_import(ImportRegionSet)
```

## What a region should conceptually contain

A region handle may need:

- source identity
- one segment or many segments
- size_total
- layout descriptor
- alignment constraints
- registration or access token
- lifetime token
- sync dependency

## Why handles beat raw addresses

Raw addresses are often the least stable representation.

A handle-based contract is more durable because it can absorb:

- virtualization differences
- registration requirements
- scatter-gather structure
- lifetime validity rules
- sync/fence coupling

## Capability matrix (conceptual)

```text
Capability                               | Old stack | Future stack | Needed long-term
-----------------------------------------|-----------|--------------|-----------------
Scheduler-owned logical page identity    | yes-ish   | yes          | yes
Residency tracking in runtime            | partial   | needed       | yes
Export page as region handle             | weak/hack | needed       | yes
Stable layout descriptor                 | unclear   | needed       | yes
Async fence / event semantics            | unclear   | needed       | yes
Host DMA                                 | some path | needed       | likely yes
Direct physical-address dependence       | hacky     | not durable  | no
Batch export/import                      | needed    | needed       | yes
Invalid-page / failure feedback          | needed    | needed       | yes
```

The exact entries will change as implementation details evolve, but the shape
of the matrix should remain stable.
