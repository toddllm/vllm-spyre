# Appendix: Old-Stack Experimental Path

## Purpose

This appendix isolates current-stack experimentation details that are important
for learning, but should not define the durable architecture.

## What the old stack is good for

The old stack can still validate:

- scheduler-owned logical page identity
- connector metadata flow
- residency tracking concepts
- lifecycle correctness
- batching and recompute-fallback behavior

## What makes the old stack awkward

The compiled execution path does not naturally expose a clean runtime seam for
returning KV pages after prefill.

As a result, current experiments may rely on:

- compiler/deeptools artifact inspection
- offset/segment reconstruction
- synthetic tensors or views rooted in resolved device regions
- eager host copies as a visibility bridge

Those may be acceptable for experimentation, but they are not the target
abstraction.

## Durable lesson vs temporary hack

Durable lesson:

- runtime needs logical page identity
- runtime needs export/import region construction
- runtime needs legal lifecycle and failure semantics

Temporary hack:

- how exactly the current old stack reconstructs or exposes the region today

## Boundary condition

If old-stack work is done, it should be explicitly framed as validating runtime
contracts, not as defining the long-term API.
