# Appendix: Conformance Invariants

## Purpose

The architecture should be enforceable, not aspirational.

This appendix names the invariant families that a conformance test suite should
cover across both old and future stacks.

## Identity conformance

- source identity remains ABA-safe
- destination placement is not confused with source identity
- shared-prefix divergence respects COW boundaries

## Lifecycle conformance

- WRITING pages are not exported
- decode does not consume before import completion
- invalid or indeterminate pages are not silently reused

## Residency conformance

- location and state transitions remain legal
- refcount and sharing state stay consistent
- preemption does not leave ambiguous reusable state

## Transport conformance

- retries do not silently corrupt state
- retries under the same identity + epoch are idempotent
- partial failure results in invalidation and recompute
- release/unregister semantics prevent stale-handle reuse

## Layout conformance

- exported/imported regions preserve page validity extent
- partial-page semantics remain correct
- layer/page layout is interpreted consistently
