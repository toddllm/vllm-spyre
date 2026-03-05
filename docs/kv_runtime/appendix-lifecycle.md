# Appendix: Lifecycle and State Machine

This appendix expands the minimal lifecycle invariant from the core note.

## Why lifecycle is a first-class concern

Moving KV safely requires explicit answers to these questions:

- When is a page stable?
- When is export safe?
- When is import visible?
- What happens if preemption or timeout occurs during transfer?
- What is the fallback rule on uncertainty?

## Detailed page state model

```text
PAGE STATE MODEL
================

UNALLOCATED
  ->
ALLOCATED
  ->
WRITING
  ->
STABLE_ON_DEVICE
  ->
EXPORT_PREPARED
  ->
EXPORT_INFLIGHT
  ->
RESIDENT_ON_HOST/REMOTE
  ->
IMPORT_PREPARED
  ->
IMPORT_INFLIGHT
  ->
STABLE_ON_DEVICE
```

Useful companion fields:

- `location`: `DEVICE`, `HOST`, `REMOTE`, `INFLIGHT`
- `valid_extent`: valid tokens or bytes
- `refcount`
- `epoch`
- `export_handle` / `import_handle`

## Legal transitions

- `WRITING -> STABLE_ON_DEVICE` only after writes are frozen and visible.
- `STABLE_ON_DEVICE -> EXPORT_PREPARED` only if page is exportable.
- `EXPORT_PREPARED -> EXPORT_INFLIGHT` only after a valid export region exists.
- `EXPORT_INFLIGHT -> RESIDENT_ON_HOST/REMOTE` only after transport success.
- `RESIDENT_ON_HOST/REMOTE -> IMPORT_PREPARED` only after destination placement
  is known.
- `IMPORT_PREPARED -> IMPORT_INFLIGHT` only after valid import regions exist.
- `IMPORT_INFLIGHT -> STABLE_ON_DEVICE` only after import completion and
  visibility.

## Illegal transitions

```text
WRITING -> EXPORT_INFLIGHT
WRITING -> RESIDENT_ON_HOST/REMOTE
HOST/REMOTE -> DECODE_CONSUME
INVALID -> DECODE_CONSUME
RELEASED_HANDLE -> TRANSPORT_USE
SHARED_STABLE -> IN_PLACE_OVERWRITE (without COW)
```

## Preemption and inflight transfer

Preemption can occur while export/import is inflight.

Safe rule:

```text
if inflight transfer cannot be proven complete and visible,
mark destination invalid for reuse and fall back to recompute.
```

This favors safety over speculative reuse.

## Timeout and cancelation

Timeout or cancellation should not leave ambiguous state.

Recommended fallback:

- mark transfer result as indeterminate
- invalidate affected destination placement
- force recompute if the request is resumed

## Minimal lifecycle invariants worth preserving across stacks

- never export `WRITING`
- never consume before import completion
- never reuse after uncertain failure without explicit revalidation
- never overwrite a shared stable page in place
