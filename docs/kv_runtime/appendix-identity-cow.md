# Appendix: Identity, Sharing, and Copy-on-Write

## Why identity needs more than block IDs

A destination block ID is a placement choice, not the identity of the source
bytes.

A safe source identity is conceptually closer to:

```text
(prefix_key or request lineage,
 layer_id,
 logical_page_index,
 epoch/generation)
```

This avoids ABA bugs where a block ID is reused for different contents over
time.

## Source identity vs destination placement

```text
SOURCE IDENTITY
---------------
what bytes are these?

DESTINATION PLACEMENT
---------------------
where do those bytes go now?
```

Those must remain separate.

## Shared prefixes

Shared-prefix reuse means multiple requests can reference the same stable pages.

That implies runtime tracking for:

- `refcount`
- whether a page is shared
- whether appending would require COW

## Copy-on-write boundary

```text
Request A: [P0][P1][P2][P3] -> append A-tail
Request B: [P0][P1][P2][P3] -> append B-tail

Shared pages P0..P3 must not be overwritten in place.
```

If a request needs to modify a shared logical tail, it must allocate new pages
or otherwise cross an explicit COW boundary.

## Extent and dirtiness

Each page should track:

- `valid_tokens` or valid byte range
- whether the page is partial
- whether a dirty tail is still growing
- when it becomes frozen and shareable/exportable

Without extent tracking, partial-page export/import is ambiguous.
