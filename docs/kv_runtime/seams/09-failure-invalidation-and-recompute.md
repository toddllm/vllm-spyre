# Seam: Failure, Invalidation, and Recompute

## Question

What is the safe fallback model when load/import state is uncertain or fails,
and which layer should own retry versus recompute policy?

## Decision target

Design choice decision.

## Current answer

The connector should surface granular failure and completion state. The
scheduler/runtime above it should own the semantic fallback policy.

The safe rule is simple: uncertain state must degrade to invalidation and
recompute rather than silent reuse.

## What this page establishes

- Current code already separates error reporting from semantic policy.
- Safe fallback depends on surfacing invalid state explicitly rather than trying
  to “best effort” reuse uncertain bytes.
- Retry policy belongs above the connector, even if transport may do smaller
  local retries internally.

## Story snippets

### 1. Upstream scheduler integration is where failure becomes a runtime decision

The scheduler-side connector step is already the place where matching,
allocation, and connector metadata are coordinated.

That makes it the correct control-plane location to decide what to do when a
load fails or imported state is incomplete.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step).

### 2. Upstream worker lifecycle already treats “no forward” and post-forward as explicit cases

`ActiveKVConnector` does not assume one happy path. It explicitly models:

- pre-forward
- post-forward
- no-forward

That is important because recovery logic depends on explicit state transitions,
not on assuming every step ran normally.

Relevant anchors in [source-map.md](../source-map.md):
[`UP-ACTIVE-POST`](../source-map.md#up-active-post),
[`UP-ACTIVE-NOFWD`](../source-map.md#up-active-nofwd).

### 3. Experimental connector code already surfaces granular load failure

The experimental connector has a dedicated load-error surface and a separate
completion surface.

That is the right split because the connector should report what happened, not
quietly invent the recovery policy on its own.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr),
[`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish).

### 4. The bridge is where old stack currently translates uncertainty into explicit outcome

On the old stack, the bridge’s post-forward phase is the visible point where
completion state, invalid blocks, and step cleanup are brought together.

That makes it the current old-stack evidence that failure handling is a
lifecycle concern, not a hidden implementation detail.

Relevant anchors in [source-map.md](../source-map.md):
[`EXP-BRIDGE-AFTER`](../source-map.md#exp-bridge-after),
[`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish),
[`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr).

### 5. Retry belongs above the connector even if transport does small local retries

A transport/backend may still do tiny local retries for transient movement
issues, but that is not the same thing as owning semantic recompute policy.

The architecture should keep this split:

- connector/transport reports granular result
- scheduler/runtime decides retry, invalidate, or recompute

Relevant anchors in [source-map.md](../source-map.md):
[`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step),
[`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr),
[`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish).

## What differs in old stack

Old stack still makes the failure story more bridge-shaped and bulk-shaped
because the worker/data-plane seam is not yet natively layered the way upstream
vLLM expects.

That does not change the fallback rule. It only changes where the current code
has to enforce it.

## What should survive into future stack

These parts should survive directly:

- granular failure reporting
- explicit invalidation feedback to the scheduler/runtime
- retry/recompute policy above the connector
- the rule that silent reuse is never an acceptable fallback for uncertain KV

## Relevant anchors in `source-map.md`

- [`UP-SCHED-CONN-STEP`](../source-map.md#up-sched-conn-step)
- [`UP-ACTIVE-POST`](../source-map.md#up-active-post)
- [`UP-ACTIVE-NOFWD`](../source-map.md#up-active-nofwd)
- [`EXP-BRIDGE-AFTER`](../source-map.md#exp-bridge-after)
- [`EXP-CONN-FINISH`](../source-map.md#exp-conn-finish)
- [`EXP-CONN-LOADERR`](../source-map.md#exp-conn-loaderr)
