# Appendix: Transport Model and Batching

## Why transport needs its own appendix

Transport questions mix architecture with economics.

The architecture question is:

- can runtime export and import the right regions safely?

The policy/economics question is:

- when is transfer better than recompute under real topology and load?

## Small-transfer vs large-transfer regime

The key architectural takeaway is not a single number.

It is the regime split:

- small transfers tend to be latency-bound
- large transfers tend to be bandwidth-bound

That implies a design rule:

```text
logical page identity may be page-sized,
but transport should operate on coalesced batches when performance matters.
```

## Transport cost model

The naive comparison is:

```text
transfer_cost < recompute_cost
```

The runtime policy comparison is closer to:

```text
effective_transfer_cost =
    raw_transfer_time
  + contention_penalty
  + decode_latency_penalty
  + bandwidth opportunity cost
```

So architecture enables the capability, but runtime policy still depends on
load and topology.

## Topology caveat

Shared host-memory, PCIe, switch, or fabric paths can make transfer interfere
with decode traffic.

That means the crossover point between offload and recompute is deployment-
specific even when the core capability is correct.
