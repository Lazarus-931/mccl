# mccl work brief — from mccl-torch integration findings (2026-06-11)

Context: mccl-torch now drives mccl as a `torch.distributed` backend. DDP and FSDP2 training are
verified end-to-end against single-process references on 2–3 rank loopback. PR #5
(`fix/direct-allreduce-determinism`) is merged; the items below assume it.

## Contracts the torch binding relies on — do not break

1. **Rank-determinism of reductions.** After PR #5, every allreduce path (direct/ring/tree/dtree)
   must produce bit-identical results on all ranks. DDP silently depends on this. Any new algorithm
   or Metal path must preserve it (one rank computes each chunk, or fold in a rank-independent
   order).
2. **Grouped p2p semantics** (`p2p.cc`): thread-local group state; `GroupStart`/`End` called from
   one thread; self-send rejected with `mcclInvalidArgument`; batches deadlock-free regardless of
   posting order. The binding composes gather / scatter / uneven all-to-all / coalesced p2p on
   these exact properties.
3. **`mcclUniqueId` layout** (`char[128]` packing IP/port): the binding memcpys it through torch's
   rendezvous store. Version it if the layout ever changes.
4. **`count == 0` is a success no-op**; validation errors return synchronously before `mcclLaunch`
   (a peer must never block on a rank that failed validation).
5. **`mcclConfig.transport`** is resolved during `mcclCommInitRankConfig` (no pointer retention)
   and all ranks fail together if the transport is unavailable — keep both properties.

## P0 — abort contract (unblocks torch-side timeout work)

The binding needs to enforce per-op timeouts; the only mechanism is calling `mcclCommAbort` from a
different thread while an op is blocked in send/recv. Verify and document this contract: abort
closes all connections → the blocked op returns a fatal `mcclResult` (no hang, no crash, no
double-free), and the comm is poisoned for subsequent ops. Add a loopback test that blocks a recv
forever and aborts from another thread. If any path can't be unblocked (e.g., the bootstrap accept
loop), fix or document it.

## P1 — RDMA over Thunderbolt 5 (macOS 26.2, TN3205)

The verbs implementation in `src/transport/rdma/rdma.cc` already targets Apple's API
(`<infiniband/verbs.h>` + `dlopen("librdma.dylib")`). Remaining:

1. **Build-path verification.** The current dev SDK lacks `infiniband/verbs.h`, so shipped
   `libmccl.a` contains the stub (0 ibv symbols). Once on macOS 26.2 + a current Xcode: confirm the
   real path compiles, `mcclRdmaAvailable()` returns true, and add a tiny `mccl-rdma-check` style
   diagnostic (prints the device list) so cluster bring-up failures are debuggable.
2. **Memory-registration cache.** Training traffic re-sends the same buffers every step (gradient
   buckets, flat params). Add a `(ptr, len)`-keyed MR cache (NCCL-style user-buffer registration)
   or an explicit `mcclCommRegister`/`Deregister` pair, with invalidation on dereg/abort. This is
   the single biggest RDMA perf item.
3. **Multi-QP striping.** TCP stripes across sockets; RDMA is one QP per peer ("one socket,
   immediate UC bring-up", `m2m.cc`). Stripe across QPs to saturate TB5 (80/120 Gbps), mirroring
   the TCP shard logic.
