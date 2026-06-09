# mccl

> **mccl** · /ˈmɪk.əl/ · *Mac Collective Communication Library*

Topology-adaptive collective communication for clusters of Apple-Silicon Macs, over unified memory and
Thunderbolt. mccl sits between your training code and the hardware: you call a collective, and mccl works
out the cluster for you — which links are physically up, how to route over them, and which algorithm is
fastest for the size — then runs the reduction on the GPU over unified memory with no host/device copies.

Much of inspiration came from [nccl](https://github.com/NVIDIA/nccl/tree/73cf112295c33aee2b895f329f592f2a9b4b0f97#)


## Build

Requirements: macOS on Apple Silicon (Metal), a C++17 `clang++`, and CMake ≥ 3.20. The Macs are wired by
Thunderbolt; the build itself has no third-party dependencies.

```sh
$ git clone https://github.com/Lazarus-931/mccl.git
$ cd mccl
$ cmake -S . -B build
$ cmake --build build -j
```

This produces `build/libmccl.a`. `reduce.metal` is compiled at runtime, so there is no offline
kernel-build step. Launch one rank per Mac with `MCCL_RANK`, `MCCL_WORLD_SIZE`, and a shared
`MCCL_BOOTSTRAP_IP` / `MCCL_BOOTSTRAP_PORT` — the only required configuration; the topology and routing
are discovered.

## Install

```sh
$ cmake --install build --prefix /usr/local
```

Installs `libmccl.a` → `<prefix>/lib`, the headers → `<prefix>/include/mccl`, and `reduce.metal` →
`<prefix>/share/mccl`. A consuming CMake project finds it with `find_package(MCCL)`; or compile by hand:

```sh
$ clang++ -std=c++17 app.cc -I<prefix>/include/mccl/include \
      <prefix>/lib/libmccl.a -framework Metal -framework Foundation -o app
```

If `reduce.metal` is not beside the binary, point `MCCL_METAL_DIR=<prefix>/share/mccl` at the installed
copy. Tests and benchmarks live in a separate `mccl-tests` repository.

## Structure

```
src/graph/     topology: discover (interfaces + liveness) -> topo (system + links) -> paths
               (all-pairs widest path) -> search (per algorithm) -> connect (per-rank ring/tree)
               -> tuning (cost model); trees = double-binary builder
src/device/    per-Mac compute: metal (MTLDevice + pipeline cache + UMA reduce), reduce_kernel,
               primitives (send / recv / reduce building blocks), and one file per collective
               (all_reduce, all_gather, reduce_scatter, broadcast, reduce)
src/transport/ m2m: striped TCP over Thunderbolt, plus an RDMA-over-TB backend behind the facade
src/           comm (init -> discover -> graph -> connect; split / finalize / abort), coll (public
               API), bootstrap (rendezvous), socket, allocator (page-aligned UMA)
```


