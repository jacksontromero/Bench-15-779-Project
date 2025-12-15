# TK Seq-First Kernel Optimization Plan

## Executive Summary

The TK seq-first kernel (`attn_seq_first.cu`) has been optimized using TK primitives. Phase 2 (scalar operations with TK memory management) is complete and working. Phase 3 (MMA integration) was attempted but requires further work due to challenges with TK's register tile layout.

## Current Status

### Phase 2: Scalar Tiling with TK Primitives [COMPLETED ✓]

**Metrics**:
- Registers: 128 (0 spills)
- All 32/32 small tests pass
- Benchmarks complete successfully

**Implementation**:
- TK `shared_allocator` for memory management
- TK types: `sv<bf16, d_head>`, `sv<float, d_head>`, `sv<float, chunk_size>`, `st<bf16, chunk_size, d_head>`
- TK softmax primitives: `warp::max`, `warp::sum`, `warp::sub`, `warp::exp`
- Scalar dot products for Q @ K^T
- Manual weighted sum for scores @ V

### Phase 3: MMA Integration [PARTIAL / NEEDS WORK]

**Attempted approach**:
1. Broadcast Q to a register tile using `warp::broadcast_col`
2. Use `warp::mma_ABt` for Q @ K^T computation
3. Use `warp::mma_AB` with `warp::swap_layout` for scores @ V

**Challenges encountered**:
1. **Register tile extraction**: TK's `rt` types have complex internal layouts. Extracting a single row (which is what we need since Q is broadcast) is non-trivial.
2. **Layout requirements**: `warp::mma_AB` requires B to be in `col_layout`, requiring `warp::swap_layout` which adds overhead.
3. **Shared memory pressure**: Additional tiles for MMA intermediate results increased shared memory from ~85KB to ~155KB.
4. **Stack frame growth**: Phase 3 prototype used 168 registers and 368 bytes stack frame.

**Recommendation for Phase 3**:
For the seq-first kernel where Q is a single vector (not a tile), MMA may not be the right approach because:
- The 16×16 MMA output contains 16 identical copies of the same result (wasted computation)
- Extracting useful data from register tiles requires storing to shared memory and reading back
- The overhead may exceed benefits for vector-matrix products

**Alternative approaches to explore**:
1. Use CUDA Cores (current Phase 2) - already efficient for this pattern
2. Consider restructuring to batch multiple sequences per block (seq-first → hybrid)
3. Use `warp::dot` for efficient vector-dot-vector operations per K row

---

## Architecture Overview

### Seq-First Kernel Pattern
- **Grid**: (n_heads, n_seqs) - one block per (head, sequence) pair
- **Warps**: 4 warps, each processes multiple chunks in round-robin
- **Core Operation**: Q (1×d_head) · K^T (d_head×chunk_size) = scores (1×chunk_size)

### Why MMA is Challenging
- Q is a **vector**, not a tile
- MMA expects **tile × tile** operations (e.g., 16×k × k×16)
- Broadcasting Q creates 16 identical rows → 16× redundant computation
- Extracting results requires understanding TK's register tile data layout

---

## File Structure

- `attn_seq_first.cu` - Kernel implementation (Phase 2)
- `harness_seq_first.impl` - Test harness
- `tests_seq_first/` - Test data files
- `run_tests_seq_first.sh` - Test runner script

## Build & Test

```bash
export THUNDERKITTENS_ROOT=/path/to/ThunderKittens
conda activate myenv
cd kernels/chunked_attn
make clean && make
./run_tests_seq_first.sh
./attn_seq_first --sweep --warmup 50 --iters 200
```

## Performance Summary (Phase 2)

| Configuration | Latency (ms) | TFLOPS | Compute Eff % |
|--------------|--------------|--------|---------------|
| H=4, C=4     | 0.019       | 1.77   | 0.18%         |
| H=32, C=16   | 0.479       | 2.24   | 0.23%         |
| H=64, C=64   | 3.77        | 2.28   | 0.23%         |
| H=128, C=128 | 15.07       | 2.28   | 0.23%         |

## Success Metrics
- **Phase 2**: All tests pass, 0 register spills ✓
- **Phase 3**: Attempted, blocked by register tile layout complexity
