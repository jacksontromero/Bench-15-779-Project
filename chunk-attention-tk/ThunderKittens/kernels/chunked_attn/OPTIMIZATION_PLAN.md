# TK Seq-First Kernel Optimization Plan

## Executive Summary

The TK seq-first kernel (`attn_seq_first.cu`) is being optimized to match and exceed the native CUDA implementation. The approach uses a **phased strategy**: Phase 2 implements scalar operations with micro-tiling to avoid register pressure, then Phase 3 will add Tensor Core acceleration.

## Architecture Comparison

### Chunk-First Kernel (Working Reference)
- **MMAs**: Uses `warpgroup::mm_ABt` (H100 tensor cores).
- **Status**: Production ready.
- See `attn_chunk_first.cu` for working TK patterns.

### Seq-First Kernel (Optimization Target)
- **Grid**: (n_heads, n_seqs) - one block per (head, sequence) pair
- **Warps**: 4 warps, each processes multiple chunks in round-robin
- **Algorithm**: Vector-matrix products (Q is 1×d_head, not a tile)

### Native CUDA Seq-First (Reference)
- **Compute**: SIMD `HFMA2` (CUDA Cores).
- **Tiling**: Streams small tiles via `warp_vect_mul_raw_major_matrix_v2`.
- **Key insight**: Q is a VECTOR, not a matrix. Each warp computes Q·K^T as a vector-matrix product.

---

## Root Cause Analysis

### Critical Issue 1: Register Pressure
Previous TK version loaded 64x128 `float` tiles (32KB), causing massive spills.
**Fix**: Implemented `SUBTILE_ROWS = 16`. Stream 16 K/V rows at a time.

### Critical Issue 2: Incorrect MMA Usage (CURRENT BUG)
The current code incorrectly uses `warp::mma_ABt` for Phase 2. This is wrong because:
1. Q is a vector (1×d_head), not a tile - MMA is wasteful
2. Type mismatches: scores computed as `rt<float,16,16>` but stored to `col_vec<rt<bf16,...>>`
3. Extracting row 0 from a 16×16 MMA result is overly complex

**Fix (This Plan)**: Revert to scalar dot products for Phase 2.

---

## Optimization Strategies

### Phase 2: Scalar Tiling with TK Primitives [IN PROGRESS]

**Goal**: Correct implementation using scalar operations (no MMA).

**Algorithm per warp, per chunk**:
```
1. Load K subtile (16 × d_head) to registers
2. For each of 16 K rows: score[i] = dot(Q, K[i])  // Manual dot product
3. Apply masking if last chunk
4. Update running max, compute exp(score - max)
5. Accumulate sum of exp scores
6. Load V subtile (16 × d_head)
7. For each of 16 V rows: out += exp_score[i] * V[i]
8. Repeat for next 16-row subtile
```

**Implementation Checklist**:
- [x] Use `shared_allocator` (linear layout, no swizzle)
- [x] Implement `load_tile_async_warp_linear` for K/V loading
- [ ] **FIX**: Remove MMA operations, use scalar dot products
- [ ] **FIX**: Use `rv<float, chunk_size>` for scores (not bf16 col_vec)
- [ ] **FIX**: Use `rv<float, d_head>` for output accumulation
- [ ] Verify: Compiles with 0 spills, all tests pass

**Key TK Primitives for Phase 2**:
- `sv<bf16, d_head>` - Q in shared memory
- `rv<float, 16>` - scores for one subtile (distributed across lanes)
- `warp::max(scalar, rv)` - reduce vector to max
- `warp::sum(scalar, rv)` - reduce vector to sum
- `warp::sub(rv, rv, scalar)` - subtract max for numerical stability
- `warp::exp(rv, rv)` - element-wise exp

### Phase 3: MMA Integration (Tensor Cores) [FUTURE]

**Goal**: After Phase 2 is working correctly, upgrade to Tensor Cores.

**Approach**:
1. Broadcast Q to 16 rows: `rt<bf16, 16, d_head>`
2. Use `warp::mma_ABt` for Q @ K^T
3. Extract row 0 from result (all rows are identical due to Q broadcast)
4. Use `warp::mma_AB` for scores @ V

**Note**: Only proceed to Phase 3 after Phase 2 passes all tests.

---

## File Structure

- `attn_seq_first.cu` - Kernel implementation
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
```

## Success Metrics
- **Phase 2**: All tests pass, 0 register spills
- **Phase 3**: >1.2x speedup vs native kernel
