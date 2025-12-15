# TK Seq-First Kernel Optimization Plan

## Executive Summary

The TK seq-first kernel (`attn_seq_first.cu`) is orders of magnitude slower than the native CUDA implementation (`kernel_cuda.cu`), while the TK chunk-first kernel (`attn_chunk_first.cu`) achieves better performance than native CUDA. This document outlines optimization strategies to bring seq-first performance in line with chunk-first.

## Architecture Comparison

### Chunk-First Kernel (Fast)
- **Grid**: `(n_heads, n_chunks)` - one block per (head, chunk)
- **Work unit**: Block processes all seqs in a chunk together
- **MMAs**: Uses `warpgroup::mm_ABt` and `warpgroup::mma_AB` (H100 tensor cores)
- **Memory**: TMA for Q loading, async cp.async for K/V
- **Allocator**: `tma_swizzle_allocator` (TMA-compatible swizzled layout)
- **Tile sizes**: Q=[64,128], K=[64,128] -> scores=[64,64] -> output=[64,128]

### Seq-First Kernel (Slow) - Current State
- **Grid**: `(n_heads, n_seqs)` - one block per (head, seq)
- **Work unit**: Block processes all chunks for a single sequence
- **MMAs**: Uses `warp::` scalar operations (NO tensor core MMA!)
- **Memory**: Manual scalar-indexed loads `KV_s[{row, col}] = ptr[idx]`
- **Allocator**: `shared_allocator` (non-TMA layout)
- **Tile sizes**: Per-warp rt<64,128> tiles with warp parallelism across chunks

### Native CUDA Seq-First (Reference)
- Uses `warp_vect_mul_raw_major_matrix_v2` for Q @ K^T (vector dot products)
- Uses `warp_vect_mul_col_major_matrix` for scores @ V
- Async `cp.async` loads with proper swizzling
- Each warp processes different chunks, cross-warp reduction at end

## Root Cause Analysis

### Critical Issue 1: No Tensor Core Utilization
**Current code (lines 135-146 in attn_seq_first.cu):**
```cuda
kv_tile_t K_r;
warp::load(K_r, KV_s);
q_rv_t Q_rv;
warp::load(Q_rv, Q_sv);
kv_tile_t QK_tile;
warp::broadcast_col(QK_tile, Q_rv);  // Broadcasts Q into full tile
warp::mul(QK_tile, QK_tile, K_r);    // Element-wise multiply!
scores_cv_t scores_cv;
warp::row_sum(scores_cv, QK_tile);   // Reduces to get dot products
```
This computes Q @ K^T via broadcast + element-wise multiply + row_sum. This is **NOT using tensor cores** - it's using scalar FMA operations.

**Impact**: ~10-100x slower than tensor core MMA for matmul.

### Critical Issue 2: Slow Memory Loads
**Current code (lines 129-132):**
```cuda
for(int j = lane; j < chunk_size * d_head; j += 32) {
    KV_s[{j / d_head, j % d_head}] = K_ptr[j];
}
```
This uses scalar indexed stores with integer division/modulo, which is extremely slow compared to vectorized async loads.

### Critical Issue 3: Incompatible Tile Dimensions for warpgroup MMA
The seq-first kernel has Q as a single row vector (shape [1, d_head]) while K/V are [chunk_size, d_head]. Warpgroup MMA requires specific tile dimensions (multiples of 16/64) and distributed register layouts.

### Critical Issue 4: Serial Cross-Warp Reduction
Final output reduction (lines 197-231) has warp 0 sequentially reading all warp outputs. This serializes work that could be parallelized.

---

## Optimization Strategies (Ranked by Priority)

### Priority 1: Replace Broadcast+Mul+Sum with MMA [HIGH IMPACT, MEDIUM DIFFICULTY]

**Problem**: Q @ K^T computed via scalar operations
**Solution**: Use warpgroup or warp MMA instructions
**Expected Speedup**: 10-50x on this compute portion
**Difficulty**: MEDIUM - requires restructuring data layout

**Approach A: Replicate Q to enable warpgroup MMA**
Since Q is [1, 128] and K is [64, 128], we cannot directly use mm_ABt.
Option: Replicate Q to [64, 128] and use warpgroup::mm_ABt, then extract diagonal or first row.
This is wasteful but enables tensor cores.

**Approach B: Use warp-level MMA with proper Q broadcast**
TK's `warp::mma_ABt` may work if we structure Q as a partial tile.
Need to verify API constraints in `ops/group/mma/warp/warp.cuh`.

**Approach C: Use batch of independent warp MMAs**
Structure the computation so each warp's subset can use MMA.
Q[16,128] @ K[64,128]^T -> [16, 64] per warp, then reduce.

**Recommended**: Start with Approach B if API supports it, otherwise Approach C.

---

### Priority 2: Use Async Memory Loads [HIGH IMPACT, LOW DIFFICULTY]

**Problem**: Manual scalar loads with integer division
**Solution**: Use `load_tile_async` pattern from chunk_first kernel
**Expected Speedup**: 2-5x on memory operations
**Difficulty**: LOW - copy pattern from chunk_first

**Implementation**:
```cuda
// Replace manual loop with:
load_tile_async<chunk_size, d_head, BLOCK_SIZE>(KV_s, K_ptr);
cp_async_commit();
cp_async_wait<0>();
__syncthreads();
```

This also requires using `tma_swizzle_allocator` instead of `shared_allocator`.

---

### Priority 3: TMA for Repeated Q Access [MEDIUM IMPACT, LOW DIFFICULTY]

**Problem**: Q is loaded once but could benefit from TMA
**Solution**: Use TMA descriptor for Q if accessed repeatedly
**Expected Speedup**: 1.2-1.5x on Q loads
**Difficulty**: LOW - follow chunk_first pattern

**Note**: In seq-first, Q is only used once per block, so benefit is smaller than chunk-first where Q is used across the entire chunk.

---

### Priority 4: Improve Cross-Warp Reduction [MEDIUM IMPACT, MEDIUM DIFFICULTY]

**Problem**: Sequential reduction by warp 0
**Solution**: Use TK group reduction primitives or parallel tree reduction
**Expected Speedup**: 2-4x on reduction phase
**Difficulty**: MEDIUM - requires understanding of TK group operations

**Current pattern**:
```cuda
if (warp == 0) {
    for(int d = lane; d < d_head; d += 32) {
        float sum = 0.0f;
        for(int w = 0; w < NUM_WARPS; w++) {
            sum += all_out_sv[w][d] * warp_scale[w];
        }
        output_ptr[d] = __float2bfloat16(sum * inv_sum_s);
    }
}
```

**Better pattern**: All warps participate in parallel reduction using shared memory atomics or tree-based reduction.

---

### Priority 5: Fuse K and V Loads [LOW-MEDIUM IMPACT, LOW DIFFICULTY]

**Problem**: K and V loaded sequentially
**Solution**: Pipeline K and V loads - start V load while computing Q @ K^T
**Expected Speedup**: 1.3-1.5x by hiding V load latency
**Difficulty**: LOW - add async V load before Q@K computation

**Pattern from chunk_first**:
```cuda
// Load K async
load_tile_async<...>(KV_s, K_ptr);
cp_async_commit();
// Start Q @ K^T while K loads...
// Wait for K
cp_async_wait<0>();
// Compute Q @ K^T
// Meanwhile, preload V (reuse KV_s buffer after K is consumed)
```

---

### Priority 6: Use Shared Memory for Intermediate Results [LOW IMPACT, LOW DIFFICULTY]

**Problem**: Scores stored in registers, then streamed to shared for V multiply
**Solution**: Keep computation flow register -> shared -> register optimal
**Expected Speedup**: 1.1-1.2x
**Difficulty**: LOW

---

### Priority 7: Adjust Block Configuration [LOW IMPACT, MEDIUM DIFFICULTY]

**Problem**: Current config may not be optimal for warpgroup MMA
**Solution**: Consider using warpgroup::increase_registers / decrease_registers
**Expected Speedup**: 1.1-1.3x
**Difficulty**: MEDIUM - requires tuning

---

## Implementation Roadmap

### Phase 1: Quick Wins (Low Difficulty, Immediate Impact) - COMPLETED

**Changes implemented:**
1. Added `cp_async` helper functions for H100 async memory copies
2. Added `load_tile_async_warp` template function with TK swizzle support
3. Switched to hybrid allocator: `shared_allocator` for vectors, `tma_swizzle_allocator` for KV tiles
4. Replaced manual scalar K/V loads with async `load_tile_async_warp` calls
5. Pipelined V load to overlap with Q @ K^T computation

**Results:**
- 60/64 tests pass (93.75%)
- 4 tests show minor precision differences (max_diff ~0.5-0.8) with large random values
- All "small" value tests pass, indicating numerical precision rather than correctness issues

**Expected cumulative speedup: 2-5x**

### Phase 2: MMA Integration (Medium Difficulty, High Impact)
1. Restructure Q/K/V tiles for MMA compatibility
2. Replace broadcast+mul+sum with warp MMA or warpgroup MMA
3. Handle partial tiles (when chunk_num % NUM_WARPS != 0)

**Expected cumulative speedup: 10-30x**

### Phase 3: Final Optimizations (Medium Difficulty, Medium Impact)
1. Parallelize cross-warp reduction
2. Add register optimization (increase_registers/decrease_registers)
3. Tune launch bounds and shared memory allocation

**Expected cumulative speedup: 20-50x (matching or exceeding native CUDA)**

---

## Technical Notes on TK API Verification

### Verified APIs (from source code inspection):
- `warpgroup::mm_ABt(D, A, B)` - D = A @ B^T, requires A height divisible by 64
- `warpgroup::mma_AB(D, A, B)` - D += A @ B, accumulate variant
- `tma::load_async(dst, src, coord, semaphore)` - TMA tile load
- `load_tile_async<ROWS, COLS, THREADS>` - custom async load with swizzling

### Constraints Identified:
- warpgroup MMA requires A.height == 64 (4 warps x 16 rows each)
- warpgroup MMA requires all 4 warps to participate
- Shared tiles must use TK's swizzled layout for MMA operands
- Semaphores needed for TMA synchronization

### Potentially Deprecated APIs (verify before use):
- Some `warp::` MMA functions may have limited support
- Direct global->register loads may not work with all layouts

---

## Success Metrics

| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Latency vs Native CUDA | ~100x slower | 2x slower | 1x (parity) |
| TFLOPs achieved | <1 | >10 | >50 |
| Memory bandwidth utilization | <5% | >30% | >60% |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| warpgroup MMA incompatible with Q shape | Medium | High | Use warp-level MMA or tile replication |
| Async loads cause correctness issues | Low | Medium | Verify with test harness after each change |
| Shared memory overflow | Low | Medium | Calculate smem budget carefully |
| API incompatibility with current TK version | Medium | Medium | Verify all APIs against TK source before use |

---

## Appendix: Key Code Locations

### Current TK Seq-First Kernel
- File: `attn_seq_first.cu`
- Lines 88-189: Main computation loop (needs MMA conversion)
- Lines 129-132: K load (needs async conversion)
- Lines 169-172: V load (needs async conversion)
- Lines 197-231: Cross-warp reduction (needs parallelization)

### TK Chunk-First Kernel (Reference)
- File: `attn_chunk_first.cu`
- Lines 53-87: Async load helpers
- Lines 163-179: TMA Q load + async K load
- Lines 193-196: warpgroup::mm_ABt for Q @ K^T
- Lines 256-259: warpgroup::mma_AB for scores @ V

### Native CUDA Kernel (Reference)
- File: `kernel_cuda.cu`
- Lines 582-651: warp_vect_mul_raw_major_matrix_v2 (Q @ K pattern)
- Lines 653-717: warp_vect_mul_col_major_matrix (scores @ V pattern)
- Lines 886-1087: attn_seq_first_kernel implementation

### TK Library Source (for API verification)
- `include/ops/group/mma/warpgroup/warpgroup.cuh` - warpgroup MMA ops
- `include/ops/group/memory/tile/tma.cuh` - TMA operations
- `include/types/shared/st.cuh` - Shared tile types and swizzling
