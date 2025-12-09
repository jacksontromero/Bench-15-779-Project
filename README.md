# ThunderKittens Prefix-Aware Attention Kernels

**Course:** 15-779 Advanced Computer Architecture  
**Focus:** Benchmarking and analyzing ThunderKittens (TK) implementations for prefix-aware attention

---

## Project Overview

This project evaluates ThunderKittens implementations of prefix-aware attention mechanisms:
1. **Chunk Attention TK** - Custom TK kernel for chunk-first attention
2. **Hydragen TK** - TK integration for Hydragen attention (attempted)

**Hardware:** NVIDIA H100 80GB HBM3  
**Environment:** `conda env tk`

---

## Project Structure

```
15-779-Project/
├── README.md                              # This file
├── background.txt                         # Project context & milestone report
├── TK_INSTALLATION_TROUBLESHOOTING.md    # TK integration challenges
│
├── chunk-attention-tk/                    # Chunk Attention + TK
│   └── ThunderKittens/kernels/chunked_attn/
│       ├── attn_chunk_first.cu           # TK kernel implementation
│       ├── attn_chunk_first              # Compiled binary
│       └── tests/                        # Test files
│
├── hydragen-tk/                          # Hydragen + TK (integration blocked)
│   └── hydragen/
│       └── attention_tk.py               # TK adapter (needs TK package)
│
└── results/
    └── CHUNK_ATTENTION_TK_RESULTS.md     # TK kernel results
```

---

## Key Results

### ✅ Chunk Attention TK (Working)

**Status:** Fully functional
- All 36/36 correctness tests passing
- Performance: ~168 GFLOPS
- Custom kernel by jacksontromero
- Standalone C++ binary (no Python package needed)

**Test & Run:**
```bash
cd chunk-attention-tk/ThunderKittens/kernels/chunked_attn

# Run all tests
./run_tests.sh

# Single test
./attn_chunk_first tests/randn_s32_h4_c4.txt

# Benchmark
python /home/user/15-779-Project/chunk_attention_tk_quick_bench.py
```

### ⚠️ Hydragen TK (Blocked)

**Status:** Cannot compile
- TK Python package fails to build
- H100 attention kernel has API incompatibilities
- Functions like `neg_infty`, `make_causal`, `mul`, `copy` are undefined
- See `TK_INSTALLATION_TROUBLESHOOTING.md` for details

---

## Findings

### 1. TK Kernel Implementation

**Chunk Attention TK successfully demonstrates:**
- Row-parallel computation across 2 warps
- bf16 operations for H100
- TK warp::mma operations
- Stable numerical results (max diff < 0.06)

**Configuration:**
- MAX_N_SEQS: 32
- CHUNK_SIZE: 64
- D_HEAD: 128
- NUM_WARPS: 2

### 2. Integration Challenges

**Hydragen TK blocked by:**
- TK Python package compilation failures
- H100 attention kernel not compatible with current TK API
- Missing namespace qualifications or API changes
- Time investment to fix: uncertain (3-4+ hours minimum)

### 3. Lessons Learned

**TK Development:**
- Header-only library approach works well
- Standalone kernels easier to deploy than Python packages
- H100 support may lag behind main TK development
- API stability is a challenge

---

## Next Steps

### For This Project:

**Option 1: Profile Chunk Attention TK** ⭐ Recommended
```bash
cd chunk-attention-tk/ThunderKittens/kernels/chunked_attn
ncu --set full -o profile.ncu-rep ./attn_chunk_first tests/randn_s32_h4_c4.txt
```

**Option 2: Compare with Original ChunkAttention CUDA**
- Benchmark original implementation
- Compare TK vs CUDA performance
- Analyze differences

**Option 3: Document & Write Up**
- Analysis of TK kernel architecture
- Performance characteristics
- Recommendations for optimization

### For Future Work:

1. Fix Hydragen TK integration (requires TK maintainer support or kernel fixes)
2. Optimize Chunk Attention TK kernel
3. Implement additional prefix-aware patterns in TK

---

## References

**Papers:**
- ChunkAttention: "ChunkAttention: Efficient Prefix-Aware KV Cache for Long Context LLM Inference"
- Hydragen: "Hydragen: High-Throughput LLM Inference with Shared Prefixes"

**Code:**
- ThunderKittens: https://github.com/HazyResearch/ThunderKittens
- Chunk Attention TK: https://github.com/jacksontromero/chunk-attention-tk
- Hydragen TK: https://github.com/ayushk7102/hydragen-tk

---

## Quick Commands

```bash
# Activate environment
conda activate tk

# Test Chunk Attention TK
cd chunk-attention-tk/ThunderKittens/kernels/chunked_attn
./run_tests.sh

# View results
cat results/CHUNK_ATTENTION_TK_RESULTS.md

# View TK troubleshooting
cat TK_INSTALLATION_TROUBLESHOOTING.md
```

---

**Status:** Phase 1 complete - TK kernel validated, ready for profiling and analysis
