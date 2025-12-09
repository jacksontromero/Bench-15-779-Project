# ThunderKittens Installation Troubleshooting

## Issue Summary

ThunderKittens (TK) fails to compile on H100 GPU due to API incompatibilities in the attention kernel.

---

## Attempted Solutions

### Attempt 1: Install from chunk-attention-tk submodule
**Command:** `pip install -e /home/user/15-779-Project/chunk-attention-tk/ThunderKittens`

**Issue:** Target was set to '4090' in config.py

**Action Taken:** Changed `config.py`:
```python
target = 'H100'  # Was: '4090'
kernels = ['attn']  # Was: ['based']
```

**Result:** Compilation errors in attention kernel

---

### Compilation Errors

**File:** `kernels/attn/h100/h100.cu`

**Errors:**
1. `identifier "neg_infty" is undefined`
2. `identifier "make_causal" is undefined`
3. `identifier "make_causal_t" is undefined`
4. `identifier "mul" is undefined`
5. `identifier "copy" is undefined`
6. `no instance of overloaded function "exp2" matches the argument list`

**Analysis:** The attention kernel appears to use an older TK API that has changed.

---

## Potential Solutions

### Solution 1: Try Official TK Repository ⭐ RECOMMENDED
The chunk-attention-tk submodule may be outdated. Try the official repo:

```bash
cd /home/user
git clone https://github.com/HazyResearch/ThunderKittens.git
cd ThunderKittens

# Check if config exists and set it
# Set target to 'H100' and kernels to ['attn']

pip install -e . --no-build-isolation
```

### Solution 2: Check for H100 Branch
The official repo may have an H100-specific branch:

```bash
cd ThunderKittens
git branch -a
git checkout <h100-branch-if-exists>
```

### Solution 3: Use Pre-built Wheels (if available)
Check PyPI or conda-forge:

```bash
pip search thunderkittens
# or
conda search thunderkittens
```

### Solution 4: Contact TK Maintainers
If none of the above work:
- Open GitHub issue: https://github.com/HazyResearch/ThunderKittens/issues
- Include: GPU model (H100), CUDA version (12.4), PyTorch version (2.5.1)
- Reference: This project attempts to benchmark prefix-aware attention

### Solution 5: Workaround - Focus on Baseline Analysis
If TK cannot be installed quickly:
1. Complete baseline benchmarking (✅ DONE)
2. Reproduce paper benchmarks using baseline implementations
3. Theoretical analysis of where TK could improve
4. Document what TK *would* need to achieve competitive performance

---

## Next Steps

**Priority 1:** Try Solution 1 (official repo)  
**Priority 2:** Check for updates/branches  
**Priority 3:** If blocked, proceed with Solution 5 (workaround)

**Timeline:** Spend max 2-3 hours on installation. If unsuccessful, pivot to baseline analysis.

---

## Alternative Approach: Manual Compilation

If pip install fails, try manual compilation:

```bash
cd /home/user/15-779-Project/chunk-attention-tk/ThunderKittens

# Check what the attention kernel actually needs
cat kernels/attn/h100/h100.cu | head -n 50

# Try building just the C++ extension manually
python setup.py build_ext --inplace

# If that works, add to PYTHONPATH
export PYTHONPATH=/home/user/15-779-Project/chunk-attention-tk/ThunderKittens:$PYTHONPATH
```

---

## Status

- ❌ TK not currently installed
- ✅ Baseline benchmarks working (FlashAttention)
- ⏳ Waiting for TK installation resolution
- ✅ Can proceed with Phase 2 (paper reproduction) using baselines

---

## Impact Assessment

**Blocked:**
- Direct TK performance benchmarks
- TK vs FlashAttention comparison
- TK kernel tuning

**Not Blocked:**
- Baseline benchmarking (✅ COMPLETE)
- Paper benchmark reproduction (using original implementations)
- Profiling and analysis
- Theoretical optimization analysis
- Documentation

**Conclusion:** Project can continue. TK installation is important but not blocking all progress.

