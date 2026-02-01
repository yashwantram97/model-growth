# Growth Verification System

## Overview

This document describes the comprehensive verification system for model growth operations. The system ensures that Bilateral Growth (width and depth scaling) preserves model functionality while enabling capacity expansion.

## The Four Critical Checks

### 1. RoPE Integrity Check (The "Geometry" Check)

**Purpose**: Ensure that Rotary Position Embeddings (RoPE) remain valid after growth.

**Problem**: RoPE depends on head dimension: `head_dim = d_model / n_heads`

**Rule**: When scaling `d_model` by factor N, you MUST scale `n_heads` by factor N.

**Verification**:
```python
assert new_model.head_dim == old_model.head_dim
```

**Failure Symptom**: 
- Model will fail on sequences with length > 1
- Position encoding becomes corrupted
- Attention patterns break down

**Fix**: In `scale_bilaterally()`, ensure:
```python
n_heads=old_model.blocks[0].attention.n_heads * scale_factor
```

---

### 2. Router Logit Scaling Check (The "Collapse Prevention" Check)

**Purpose**: Prevent MoE router collapse after scaling.

**Problem**: 
- Router projects `d_model → num_experts`
- When `d_model` grows, input vectors are longer
- If weights aren't scaled, dot products double in magnitude
- Larger logits → sharper softmax → router stops exploring

**Example**:
```
Before: logits = [2.0, 1.5, 1.8] → softmax = [0.38, 0.22, 0.40]  (balanced)
After:  logits = [4.0, 3.0, 3.6] → softmax = [0.55, 0.11, 0.34]  (collapsing)
```

**Verification**:
- Create identical semantic input (old size vs tiled new size)
- Compare router output magnitudes
- Maximum difference should be < 1e-4

**Fix**: In `hyperclone_weight()` for router:
```python
w_new = w.repeat(1, scale_factor) / scale_factor  # Critical division!
```

---

### 3. Symmetry Breaking Check (The "Twin" Check)

**Purpose**: Ensure that cloned heads/experts can learn independently.

**Problem**: 
- HyperCloning creates identical copies: `[Head_A, Head_A, Head_A, ...]`
- Identical weights + identical inputs = identical gradients
- Model can't learn diverse representations

**Solution**: Add small opposing noise during cloning:
```python
# Head A gets +noise, Head B gets -noise
split = w_new.shape[0] // 2
w_new[:split] += noise[:split]
w_new[split:] -= noise[:split]
```

**Verification**:
1. Weight difference: `Weight(Head_A) != Weight(Head_B)`
2. Gradient difference after backward pass
3. If gradients are identical → symmetry breaking failed

**Failure Symptom**:
- All heads learn the same thing
- No benefit from increased capacity
- Model equivalent to smaller model

**Fix**: Increase `noise_std` parameter in `scale_bilaterally()`:
```python
scale_bilaterally(model, scale_factor=2, noise_std=1e-5)  # Increase if needed
```

**Recommended noise levels**:
- `1e-5` to `1e-4`: Good for most cases
- `< 1e-7`: Too small, may not break symmetry
- `> 1e-3`: Too large, may hurt functional preservation

---

### 4. Functional Identity Check (The "Do No Harm" Check)

**Purpose**: Verify that growth doesn't change model behavior.

**Principle**: 
```
new_model(x) ≈ old_model(x)  for all x
```

**Why**: Growth should only add capacity, not change what the model has learned.

**Verification Strategy**:

#### Basic Check (from Phase 1→2):
```python
verify_functional_equivalence(old_model, new_model, probe, device)
```
- Compares final logits
- Maximum difference < 1e-4

#### Layer-by-Layer Check (from Phase 2→3):
```python
check_functional_identity_layerwise(old_model, new_model, probe, device)
```
- Compares activations at EACH layer
- Identifies WHERE drift occurs
- Checks Gstack layers act as near-identity

**Failure Symptoms**:
- Loss spike at step 1 after transition
- NaN or Inf values in outputs
- Large magnitude change in predictions

**Debugging**:
1. Check embedding expansion (should be tiled correctly)
2. Check each layer output
3. Check new layers (should change input minimally)
4. Check final logits

---

## Running Verification

### Automatic (Integrated in Training)

Verification runs automatically in `train_tinystories.py`:

```python
# After Phase 1→2 (Dense → MoE)
verify_functional_equivalence(dense_model, moe_model, probe, device)

# After Phase 2→3 (MoE → Large MoE)
detailed_growth_check(
    old_model=moe_model,
    new_model=large_model,
    probe=probe,
    device=device,
    scale_factor=2,
    tolerance=1e-4
)
```

### Manual Testing

Run the test suite:

```bash
python test_growth_checks.py
```

This will:
1. Test Dense → MoE transition
2. Test Bilateral Growth with all checks
3. Test different noise levels for symmetry breaking
4. Test RoPE integrity

---

## Interpreting Results

### ✓ All Checks Passed

```
#####################################################################
#                      VERIFICATION SUMMARY                         #
#####################################################################
  ROPE                : ✓ PASSED
  ROUTER              : ✓ PASSED
  SYMMETRY            : ✓ PASSED
  FUNCTIONAL          : ✓ PASSED
#####################################################################
  ✓ ALL CHECKS PASSED
  Model is ready for Phase 3 training.
#####################################################################
```

**Action**: Proceed with training. Model is safe to use.

---

### ✗ RoPE Check Failed

```
✗ CRITICAL FAILURE: RoPE broken!
  Head dimension changed from 64 to 128
```

**Cause**: `n_heads` not scaled proportionally with `d_model`

**Fix**: 
```python
# In scale_bilaterally():
n_heads = old_model.blocks[0].attention.n_heads * scale_factor
```

**Critical**: Do NOT proceed with training. Fix immediately.

---

### ⚠ Router Scaling Warning

```
⚠ WARNING: Router logits drifted by 1.5e-3 (> 1e-4)
  Router collapse risk: MODERATE
```

**Cause**: Router weights not scaled correctly

**Fix**:
```python
# In hyperclone_weight() for dim_mode="in":
w_new = w.repeat(1, scale_factor) / scale_factor  # Add division!
```

**Action**: Can proceed but monitor expert usage during training.

---

### ✗ Symmetry Breaking Failed

```
✗ WARNING: Gradients are identical! (diff=3.45e-12)
  Recommendation: Increase noise_std to at least 1e-5
```

**Cause**: `noise_std` too small or zero

**Fix**:
```python
scale_bilaterally(model, scale_factor=2, noise_std=1e-5)  # Increase
```

**Impact**: Model will train but won't learn diverse representations. Capacity increase wasted.

---

### ✗ Functional Check Failed

```
✗ FAILED: Layer-wise drift exceeds tolerance 1e-4
  Layer  0: max_diff=2.34e-5, mean_diff=3.21e-6 ✓
  Layer  1: max_diff=5.67e-3, mean_diff=8.45e-4 ✗
```

**Cause**: 
- Weight expansion logic incorrect
- Scaling factor wrong for specific layer type
- Numerical precision issues

**Debug**:
1. Check which layer failed (attention vs FFN vs norm)
2. Review `hyperclone_weight()` for that layer type
3. Check `dim_mode` parameter matches layer geometry

**Action**: Do NOT proceed. Debug weight expansion logic.

---

## Order of Operations

### Phase 1 → Phase 2 (Dense → MoE)

```
1. Train Dense Model
2. Create probe batch (freeze it)
3. transition_to_moe()
4. verify_functional_equivalence()
   ├─ If PASS: Continue to Phase 2 training
   └─ If FAIL: Debug weight copy logic
5. Train MoE Model
```

### Phase 2 → Phase 3 (MoE → Large MoE)

```
1. Train MoE Model (Phase 2)
2. scale_bilaterally()
3. detailed_growth_check()
   ├─ Check 1: RoPE Integrity
   ├─ Check 2: Router Scaling
   ├─ Check 3: Symmetry Breaking
   └─ Check 4: Functional Identity
4. If ALL PASS:
   └─ Train Large Model (Phase 3)
5. If ANY FAIL:
   ├─ Fix hyperparameters (noise_std, scale_factor)
   ├─ OR fix weight expansion logic
   └─ Re-run scale_bilaterally()
```

---

## Quick Reference

### Parameters to Tune

| Parameter | Location | Typical Value | Effect |
|-----------|----------|---------------|--------|
| `scale_factor` | `scale_bilaterally()` | 2 | Width multiplier (2x, 3x, etc.) |
| `extra_layers` | `scale_bilaterally()` | 2-4 | Depth increase |
| `noise_std` | `scale_bilaterally()` | 1e-5 to 1e-4 | Symmetry breaking strength |
| `tolerance` | `detailed_growth_check()` | 1e-4 | Max allowed drift |

### Common Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Loss → NaN at step 1 | RoPE broken | Scale n_heads with d_model |
| Router uses only 1 expert | Router collapse | Scale router weights by factor |
| No improvement after growth | Symmetry not broken | Increase noise_std |
| Loss spike after growth | Functional drift | Check weight expansion logic |

---

## API Reference

### Functions

#### `detailed_growth_check(old_model, new_model, probe, device, scale_factor=2, tolerance=1e-4)`

Runs all four verification checks.

**Returns**: `bool` - True if all checks pass

**Raises**: 
- `ValueError` if RoPE integrity fails
- `AssertionError` if functional identity fails

---

#### `check_rope_integrity(old_model, new_model)`

Verifies head dimensions are preserved.

**Returns**: `bool`

---

#### `check_router_scaling(old_model, new_model, scale_factor, device)`

Verifies router logits are scaled correctly.

**Returns**: `bool`

---

#### `check_symmetry_breaking(new_model, device)`

Verifies gradients diverge between cloned heads.

**Returns**: `(bool, float)` - (passed, gradient_difference)

---

#### `check_functional_identity_layerwise(old_model, new_model, probe, device, tolerance=1e-4)`

Verifies each layer preserves function.

**Returns**: `bool`

---

#### `quick_sanity_check(model, probe, device, label="Model")`

Quick diagnostic check for model health.

**Returns**: `dict` with statistics (mean, std, min, max, has_nan, has_inf)

---

## Examples

### Example 1: Basic Growth with Verification

```python
from transfer.simple_growth import scale_bilaterally
from transfer.verify_growth_mechanics import detailed_growth_check

# Perform growth
large_model = scale_bilaterally(
    small_model,
    scale_factor=2,
    extra_layers=4,
    noise_std=1e-5
).to(device)

# Verify
passed = detailed_growth_check(
    old_model=small_model,
    new_model=large_model,
    probe=probe_batch,
    device=device,
    scale_factor=2
)

if passed:
    # Continue training
    train_phase(large_model, ...)
```

### Example 2: Debugging a Failed Check

```python
from transfer.verify_growth_mechanics import (
    check_rope_integrity,
    check_symmetry_breaking,
    quick_sanity_check
)

# Quick health check
quick_sanity_check(model, probe, device, "My Model")

# Individual checks
try:
    check_rope_integrity(old_model, new_model)
except ValueError as e:
    print(f"RoPE check failed: {e}")
    # Fix and retry

passed, grad_diff = check_symmetry_breaking(new_model, device)
if not passed:
    print(f"Symmetry breaking failed. Grad diff: {grad_diff:.2e}")
    # Increase noise_std and retry
```

### Example 3: Testing Different Configurations

```python
# Test different noise levels
for noise in [1e-7, 1e-5, 1e-4, 1e-3]:
    large_model = scale_bilaterally(
        small_model,
        scale_factor=2,
        extra_layers=2,
        noise_std=noise
    )
    
    passed, grad_diff = check_symmetry_breaking(large_model, device)
    print(f"noise={noise}: grad_diff={grad_diff:.2e}, passed={passed}")
```

---

## References

1. **HyperCloning**: Weight expansion strategy that preserves function
2. **Gstack**: Depth scaling by copying layers (acts as near-identity)
3. **MLA (Multi-head Latent Attention)**: Symmetry breaking enables diverse head learning
4. **RoPE**: Position encoding that depends on head dimension staying constant

---

## Troubleshooting

### Q: Checks pass but loss still spikes?

**A**: Check:
1. Learning rate (may need to lower for larger model)
2. Batch statistics (larger model may need different batch size)
3. Gradient clipping (may need adjustment)

### Q: Symmetry breaking passes but no capacity gain?

**A**: 
- Check if noise_std is at lower bound (try increasing)
- Verify heads are actually diverging during training (log gradient stats)
- May need longer training for divergence to manifest

### Q: Router check warns but training seems fine?

**A**: 
- Monitor expert usage during training
- If experts are balanced, warning may be false alarm
- If one expert dominates, reduce learning rate or re-initialize router

---

## Version History

- **v1.0** (2026-02-01): Initial implementation with all four checks
  - RoPE integrity check
  - Router scaling check  
  - Symmetry breaking check
  - Layer-wise functional identity check
