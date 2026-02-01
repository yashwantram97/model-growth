# Quick Start: Growth Verification System

## TL;DR

✅ **Verification is automatic** - just run your training script  
✅ **Tests available** - run `python test_growth_checks.py` to verify setup  
✅ **Checks run at phase transitions** - Phase 1→2 and Phase 2→3  

---

## 30-Second Overview

The verification system ensures model growth doesn't break your model. It checks:

1. **RoPE** - Position encoding still works
2. **Router** - MoE experts stay balanced  
3. **Symmetry** - Cloned heads learn different things
4. **Function** - Model still does what it learned

---

## Run Tests (2 minutes)

```bash
python test_growth_checks.py
```

Expected output:
```
======================================================================
  TEST 1: Dense → MoE Transition
======================================================================
  ✓ PASSED — functional equivalence confirmed

======================================================================
  TEST 2: Bilateral Growth Verification
======================================================================
  ✓ ALL CHECKS PASSED
  Model is ready for Phase 3 training.

✓ Dense → MoE transition test completed!
✓ Bilateral growth test completed!
```

---

## Run Training (with auto-verification)

```bash
python train_tinystories.py
```

Verification happens automatically:
- **After Phase 1 (Dense training)** → Check before MoE conversion
- **After Phase 2 (MoE training)** → Full 4-check verification before growth

If you see:
```
✓ ALL CHECKS PASSED
Model is ready for Phase 3 training.
```

→ **Everything is working!** 🎉

---

## If Checks Fail

### "RoPE broken"
```python
# In scale_bilaterally(), ensure:
n_heads = old_model.blocks[0].attention.n_heads * scale_factor
```

### "Router logits drifted"
```python
# In hyperclone_weight() for router:
w_new = w.repeat(1, scale_factor) / scale_factor  # ← Add division
```

### "Gradients are identical"
```python
# Increase noise in train_tinystories.py:
noise_std = 1e-4  # Was 1e-5, try 1e-4 or higher
```

### "Layer-wise drift exceeds tolerance"
→ Read `GROWTH_VERIFICATION.md` section 4 for detailed debugging

---

## Adjust Parameters

Edit in `train_tinystories.py` (lines 359-366):

```python
scale_factor = 2      # Width: 2x = double size, 3x = triple size
extra_layers = 4      # Depth: how many layers to add
noise_std = 1e-5      # Symmetry breaking: 1e-5 to 1e-4 typical
```

---

## Files You Care About

### Use These:
- `train_tinystories.py` - Main training script (verification included)
- `test_growth_checks.py` - Test verification without training
- `QUICKSTART_VERIFICATION.md` - This file

### Read If Curious:
- `GROWTH_VERIFICATION.md` - Full explanation of why checks matter
- `VERIFICATION_SUMMARY.md` - Implementation details

### Modify If Needed:
- `transfer/verify_growth_mechanics.py` - Check implementations
- `transfer/simple_growth.py` - Growth algorithm
- `config.py` - Training hyperparameters

---

## Common Questions

### Q: Do I need to call verification manually?
**A**: No! It runs automatically in `train_tinystories.py`.

### Q: Can I skip verification?
**A**: Not recommended. Takes <1 minute, saves hours of debugging bad models.

### Q: What if I only want to test one check?
**A**: Import individual functions:
```python
from transfer.verify_growth_mechanics import check_rope_integrity
check_rope_integrity(old_model, new_model)
```

### Q: Can I make checks stricter/looser?
**A**: Yes, adjust `tolerance` parameter:
```python
detailed_growth_check(..., tolerance=1e-5)  # Stricter (less drift allowed)
detailed_growth_check(..., tolerance=1e-3)  # Looser (more drift allowed)
```

---

## What Success Looks Like

```
######################################################################
#                     VERIFICATION SUMMARY                           #
######################################################################
  ROPE                : ✓ PASSED
  ROUTER              : ✓ PASSED
  SYMMETRY            : ✓ PASSED
  FUNCTIONAL          : ✓ PASSED
######################################################################
  ✓ ALL CHECKS PASSED
  Model is ready for Phase 3 training.
######################################################################

============================================================
  PHASE 3  —  Large MoE (Bilateral Growth)
  Steps: 1000  |  LR: 8e-05  |  Device: cuda
  ─────────────────────────────────────────────────────────
  Total Parameters:       XXX.XX M
  Active Params/tok:      XXX.XX M  (top-2 of 8 experts)
  ─────────────────────────────────────────────────────────
  Efficiency: XX.X% active per forward pass
============================================================
  step    0 | loss 2.3456 | 0.5s
  step   50 | loss 2.1234 | 3.2s
  ...
```

---

## Getting Help

1. **Run tests first**: `python test_growth_checks.py`
2. **Check which test failed**: Look at the CHECK # in output
3. **Read error message**: It tells you exactly what's wrong
4. **Try suggested fix**: Error messages include fixes
5. **Still stuck?**: Read full docs in `GROWTH_VERIFICATION.md`

---

## Summary

- ✅ **Automatic**: Verification runs during training
- ✅ **Fast**: Takes ~30 seconds to verify growth
- ✅ **Helpful**: Tells you exactly what's wrong and how to fix it
- ✅ **Complete**: Covers all 4 critical growth mechanics

**Just run `python train_tinystories.py` and watch for ✓ PASSED messages!**
