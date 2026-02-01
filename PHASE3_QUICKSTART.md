# Phase 3 Bilateral Growth - Quick Start Guide

## What Was Added

### New Files
1. **`transfer/simple_growth.py`** - Core bilateral growth implementation
2. **`PHASE3_BILATERAL_GROWTH.md`** - Comprehensive documentation
3. **`PHASE3_QUICKSTART.md`** - This guide

### Modified Files
1. **`train_tinystories.py`** - Added Phase 3 training section

## Key Implementation

### HyperCloning Function
Expands weight matrices while preserving function:
- **Width (d_model)**: 832 → 1664 (2x)
- **Heads (n_heads)**: 8 → 16 (2x)
- **Head dim**: 104 → 104 (constant, RoPE-safe!)
- **Layers**: 7 → 11 (+4 new layers)

### Expansion Strategy
```
Original MoE Model (Phase 2):
├─ d_model: 832
├─ n_heads: 8
├─ head_dim: 104
├─ d_ff: 3328
├─ n_layers: 7
├─ num_experts: 8
└─ Parameters: ~372M total (~139M active)

Large MoE Model (Phase 3):
├─ d_model: 1664  (2x)
├─ n_heads: 16    (2x)
├─ head_dim: 104  (constant!)
├─ d_ff: 6656     (2x)
├─ n_layers: 11   (+4)
├─ num_experts: 8 (same)
└─ Parameters: ~1.4B total (~520M active)
```

## Running the Code

### Option 1: Full 3-Phase Training
```bash
python3 train_tinystories.py
```

This will run:
1. **Phase 1** (steps 0-999): Dense training
2. **Phase 2** (steps 1000-1999): MoE upcycling + training
3. **Phase 3** (steps 2000-2999): Bilateral growth + training

### Option 2: Start from Phase 2 Checkpoint
```python
import torch
from transfer.simple_growth import scale_bilaterally
from transfer.simple_transfer import verify_functional_equivalence

# Load Phase 2 checkpoint
checkpoint = torch.load("checkpoints/moe_model_final.pt")
moe_model = SLM(**checkpoint['config'])
moe_model.load_state_dict(checkpoint['model_state_dict'])

# Expand to Phase 3
large_model = scale_bilaterally(
    moe_model,
    scale_factor=2,
    extra_layers=4,
    noise_std=1e-5
)

# Verify functional preservation
probe = torch.randint(0, 50257, (8, 256))  # Example probe
verify_functional_equivalence(moe_model, large_model, probe, device)
```

## Expected Output

### Console Output During Growth
```
============================================================
  PHASE 3  —  Bilateral Growth (Width x2, Layers +4)
============================================================

[Bilateral Growth] Scaling x2 width, +4 layers...
  > Initialized Layer 7 as Gstack copy of Layer 6
  > Initialized Layer 8 as Gstack copy of Layer 6
  > Initialized Layer 9 as Gstack copy of Layer 6
  > Initialized Layer 10 as Gstack copy of Layer 6
✓ Growth Complete. New Config: 1664 dim, 11 layers, 16 heads.

  [Verification] Checking Growth Preservation...
  [Verification] Per-sequence max |logit_dense − logit_moe|:
    Seq 0: 8.54e-06  OK
    Seq 1: 9.12e-06  OK
    ...

  Overall max difference: 9.12e-06
  ✓ PASSED — functional equivalence confirmed (< 0.0001)
```

### Boundary Summary
```
============================================================
  BOUNDARY SUMMARY
============================================================
  Phase 1 final loss      :  6.8234
  Phase 2 first loss      :  6.8456
  Phase 2 final loss      :  6.2145
  Phase 1→2 jump          :  0.0222 (normal batch variation)
  Phase 2 total drop      :  0.6311

  Phase 3 first loss      :  6.2146  ← Should be VERY close to Phase 2 end
  Phase 3 final loss      :  5.8923
  Phase 2→3 jump          :  0.0001  ← Tiny! Functional preservation works!
  Phase 3 total drop      :  0.3223
============================================================
```

## Key Parameters

### Growth Configuration
- **`scale_factor`**: How much to expand width (default: 2)
  - Recommended: 2, 3, or 4
  - Non-integer values may cause slight misalignments

- **`extra_layers`**: Number of layers to add (default: 4)
  - Start conservative (2-4), then increase if needed
  - Each layer adds ~100-200M parameters

- **`noise_std`**: Symmetry-breaking noise (default: 1e-5)
  - Too low (< 1e-6): Heads may not specialize fast enough
  - Too high (> 1e-4): May break functional preservation
  - Sweet spot: 1e-5 to 5e-5

### Training Configuration
- **Learning Rate**: 0.8 × Phase 2 LR (default: 8e-5)
  - Larger models typically need lower LR
  - Can adjust based on loss trajectory

- **Steps**: 1000 (same as Phase 2)
  - Can increase to 2000-3000 for better convergence
  - Monitor loss - should continue dropping steadily

## Checkpoints Created

After running, you'll have:
```
checkpoints/
├── dense_model.pt          # Phase 1 output
├── moe_model_init.pt       # Phase 2 start (after upcycling)
├── moe_model_final.pt      # Phase 2 output
├── large_moe_final.pt      # Phase 3 output (NEW!)
└── training_history.jsonl  # All phases logged
```

## Verification Checklist

✅ **Imports work**: `from transfer.simple_growth import scale_bilaterally`
✅ **Phase 2→3 jump < 0.01**: Functional preservation confirmed
✅ **Loss continues dropping in Phase 3**: Model is learning effectively
✅ **No CUDA OOM errors**: Memory management working
✅ **Checkpoints saved**: All models preserved for analysis

## Troubleshooting

### Issue: "Phase 2→3 jump is large (> 0.1)"
**Solution**: Check if embeddings were expanded correctly. Try lowering `noise_std`.

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce `scale_factor` to 1.5 or keep at 2
- Reduce `extra_layers` to 2
- Lower batch size in config

### Issue: "Loss spikes in Phase 3"
**Solution**:
- Lower learning rate (try 0.5x or 0.3x of Phase 2)
- Add gradient clipping (already at 1.0, can try 0.5)
- Verify functional preservation passed

### Issue: "Heads not specializing"
**Solution**:
- Increase `noise_std` to 5e-5
- Train for more steps (2000-3000)
- Check router gradients are flowing

## Next Steps

1. **Run the full pipeline**: `python3 train_tinystories.py`
2. **Monitor the logs**: Watch for the Phase 2→3 jump
3. **Analyze results**: Check `training_history.jsonl` for loss curves
4. **Experiment**: Try different `scale_factor` and `extra_layers` values

## Performance Expectations

### Training Time (on M1 Mac)
- Phase 1: ~10-15 minutes
- Phase 2: ~20-30 minutes
- Phase 3: ~60-90 minutes (4x larger model)

### GPU Memory Usage
- Phase 1: ~2-3 GB
- Phase 2: ~4-5 GB
- Phase 3: ~12-16 GB (watch for OOM!)

### Loss Trajectory
- Phase 1 drop: ~1.5-2.0
- Phase 2 drop: ~0.5-0.8
- Phase 3 drop: ~0.3-0.5 (diminishing returns expected)

## Further Reading

- **`PHASE3_BILATERAL_GROWTH.md`**: Comprehensive technical documentation
- **`Explanation.md`**: Mathematical foundations and theory
- **`transfer/simple_growth.py`**: Implementation details with comments

---

**Happy scaling! 🚀**
