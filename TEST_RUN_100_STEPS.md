# Test Run Configuration (100 Steps Per Phase)

## Configuration Changes

✅ **Reduced to 100 steps per phase** for fast testing on Mac 24GB
✅ **Enhanced parameter reporting** showing total and active parameters
✅ **More frequent logging** (every 10 steps instead of 50)

## Updated Configuration

```python
@dataclass
class TrainingConfig:
    steps_phase1: int = 100  # Dense training
    steps_phase2: int = 100  # MoE training
    steps_phase3: int = 100  # Bilateral growth training
    log_every: int = 10      # Log every 10 steps
```

## Expected Output Format

### Phase 1: Dense Model Training
```
============================================================
  Phase 1 — Dense
  Steps: 100  |  LR: 0.0003  |  Device: mps
  ─────────────────────────────────────────────────────────
  Total Parameters:   100.04 M
  Active Params/tok:  100.04 M  (Dense model)
  ─────────────────────────────────────────────────────────
============================================================

  step    0 | loss 10.8234 | 0.5s
  step   10 | loss 9.2145 | 5.2s
  step   20 | loss 8.5678 | 10.1s
  ...
  step   90 | loss 6.8234 | 45.2s
```

### Phase 2: MoE Model Training
```
============================================================
  Phase 2 — MoE
  Steps: 100  |  LR: 0.0001  |  Device: mps
  ─────────────────────────────────────────────────────────
  Total Parameters:   371.64 M
  Active Params/tok:  138.88 M  (top-2 of 8 experts)
  Inactive Params:    232.76 M  (42 experts idle)
  ─────────────────────────────────────────────────────────
  Efficiency: 37.4% active per forward pass
============================================================

  step    0 | loss 6.8456 | 0.8s
  step   10 | loss 6.7234 | 8.5s
  ...
```

### Phase 3: Large MoE (Bilateral Growth)
```
============================================================
  Phase 3  —  Bilateral Growth (Width x2, Layers +4)
============================================================

[Bilateral Growth] Scaling x2 width, +4 layers...
  > Initialized Layer 7 as Gstack copy of Layer 6
  > Initialized Layer 8 as Gstack copy of Layer 6
  > Initialized Layer 9 as Gstack copy of Layer 6
  > Initialized Layer 10 as Gstack copy of Layer 6
✓ Growth Complete. New Config: 1664 dim, 11 layers, 16 heads.

  [Verification] Checking Growth Preservation...
  Overall max difference: 9.12e-06
  ✓ PASSED — functional equivalence confirmed (< 0.0001)

============================================================
  Phase 3 — Large MoE (Bilateral Growth)
  Steps: 100  |  LR: 8e-05  |  Device: mps
  ─────────────────────────────────────────────────────────
  Total Parameters:  1467.60 M
  Active Params/tok:  548.70 M  (top-2 of 8 experts)
  Inactive Params:    918.90 M  (66 experts idle)
  ─────────────────────────────────────────────────────────
  Efficiency: 37.4% active per forward pass
============================================================

  step    0 | loss 6.2146 | 1.5s
  step   10 | loss 6.1234 | 15.2s
  ...
```

## Parameter Growth Summary

| Phase | Model Type | Total Params | Active Params/tok | Efficiency |
|-------|-----------|--------------|-------------------|------------|
| **Phase 1** | Dense (7 layers) | 100.04 M | 100.04 M | 100.0% |
| **Phase 2** | MoE (7 layers, 8 experts) | 371.64 M | 138.88 M | 37.4% |
| **Phase 3** | Large MoE (11 layers, 8 experts) | ~1.47 B | ~549 M | 37.4% |

## Key Metrics Displayed

### For Dense Models:
- ✅ Total Parameters
- ✅ Active Params/tok (same as total)

### For MoE Models:
- ✅ Total Parameters (all experts loaded in memory)
- ✅ Active Params/tok (only top-k experts per token)
- ✅ Inactive Parameters (idle experts)
- ✅ Efficiency percentage (what % is actually used per forward pass)
- ✅ Expert counts (e.g., "top-2 of 8 experts", "42 experts idle")

## Memory Expectations (Mac 24GB)

### Phase 1 (Dense):
- Model: ~400 MB
- Gradients: ~400 MB
- Optimizer states: ~800 MB
- Activations (batch=8): ~200 MB
- **Total: ~1.8 GB** ✅ Safe

### Phase 2 (MoE):
- Model: ~1.5 GB
- Gradients: ~1.5 GB
- Optimizer states: ~3.0 GB
- Activations (batch=8): ~300 MB
- **Total: ~6.3 GB** ✅ Safe

### Phase 3 (Large MoE):
- Model: ~5.9 GB
- Gradients: ~5.9 GB
- Optimizer states: ~11.8 GB
- Activations (batch=8): ~500 MB
- **Total: ~24.1 GB** ⚠️ Close to limit!

**Recommendations for Mac 24GB:**
- Phase 3 should work but will be tight
- If you get OOM errors, try:
  - Reduce batch size from 8 to 4
  - Reduce scale_factor from 2 to 1.5
  - Reduce extra_layers from 4 to 2

## Running the Test

```bash
# Run the full 3-phase training (100 steps each)
python3 train_tinystories.py
```

**Expected runtime on Mac M1/M2:**
- Phase 1: ~2-3 minutes
- Phase 2: ~4-6 minutes
- Phase 3: ~12-18 minutes
- **Total: ~20-30 minutes** for full pipeline

## What to Watch For

### ✅ Success Indicators:
1. **Phase 1→2 jump**: Should be small (~0.01-0.05)
2. **Phase 2→3 jump**: Should be VERY small (~0.0001)
3. **Loss continues dropping** in each phase
4. **No memory errors**
5. **Parameter counts match** the table above

### ⚠️ Warning Signs:
1. **Phase 2→3 jump > 0.1**: Functional preservation failed
2. **Loss spikes** during training
3. **OOM errors**: Need to reduce model size or batch size
4. **Loss not decreasing**: Learning rate too high/low

## Output Files

After completion, you'll have:
```
checkpoints/
├── dense_model.pt              (~400 MB)
├── moe_model_init.pt          (~1.5 GB)
├── moe_model_final.pt         (~1.5 GB)
├── large_moe_final.pt         (~5.9 GB) ← Final large model
└── training_history.jsonl     (~50 KB)
```

**Total disk space needed: ~10 GB**

## Quick Verification

After running, check the final summary:
```
BOUNDARY SUMMARY
════════════════════════════════════════════════════════════
  Phase 1 final loss      :  X.XXXX
  Phase 2 first loss      :  X.XXXX
  Phase 2 final loss      :  X.XXXX
  Phase 1→2 jump          :  0.0XXX  ← Small is good
  Phase 2 total drop      :  X.XXXX

  Phase 3 first loss      :  X.XXXX
  Phase 3 final loss      :  X.XXXX
  Phase 2→3 jump          :  0.0001  ← VERY small means success!
  Phase 3 total drop      :  X.XXXX
════════════════════════════════════════════════════════════
```

If `Phase 2→3 jump` is < 0.001, your bilateral growth is working perfectly! 🎉
