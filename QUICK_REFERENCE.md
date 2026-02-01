# Quick Reference Card - 3-Phase Training (100 Steps)

## 🚀 Run Command
```bash
python3 train_tinystories.py
```

## ⏱️ Expected Time (Mac M1/M2)
- Phase 1: ~3 minutes
- Phase 2: ~6 minutes  
- Phase 3: ~18 minutes
- **Total: ~27 minutes**

## 📊 Model Progression

| Phase | Type | d_model | Layers | Heads | Total Params | Active Params |
|-------|------|---------|--------|-------|--------------|---------------|
| **1** | Dense | 832 | 7 | 8 | 100 M | 100 M (100%) |
| **2** | MoE | 832 | 7 | 8 | 372 M | 139 M (37.4%) |
| **3** | Large MoE | 1664 | 11 | 16 | 1.47 B | 549 M (37.4%) |

## 🎯 Success Indicators

✅ **Phase 1→2 jump**: < 0.05 (normal)  
✅ **Phase 2→3 jump**: < 0.001 (functional preservation!)  
✅ **Loss dropping**: Each phase shows improvement  
✅ **No OOM errors**: Memory management working  

## 📈 Expected Output Format

### Phase Start (Dense)
```
Total Parameters:   100.04 M
Active Params/tok:  100.04 M  (Dense model)
```

### Phase Start (MoE)
```
Total Parameters:   371.64 M
Active Params/tok:  138.88 M  (top-2 of 8 experts)
Inactive Params:    232.76 M  (42 experts idle)
Efficiency: 37.4% active per forward pass
```

## 🔧 Configuration (config.py)
```python
steps_phase1: int = 100  # Phase 1 steps
steps_phase2: int = 100  # Phase 2 steps  
steps_phase3: int = 100  # Phase 3 steps
log_every: int = 10      # Log frequency
batch_size: int = 8      # Batch size
```

## 💾 Memory Usage (Mac 24GB)

| Phase | Model | Training | Total | Status |
|-------|-------|----------|-------|--------|
| 1 | 0.4 GB | 1.4 GB | ~1.8 GB | ✅ Safe |
| 2 | 1.5 GB | 4.8 GB | ~6.3 GB | ✅ Safe |
| 3 | 5.9 GB | 18.2 GB | ~24.1 GB | ⚠️ Tight |

## 🆘 If OOM in Phase 3
```python
# Option 1: Reduce batch size
batch_size: int = 4  # Instead of 8

# Option 2: Smaller growth
scale_factor=1.5     # Instead of 2
extra_layers=2       # Instead of 4
```

## 📁 Output Files (~10 GB total)
```
checkpoints/
├── dense_model.pt          (~400 MB)
├── moe_model_final.pt      (~1.5 GB)
├── large_moe_final.pt      (~5.9 GB)
└── training_history.jsonl  (~50 KB)
```

## 📝 Key Metrics to Watch

### During Training
- **Step 0 loss**: Initial loss value
- **Step 10, 20...**: Loss trajectory
- **Final loss**: End-of-phase performance

### At Boundaries
- **Phase 1 end**: ~6.8 (typical)
- **Phase 2 start**: ~6.8-6.9 (small jump OK)
- **Phase 3 start**: ~6.2 (should match Phase 2 end!)

### Parameter Reporting
- **Total params**: All weights in memory
- **Active/tok**: Parameters used per token
- **Efficiency**: Active / Total ratio

## 🔍 Verification Commands

```bash
# Check config
python3 -c "from config import TrainingConfig; print(TrainingConfig())"

# Test imports
python3 -c "from transfer.simple_growth import scale_bilaterally; print('OK')"

# Check GPU/MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

## 📖 Documentation Files

- **PHASE3_BILATERAL_GROWTH.md**: Complete technical docs
- **PHASE3_QUICKSTART.md**: Usage guide
- **TEST_RUN_100_STEPS.md**: Test configuration details
- **CHANGES_SUMMARY.md**: What was changed
- **QUICK_REFERENCE.md**: This file

## 🎓 Understanding the Output

### "top-2 of 8 experts"
- 8 experts exist per MoE layer
- Only top-2 are activated per token
- 6 experts are idle (but still in memory)

### "42 experts idle"
- 7 layers × 6 idle experts = 42 idle experts
- These consume memory but not compute

### "Efficiency: 37.4%"
- Only 37.4% of parameters compute per token
- 62.6% are loaded but waiting
- This enables 2.7× capacity at same cost

## ⚡ Quick Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| OOM in Phase 3 | Model too large | Reduce batch_size to 4 |
| High P2→P3 jump | Preservation failed | Check noise_std (try 1e-6) |
| Loss spike | LR too high | Reduce lr_phase3 by 0.5× |
| Slow training | CPU mode | Check MPS is available |

## 🎯 What Success Looks Like

```
BOUNDARY SUMMARY
═══════════════════════════════════════════════════════════
  Phase 1 final loss      :  6.8234
  Phase 2 first loss      :  6.8456
  Phase 2 final loss      :  6.2145
  Phase 1→2 jump          :  0.0222  ✅ Small (normal)

  Phase 3 first loss      :  6.2146
  Phase 3 final loss      :  5.8923  
  Phase 2→3 jump          :  0.0001  ✅ TINY (success!)
  Phase 3 total drop      :  0.3223
═══════════════════════════════════════════════════════════
```

**If Phase 2→3 jump < 0.001**: 🎉 Bilateral growth working perfectly!

---

**Ready?** Run: `python3 train_tinystories.py`
