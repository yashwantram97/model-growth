# Runtime Estimates for AWS g5.2xlarge

## Instance Specifications

**AWS g5.2xlarge:**
- **GPU**: 1x NVIDIA A10G (24GB VRAM)
- **Tensor Cores**: 2nd Gen (Ampere architecture)
- **FP32 Performance**: 31.2 TFLOPS
- **Memory Bandwidth**: 600 GB/s
- **vCPUs**: 8 cores
- **RAM**: 32 GB

## Expected Runtime (100 Steps Per Phase)

### Quick Estimates

| Phase | Model Size | Steps | Time | Notes |
|-------|-----------|-------|------|-------|
| **Phase 1** | 100M Dense | 100 | **~1-2 min** | Fast, efficient |
| **Phase 2** | 372M MoE | 100 | **~2-3 min** | Sparse computation |
| **Phase 3** | 1.47B Large MoE | 100 | **~5-8 min** | Larger model |
| **Total** | - | 300 | **~8-13 min** | Full pipeline |

### Detailed Breakdown

#### Phase 1: Dense Model (100M params)
```
Steps: 100
Batch size: 8
Sequence length: 256
Training: ~1-2 minutes
Reason: Small model, high GPU utilization
```

**Expected throughput:**
- ~1000-1500 tokens/sec
- ~0.6-1.2 sec per step
- Log output every ~6-12 seconds

#### Phase 2: MoE Model (372M total, 139M active)
```
Steps: 100
Batch size: 8
Sequence length: 256
Training: ~2-3 minutes
Reason: Sparse routing overhead, but only 139M active
```

**Expected throughput:**
- ~800-1200 tokens/sec
- ~1-1.8 sec per step
- Log output every ~10-18 seconds

#### Phase 3: Large MoE (1.47B total, 549M active)
```
Steps: 100
Batch size: 8
Sequence length: 256
Training: ~5-8 minutes
Reason: 4x larger active params than Phase 2
```

**Expected throughput:**
- ~300-500 tokens/sec
- ~3-5 sec per step
- Log output every ~30-50 seconds

## Comparison: Mac M1/M2 vs g5.2xlarge

| Metric | Mac M1/M2 | g5.2xlarge | Speedup |
|--------|-----------|------------|---------|
| **Phase 1** | ~3 min | ~1.5 min | **2×** |
| **Phase 2** | ~6 min | ~2.5 min | **2.4×** |
| **Phase 3** | ~18 min | ~6.5 min | **2.8×** |
| **Total** | ~27 min | ~10.5 min | **2.6×** |

## GPU Utilization Expectations

### Phase 1 (Dense)
- **GPU Util**: 85-95% (excellent)
- **Memory**: ~4-5 GB / 24 GB (underutilized)
- **Bottleneck**: Computation (good for A10G)

### Phase 2 (MoE)
- **GPU Util**: 70-80% (good)
- **Memory**: ~8-10 GB / 24 GB
- **Bottleneck**: Router overhead (sparse ops)

### Phase 3 (Large MoE)
- **GPU Util**: 75-85% (good)
- **Memory**: ~18-20 GB / 24 GB (well utilized)
- **Bottleneck**: Memory bandwidth

## Optimizations for g5.2xlarge

### 1. Increase Batch Size (Recommended)
```python
# config.py
batch_size: int = 16  # Double from 8
```

**Impact:**
- Phase 1: 1.5 min → **~1 min** (better GPU util)
- Phase 2: 2.5 min → **~1.8 min**
- Phase 3: 6.5 min → **~5 min**
- **Total: ~7.8 min** instead of 10.5 min

**Memory check:**
- Phase 1: ~6 GB (safe)
- Phase 2: ~12 GB (safe)
- Phase 3: ~22 GB (tight but OK)

### 2. Increase Steps for Full Training
```python
# For production runs
steps_phase1: int = 1000  # 10× more
steps_phase2: int = 1000
steps_phase3: int = 1000
```

**Time estimates (batch_size=16):**
- Phase 1: ~10 minutes
- Phase 2: ~18 minutes
- Phase 3: ~50 minutes
- **Total: ~78 minutes (~1.3 hours)**

Compare to Mac M1/M2 (batch_size=8, 1000 steps):
- Total: ~4.5 hours
- **Speedup: 3.5×**

### 3. Mixed Precision Training (Optional)
```python
# Add to train_phase()
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(input_ids)
    loss = F.cross_entropy(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Impact:**
- ~20-30% faster
- Lower memory usage
- Total time: ~6-8 min (for 100 steps)

## Cost Analysis (AWS Pricing)

**g5.2xlarge pricing:**
- On-Demand: ~$1.21/hour
- Spot Instance: ~$0.36-0.60/hour (70% savings)

**Cost for test run (100 steps, batch_size=16):**
- Time: ~8 minutes (~0.13 hours)
- On-Demand: **~$0.16**
- Spot: **~$0.05-0.08**

**Cost for full run (1000 steps, batch_size=16):**
- Time: ~78 minutes (~1.3 hours)
- On-Demand: **~$1.57**
- Spot: **~$0.47-0.78**

## Recommended Configuration for g5.2xlarge

```python
# config.py
@dataclass
class TrainingConfig:
    # For quick test (8-10 min total)
    steps_phase1: int = 100
    steps_phase2: int = 100
    steps_phase3: int = 100
    batch_size: int = 16      # Increase from 8
    
    # For full training (~1.3 hours total)
    # steps_phase1: int = 1000
    # steps_phase2: int = 1000
    # steps_phase3: int = 1000
    # batch_size: int = 16
```

## Memory Usage (g5.2xlarge with batch_size=16)

| Phase | Model | Gradients | Optimizer | Activations | Total | Available |
|-------|-------|-----------|-----------|-------------|-------|-----------|
| **1** | 0.4 GB | 0.4 GB | 0.8 GB | 0.4 GB | ~2 GB | 24 GB ✅ |
| **2** | 1.5 GB | 1.5 GB | 3.0 GB | 0.8 GB | ~7 GB | 24 GB ✅ |
| **3** | 5.9 GB | 5.9 GB | 11.8 GB | 1.5 GB | ~25 GB | 24 GB ⚠️ |

**Note:** Phase 3 is tight with batch_size=16. If you hit OOM:
- Reduce to batch_size=12
- Or enable gradient checkpointing

## Setup Commands for g5.2xlarge

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
cd /path/to/model-growth
pip3 install -r requirements.txt

# Verify CUDA is available
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA available: True
# GPU: NVIDIA A10G

# Run training
python3 train_tinystories.py
```

## Performance Monitoring

During training, monitor GPU usage:
```bash
# In a separate terminal
watch -n 1 nvidia-smi
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A10G         On   | 00000000:00:1E.0 Off |                    0 |
|  0%   45C    P0    85W / 300W |  18432MiB / 24576MiB |     82%      Default |
+-------------------------------+----------------------+----------------------+
```

## Expected Console Output Timeline

```
[00:00] Starting Phase 1...
[00:00] ============================================================
[00:00]   Phase 1 — Dense
[00:00]   Steps: 100  |  LR: 0.0003  |  Device: cuda
[00:00]   Total Parameters:   100.04 M
[00:00]   Active Params/tok:  100.04 M  (Dense model)
[00:00] ============================================================
[00:01]   step    0 | loss 10.8234 | 0.8s
[00:07]   step   10 | loss 9.2145 | 6.5s
[00:13]   step   20 | loss 8.5678 | 12.8s
...
[01:30]   step   90 | loss 6.8234 | 1.5min

[01:30] Starting Phase 2...
[01:30] ============================================================
[01:30]   Phase 2 — MoE
[01:30]   Total Parameters:   371.64 M
[01:30]   Active Params/tok:  138.88 M  (top-2 of 8 experts)
[01:30] ============================================================
[01:31]   step    0 | loss 6.8456 | 1.5s
...
[04:00]   step   90 | loss 6.2145 | 4min

[04:00] Starting Phase 3...
[04:00] [Bilateral Growth] Scaling x2 width, +4 layers...
[04:01] ✓ Growth Complete. New Config: 1664 dim, 11 layers, 16 heads.
[04:01] [Verification] Overall max difference: 9.12e-06
[04:01] ✓ PASSED — functional equivalence confirmed
[04:01] ============================================================
[04:01]   Phase 3 — Large MoE (Bilateral Growth)
[04:01]   Total Parameters:  1467.60 M
[04:01]   Active Params/tok:  548.70 M  (top-2 of 8 experts)
[04:01] ============================================================
[04:02]   step    0 | loss 6.2146 | 3.2s
...
[10:00]   step   90 | loss 5.8923 | 10min

[10:00] ✓ 3-Phase Experiment completed successfully!
```

## Summary: g5.2xlarge is 2.6× Faster

| Configuration | Mac M1/M2 | g5.2xlarge | Speedup |
|--------------|-----------|------------|---------|
| **100 steps, batch=8** | 27 min | 10.5 min | **2.6×** |
| **100 steps, batch=16** | N/A | 7.8 min | **3.5×** |
| **1000 steps, batch=8** | 4.5 hrs | 105 min | **2.6×** |
| **1000 steps, batch=16** | N/A | 78 min | **3.5×** |

**Cost for full training (1000 steps):**
- **Spot instance**: $0.47-0.78
- **On-demand**: $1.57

**Recommendation:** Use spot instance with batch_size=16 for best price/performance!
