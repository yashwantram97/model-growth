# Multi-Stage Growth Plan: 70M → 90M → 350M → 450M Active Parameters

## Overview
This experiment demonstrates progressive model growth through 4 phases, scaling from ~70M to ~450M active parameters using a combination of MoE upcycling, width scaling (HyperCloning), and depth scaling (Gstack).

**Growth Strategy**: Start small (512 dim), scale width first (512→1024), then scale depth (12→15 layers). This approach demonstrates both HyperCloning and Gstack while staying within 22GB GPU limits.

## Phase Progression

### Phase 1: Dense Training (~70M params)
- **Architecture**: Standard Transformer
- **Config**: 
  - d_model: 512
  - n_layers: 12
  - n_heads: 8
  - head_dim: 64 (RoPE-safe)
  - d_ff: 2048
- **Total Params**: ~70M
- **Training**: 10 steps @ LR=3e-4, batch=8

### Phase 2: MoE Upcycling (~90M active, ~240M total)
- **Architecture**: Sparse MoE
- **Operation**: Convert each FFN → 8 experts with top-2 routing
- **Config**:
  - d_model: 512
  - n_layers: 12
  - n_heads: 8
  - head_dim: 64
  - d_ff: 2048
  - num_experts: 8
  - top_k: 2
- **Active Params/token**: ~90M (top-2 of 8 experts)
- **Total Params**: ~240M
- **Efficiency**: ~38% active per forward pass
- **Training**: 10 steps @ LR=1e-4, batch=8
- **Verification**: Functional equivalence check (should be bit-exact)

### Phase 3: Width Scaling (~350M active, ~950M total)
- **Architecture**: Medium MoE
- **Operation**: HyperCloning - double width (scale_factor=2)
- **Config**:
  - d_model: 512 → **1024** (2x)
  - n_layers: 12 (unchanged)
  - n_heads: 8 → **16** (2x, maintains head_dim=64)
  - d_ff: 2048 → **4096** (2x)
  - num_experts: 8
  - top_k: 2
- **Active Params/token**: ~350M
- **Total Params**: ~950M
- **Efficiency**: ~37% active per forward pass
- **Training**: 10 steps @ LR=8e-5, batch=2
- **Verification**: 4 critical checks
  1. RoPE Integrity (head_dim preserved at 64)
  2. Router Logit Scaling (prevents collapse)
  3. Symmetry Breaking (heads diverge)
  4. Functional Identity (layer-wise preservation)
- **Memory**: 950M × 3 (Adam) × 4 bytes = ~11GB with batch=2 (fits in 22GB)

### Phase 4: Depth Scaling (~450M active, ~1.2B total)
- **Architecture**: Large MoE
- **Operation**: Gstack - add layers (extra_layers=3)
- **Config**:
  - d_model: 1024 (unchanged)
  - n_layers: 12 → **15** (1.25x depth)
  - n_heads: 16 (unchanged)
  - d_ff: 4096 (unchanged)
  - num_experts: 8
  - top_k: 2
- **Active Params/token**: ~450M
- **Total Params**: ~1.2B
- **Efficiency**: ~37% active per forward pass
- **Training**: 10 steps @ LR=5e-5, batch=1
- **Verification**: Growth mechanics check (lighter for depth-only)
- **Memory**: 1.2B × 3 (Adam) × 4 bytes = ~14GB with batch=1 (fits comfortably in 22GB)

## Growth Strategy Rationale

### Why Width First, Then Depth?
1. **Phase 3 (Width Scaling)**: Double width (512→1024)
   - Increases representational capacity per layer
   - Uses HyperCloning to preserve function
   - Maintains RoPE integrity by scaling heads proportionally (8→16, keeps head_dim=64)
   - Demonstrates the width scaling technique

2. **Phase 4 (Depth Scaling)**: Add 3 layers (12→15)
   - Further increases model capacity through depth
   - Uses Gstack (copy last layer) to initialize new layers
   - More memory-efficient than another width doubling
   - Demonstrates the depth scaling technique

**Why start with 512 dim?** Starting with smaller d_model (512 vs 832) allows us to:
- Demonstrate both width scaling (×2) and depth scaling (+3 layers)
- Stay within GPU memory limits
- Fit a 1.2B total parameter model with Adam optimizer in 22GB

### Memory Management
- **Phase 1-2**: Batch size 8 (240M total params → ~3GB with Adam)
- **Phase 3**: Batch size 2 (950M total params → ~11GB with Adam)
- **Phase 4**: Batch size 1 (1.2B total params → ~14GB with Adam)
- **Optimizer states**: Adam needs 2x model params for momentum/variance
- **Total memory at Phase 4**: 1.2B params × 3 (model + optimizer) × 4 bytes ≈ 14GB
  - Fits comfortably in 22GB GPU with batch=1 and room to spare

### Scale Factor Constraints
- **Must use integers**: HyperCloning uses `tensor.repeat()` which requires integer repetitions
- **Phase 3 uses scale_factor=2**: Doubles width (512→1024) for HyperCloning demonstration
- **Phase 4 uses scale_factor=1**: Depth-only growth via Gstack (no width scaling)

## Key Verification Points

### Phase 1→2 (Dense → MoE)
- **Expected**: Bit-exact functional equivalence
- **Why**: Router initialized to zero, each expert gets 1/8th of dense FFN

### Phase 2→3 (Width Scaling)
- **Expected**: Near-functional preservation (small differences acceptable)
- **Checks**: RoPE integrity, router scaling, symmetry breaking, layer-wise drift
- **Why**: HyperCloning with noise introduces small perturbations

### Phase 3→4 (Depth Scaling)
- **Expected**: Minimal function change initially
- **Why**: New layers are copies of trained layer (near-identity function)

## Checkpoints Saved
1. `dense_model.pt` - Phase 1 final (~70M dense)
2. `moe_model_init.pt` - Phase 2 initial (after upcycling, before training)
3. `moe_model_final.pt` - Phase 2 final (~90M active)
4. `medium_moe_final.pt` - Phase 3 final (~350M active, 12L × 1024D)
5. `large_moe_final.pt` - Phase 4 final (~450M active, 15L × 1024D)
6. `training_history.jsonl` - Full loss trajectory

## Expected Outcomes

### Loss Trajectory
- **Phase 1**: Rapid initial drop (11.0 → 7.4)
- **Phase 1→2 Jump**: Small (~0.1-0.5) - just different batch
- **Phase 2**: Continued learning (7.2 → 6.6)
- **Phase 2→3 Jump**: Small (<0.5) - HyperCloning introduces small perturbations
- **Phase 3**: Continued learning with wider model (1024 dim)
- **Phase 3→4 Jump**: Very small (<0.2) - Gstack initialization
- **Phase 4**: Continued learning with deeper + wider model

### Text Generation Quality
- **Phase 1**: Likely repetitive (only 10 steps, 70M params)
- **Phase 2**: Slight improvement with MoE (90M active)
- **Phase 3**: Better with 2x width (350M active, more representational capacity)
- **Phase 4**: Best quality (450M active, 15 layers × 1024 dim)

Note: With only 10 steps per phase, we won't see convergence. This is a proof-of-concept for growth mechanics, not production training.

## Hardware Constraint Analysis

### Achieved Target
This plan successfully demonstrates growth from 70M → 450M active parameters:
- **Phase 4 memory**: 1.2B × 3 (Adam) × 4 bytes = **~14GB**
- **Available GPU memory**: 22GB
- **Headroom**: ~8GB spare capacity ✅

### Comparison to Original 100M→1B Goal
The user originally wanted 100M → 300M → 1B progression. Due to memory constraints:
- **Achieved**: 70M → 90M → 350M → 450M (demonstrates the techniques)
- **Not attempted**: 1B active (would need ~3.3B total = 40GB with Adam)

Starting with 512 dim (vs 832) allowed us to:
- Demonstrate width scaling (×2)
- Demonstrate depth scaling (+3 layers)
- Fit both techniques in 22GB GPU
- Keep batch sizes reasonable (8 → 2 → 1)

### Alternative Approaches for Larger Models
To reach 1B+ active parameters with 22GB GPU:
1. **Use SGD instead of Adam**: Saves 2/3 of optimizer memory (~9GB → ~5GB)
2. **Use Adafactor**: More memory-efficient optimizer
3. **Use gradient checkpointing**: Trade compute for memory
4. **Use mixed precision (FP16/BF16)**: Halve memory usage
5. **Use quantization**: 8-bit weights/optimizer states
6. **Use multiple GPUs**: Distribute model across devices
