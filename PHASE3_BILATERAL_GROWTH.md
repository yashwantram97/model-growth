# Phase 3: Bilateral Growth Implementation

## Overview

This document describes the implementation of **Bilateral Growth** strategy for scaling a Mixture-of-Experts (MoE) Small Language Model in both **width** (hidden dimensions) and **depth** (number of layers) while maintaining functional preservation to avoid loss spikes.

## What is Bilateral Growth?

Bilateral Growth is a progressive learning technique that expands a trained model's capacity through two simultaneous strategies:

1. **Width Expansion (HyperCloning)**: Increases the hidden dimension (`d_model`), FFN dimension (`d_ff`), and number of attention heads while keeping `head_dim` constant to preserve RoPE (Rotary Position Embedding).

2. **Depth Expansion (Gstack)**: Adds new transformer layers by copying the last trained layer, creating an initial identity-like pass-through that smoothly integrates during training.

## Key Features

### RoPE Safety
- The number of heads is scaled proportionally with `d_model` (e.g., 8 heads → 16 heads)
- `head_dim` remains constant (e.g., 832/8 = 104 remains 104 with 1664/16)
- RoPE rotation frequencies remain valid since they operate on `head_dim`

### Multi-Head Attention (MLA) Support
- Symmetry-breaking noise (±ε) is added to new weights
- Twin heads start as functional clones but immediately diverge
- Enables different attention patterns to emerge during Phase 3 training

### Functional Preservation
- Weight tiling with division by scale factor preserves output sums
- Initial output difference ~1e-5 (due to noise), ensuring training continuity
- Verified through functional equivalence checks on frozen probe batches

## Implementation Details

### File Structure

```
transfer/
├── simple_transfer.py      # Phase 2: MoE Upcycling
└── simple_growth.py        # Phase 3: Bilateral Growth (NEW)
```

### Core Functions

#### `hyperclone_weight(weight, scale_factor, noise_std, dim_mode)`

Expands weight matrices using three expansion modes:

- **`"both"`**: Expands both input and output dimensions (Linear layers, Attention projections)
  - Formula: `w_new = w.repeat(S, S) / S`
  - Used for: Q, K, V, O projections, Expert FFN layers

- **`"in"`**: Expands only input dimension (Router, Output Head)
  - Formula: `w_new = w.repeat(1, S) / S`
  - Used for: Router (d_model grows, num_experts constant), LM Head

- **`"out"`**: Expands only output dimension (Embeddings, RMSNorm)
  - Formula: `w_new = w.repeat(S, 1)` or `w.repeat(S)` for 1D
  - Used for: Token embeddings, LayerNorm/RMSNorm weights

**Symmetry Breaking:**
```python
if noise_std > 0.0 and dim_mode == "both":
    noise = torch.randn_like(w_new) * noise_std
    split = w_new.shape[0] // 2
    w_new[:split] += noise[:split]
    w_new[split:] -= noise[:split]  # Opposing noise
```

#### `scale_bilaterally(old_model, scale_factor, extra_layers, noise_std)`

Main function that performs the complete bilateral expansion:

**Width Expansion:**
- `d_model`: 832 → 1664 (2x)
- `d_ff`: 3328 → 6656 (2x)
- `n_heads`: 8 → 16 (2x)
- `head_dim`: 104 → 104 (constant!)

**Depth Expansion:**
- `n_layers`: 7 → 11 (+4 layers)
- New layers initialized as copies of layer 6 (last trained layer)

**Component Expansion Table:**

| Component | Old Shape | New Shape | Mode | Division |
|-----------|-----------|-----------|------|----------|
| Embeddings | (50257, 832) | (50257, 1664) | out | No |
| RMSNorm | (832,) | (1664,) | out | No |
| Q/K/V/O | (832, 832) | (1664, 1664) | both | Yes (/2) |
| Expert w1 | (832, 3328) | (1664, 6656) | both | Yes (/2) |
| Expert w2 | (3328, 832) | (6656, 1664) | both | Yes (/2) |
| Expert w3 | (832, 3328) | (1664, 6656) | both | Yes (/2) |
| Router | (832, 8) | (1664, 8) | in | Yes (/2) |
| LM Head | (50257, 832) | (50257, 1664) | in | Yes (/2) |

## Training Pipeline

### Three-Phase Training Strategy

```
Phase 1 (Steps 0-999):    Dense Transformer Training
                          └─> ~100M parameters
                          
Phase 2 (Steps 1000-1999): MoE Upcycling
                          └─> ~372M total params (~139M active)
                          
Phase 3 (Steps 2000-2999): Bilateral Growth
                          └─> ~1.4B total params (~520M active)
```

### Phase 3 Workflow

1. **Expansion**
   ```python
   large_model = scale_bilaterally(
       moe_model, 
       scale_factor=2,   # Double width
       extra_layers=4,   # Add 4 layers
       noise_std=1e-5    # Symmetry-breaking noise
   )
   ```

2. **Verification**
   - Runs functional equivalence check on frozen probe batch
   - Expected max difference: ~1e-5 (due to noise injection)
   - Ensures smooth transition without loss spikes

3. **Training**
   - 1000 steps with LR = 8e-5 (80% of Phase 2 LR)
   - Standard cross-entropy loss on TinyStories
   - Gradient clipping at 1.0

4. **Checkpointing**
   - Saves to `checkpoints/large_moe_final.pt`
   - Includes full model state and training logs

## Expected Results

### Parameter Counts

| Phase | Total Params | Active Params/Token |
|-------|--------------|---------------------|
| Phase 1 (Dense) | 100M | 100M |
| Phase 2 (MoE) | 372M | 139M |
| Phase 3 (Large MoE) | ~1.4B | ~520M |

### Loss Trajectory

Expected boundary jumps:
- **Phase 1→2**: Small jump (~0.01-0.05) due to different batch, not architecture change
- **Phase 2→3**: Tiny jump (~1e-5) due to functional preservation with noise

Successful run should show:
```
BOUNDARY SUMMARY
═══════════════════════════════════════════════════════════
  Phase 1 final loss      :  X.XXXX
  Phase 2 first loss      :  X.XXXX
  Phase 2 final loss      :  X.XXXX
  Phase 1→2 jump          :  0.XXXX (small, normal batch variation)
  Phase 2 total drop      :  X.XXXX

  Phase 3 first loss      :  X.XXXX
  Phase 3 final loss      :  X.XXXX
  Phase 2→3 jump          :  0.0000X (VERY small due to preservation)
  Phase 3 total drop      :  X.XXXX
═══════════════════════════════════════════════════════════
```

## Usage

### Running the Full Pipeline

```bash
python train_tinystories.py
```

This will:
1. Train dense model (Phase 1)
2. Convert to MoE and train (Phase 2)
3. Expand bilaterally and train (Phase 3)
4. Save all checkpoints and logs

### Using Bilateral Growth Standalone

```python
from transfer.simple_growth import scale_bilaterally
from transfer.simple_transfer import verify_functional_equivalence

# Load your trained MoE model
moe_model = torch.load("checkpoints/moe_model_final.pt")

# Expand the model
large_model = scale_bilaterally(
    moe_model,
    scale_factor=2,      # Can be 2, 3, 4, etc.
    extra_layers=4,      # Number of layers to add
    noise_std=1e-5       # Noise for breaking symmetry
)

# Verify functional preservation
probe = torch.randint(0, vocab_size, (8, 256))
verify_functional_equivalence(moe_model, large_model, probe, device)

# Continue training...
```

## Mathematical Foundations

### Why Division by Scale Factor?

For a linear layer: `y = Wx + b`

If we tile the weight: `W' = tile(W, S, S)`, then:
- Input x is also tiled S times (from width expansion upstream)
- Output would be S² times larger without normalization
- Dividing by S: `W' = tile(W, S, S) / S` preserves `y' ≈ y`

### Why Opposing Noise?

Given twins `w_A` and `w_B` from tiling:
- Without noise: `w_A = w_B` → identical gradients → no specialization
- With opposing noise: `w_A = w_B + ε`, `w_B = w_B - ε`
- Total weight sum unchanged: `w_A + w_B = 2w_B`
- Gradients differ → heads/experts specialize during training

### Head Dimension Preservation

Original: `d_model = n_heads × head_dim` (832 = 8 × 104)

After scaling with `scale_factor = 2`:
- `d_model' = 2 × 832 = 1664`
- `n_heads' = 2 × 8 = 16`
- `head_dim' = 1664 / 16 = 104` ✓ (preserved!)

RoPE applies rotations to `head_dim`, so frequencies remain valid.

## Advanced Configuration

### Custom Scale Factors

```python
# Moderate growth (1.5x)
large_model = scale_bilaterally(moe_model, scale_factor=1.5, extra_layers=2)

# Aggressive growth (3x)
large_model = scale_bilaterally(moe_model, scale_factor=3, extra_layers=8)
```

**Note:** Non-integer scale factors work but may cause slight misalignments. Recommended: 2, 3, or 4.

### Adjusting Noise Levels

```python
# Conservative (more preservation, slower specialization)
large_model = scale_bilaterally(moe_model, noise_std=1e-6)

# Aggressive (faster specialization, slightly higher initial loss)
large_model = scale_bilaterally(moe_model, noise_std=1e-4)
```

## Troubleshooting

### High Phase 2→3 Jump (> 0.01)

**Possible causes:**
- Scale factor too large
- Noise too high
- Bug in weight expansion logic

**Solutions:**
- Reduce `scale_factor` to 2
- Lower `noise_std` to 1e-6
- Check `hyperclone_weight` implementation

### Loss Spike After Phase 3 Start

**Possible causes:**
- Learning rate too high
- Model instability from extreme growth

**Solutions:**
- Reduce LR further (try 0.5x or 0.3x of Phase 2)
- Add warmup steps at beginning of Phase 3
- Check if embeddings were expanded correctly

### Poor Phase 3 Performance

**Possible causes:**
- Insufficient training steps
- Learning rate too low
- Model capacity mismatch

**Solutions:**
- Increase Phase 3 steps (e.g., 2000-3000)
- Experiment with LR in range [5e-5, 1e-4]
- Try different `extra_layers` values

## References

- FLM-101B: Progressive learning with structural growth
- Masked Structural Growth (MSG): Function-preserving expansion
- Drop-Upcycling: MoE initialization strategies
- HyperCloning: Width expansion with symmetry breaking
- Gstack: Depthwise model stacking

## Citation

If you use this implementation, please refer to:
- Explanation.md (comprehensive technical analysis)
- Original papers on bilateral growth and MoE upcycling
