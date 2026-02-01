# Dense to MoE Transition Experiment

A modular implementation demonstrating spike-free weight transfer from a dense transformer to a Mixture of Experts (MoE) model, following SmolLM-style architecture.

## Overview

This experiment validates the **functional preservation** principle during model growth:
1. **Phase 1**: Train a dense transformer for 1000 steps
2. **Phase 2**: Transfer weights to MoE by copying FFN to all experts
3. **Phase 3**: Train MoE for 1000 steps

**Key Goal**: Loss should continue decreasing without spiking during the transition.

## Project Structure

```
Minimal-exp-growth/
├── config.py                    # Model and training configurations
├── train.py                     # Main training script
├── requirements.txt             # Dependencies
├── models/
│   ├── __init__.py
│   ├── dense_model.py          # Dense transformer (SmolLM-style)
│   └── moe_model.py            # MoE transformer
├── transfer/
│   ├── __init__.py
│   └── weight_transfer.py      # Weight transfer logic
└── utils/
    ├── __init__.py
    └── data.py                  # Data loading utilities
```

## Model Architecture (SmolLM-style)

### Dense Model Features:
- **RMSNorm**: Root Mean Square Layer Normalization
- **RoPE**: Rotary Position Embeddings
- **SwiGLU**: Gated activation in FFN (w1, w2, w3 weights)
- **Pre-norm architecture**: LayerNorm before attention and FFN
- **Weight tying**: Embeddings tied with output head

### MoE Model Features:
- **Top-k routing**: Selects k experts per token
- **Expert replication**: All experts initialized as copies of dense FFN
- **Load balancing**: Auxiliary loss for uniform expert usage
- **Sparse computation**: Only top-k experts active per token

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python train.py
```

### Custom Configuration

Edit `config.py` or modify in `train.py`:

```python
from config import ModelConfig, TrainingConfig

model_config = ModelConfig(
    vocab_size=50257,
    d_model=768,      # Hidden dimension
    n_layers=12,      # Number of layers
    n_heads=12,       # Attention heads
    d_ff=3072,        # FFN dimension
    num_experts=8,    # Number of experts in MoE
    top_k=2          # Top-k routing
)

training_config = TrainingConfig(
    dense_steps=1000,
    moe_steps=1000,
    learning_rate=1e-4,
    batch_size=8,
    seq_length=128
)
```

### Output

The experiment produces:
- **Checkpoints**: `./checkpoints/`
  - `dense_model.pt`: Trained dense model
  - `moe_model_init.pt`: MoE after weight transfer
  - `moe_model_final.pt`: MoE after training
  - `training_history.json`: Loss curves
  
- **Console Output**:
  - Training progress with loss values
  - Weight transfer verification
  - Functional identity check
  - Expert usage statistics

## Weight Transfer Mechanism

The `transfer_dense_to_moe()` function ensures functional preservation:

1. **Copy Embeddings**: Token embeddings → MoE embeddings
2. **Copy Attention**: All attention weights transferred unchanged
3. **Replicate FFN**: Dense FFN → All MoE experts (identical copies)
4. **Initialize Router**: Near-zero weights for uniform expert selection
5. **Copy Norms**: All layer normalizations transferred

**Result**: MoE model produces nearly identical outputs to dense model immediately after transfer.

## Key Implementation Details

### 1. Functional Identity

After weight transfer, verify that:
```
dense_output ≈ moe_output (within tolerance)
```

This is achieved by:
- Copying FFN to all experts
- Initializing router near zero (uniform distribution)
- When all experts are identical and equally weighted, MoE = Dense

### 2. Load Balancing Loss

```python
aux_loss = mean((expert_usage - 1/num_experts)²)
```

Encourages uniform expert usage to prevent expert collapse.

### 3. SwiGLU Activation

```python
output = w2(dropout(silu(w1(x)) * w3(x)))
```

Three weight matrices (w1, w2, w3) instead of two for better performance.

## Expected Results

### Success Criteria:
✓ Loss decreases smoothly during dense training  
✓ No loss spike during transition to MoE  
✓ Functional identity preserved (mean diff < 1e-3)  
✓ Loss continues decreasing during MoE training  
✓ MoE achieves lower final loss than dense model  

### Parameter Comparison (Example: 512d × 6L):
- Dense model: ~24M parameters
- MoE model: ~96M total, ~30M active per token
- Expansion ratio: 4x total, 1.25x active

## Experiments

### Experiment 1: Baseline (512d × 6L)
```bash
python train.py
```

### Experiment 2: Larger Model (768d × 12L)
Modify `config.py`:
```python
d_model=768, n_layers=12, n_heads=12, d_ff=3072
```

### Experiment 3: More Experts
```python
num_experts=16, top_k=2
```

## Visualization

To visualize training curves:

```python
import json
import matplotlib.pyplot as plt

with open('checkpoints/training_history.json') as f:
    history = json.load(f)

plt.plot(history['dense_steps'], history['dense_losses'], label='Dense')
plt.plot([1000] + [1000 + s for s in history['moe_steps']], 
         [history['dense_losses'][-1]] + history['moe_losses'], 
         label='MoE')
plt.axvline(x=1000, color='r', linestyle='--', label='Transfer')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Dense → MoE Transition (Spike-Free)')
plt.savefig('loss_curve.png')
```

## Troubleshooting

### Loss Spike Detected
If loss spikes during transition:
- Check router initialization (should be near zero)
- Verify all experts are identical copies
- Ensure weight transfer completed successfully

### Out of Memory
Reduce:
- `batch_size` in `TrainingConfig`
- `d_model` or `n_layers` in `ModelConfig`
- `seq_length`

### Slow Training
- Reduce `d_ff` (FFN dimension)
- Use fewer experts
- Enable mixed precision training (add AMP)

## References

Based on research from:
- **FLM-101B**: Progressive learning and structural growth
- **Masked Structural Growth**: Spike-free model expansion
- **Sparse Upcycling**: Dense to MoE conversion
- **SmolLM**: Modern small language model architecture

See `Explanation.md` for detailed technical analysis.

## License

MIT

## Citation

```bibtex
@misc{dense2moe2026,
  title={Dense to MoE Transition with Functional Preservation},
  author={Your Name},
  year={2026}
}
```
