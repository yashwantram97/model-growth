# Model Growth: Dense → MoE Training Pipeline

A PyTorch implementation of progressive model growth through four phases: Dense training, MoE upcycling, width scaling, and depth scaling. This demonstrates functional-preserving model architecture transitions during training.

## Overview

This repository implements a 4-phase training pipeline:

1. **Phase 1 (Dense)**: Train a dense transformer (~70M params, 12 layers × 512 dim)
2. **Phase 2 (MoE Upcycling)**: Convert FFN layers to 8-expert MoE with top-2 routing (~90M active, ~240M total)
3. **Phase 3 (Width Scaling)**: Double model width via bilateral growth (512→1024 dim, ~350M active)
4. **Phase 4 (Depth Scaling)**: Add layers via Gstack (12→15 layers, ~450M active)

### Key Features

- **Functional Preservation**: Growth transitions maintain model outputs (verified at each phase)
- **MoE Efficiency**: Use only ~30% of parameters per forward pass (top-k routing)
- **Progressive Growth**: Scale models during training without starting from scratch
- **Real Dataset**: Uses TinyStories dataset for meaningful learning signals

## Quick Start

### Installation

Using `uv` (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -r requirements.txt
```

### Training

Using `uv`:
```bash
uv run train.py
```

Or directly:
```bash
python train.py
```

This will:
- Download TinyStories dataset automatically
- Train through all 4 phases
- Save checkpoints to `./checkpoints/`
- Log training metrics to `./checkpoints/training_history.jsonl`

### Configuration

Edit `config.py` to adjust:
- Model architecture (dimensions, layers, experts)
- Training hyperparameters (steps, learning rates, batch sizes)
- Dataset settings (number of stories)

## Architecture

### Directory Structure

```
model-growth/
├── train.py                   # Main training script (4 phases)
├── config.py                  # Configuration dataclasses
├── pyproject.toml            # uv/pip package configuration
├── requirements.txt          # Python dependencies
├── models/
│   ├── __init__.py
│   └── model.py              # SLM: Dense/MoE transformer
├── transfer/
│   ├── __init__.py
│   ├── transfer.py           # Dense→MoE upcycling
│   ├── growth.py             # Bilateral growth (width + depth)
│   └── verify_growth_mechanics.py  # Verification tests
└── utils/
    ├── __init__.py
    └── data.py               # TinyStories dataset loader
```

### Model Components

**SLM (Simple Language Model)** - `models/model.py`
- Configurable dense/MoE transformer
- Supports top-k expert routing
- Shared embeddings (input/output)

**MoE Block** - Expert routing with:
- 8 experts per FFN layer
- Top-2 routing (highest scoring experts)
- Load balancing loss

## Training Phases

### Phase 1: Dense Training
- **Architecture**: 12 layers × 512 dim
- **Parameters**: ~70M
- **Goal**: Establish baseline model

### Phase 2: MoE Upcycling
- **Transition**: Dense FFN → 8-expert MoE
- **Parameters**: ~90M active (out of ~240M total)
- **Verification**: Output identity on frozen probe batch
- **Key Insight**: No loss spike at transition

### Phase 3: Width Scaling
- **Growth**: 512 → 1024 dimensions (bilateral)
- **Parameters**: ~350M active (out of ~950M total)
- **Method**: Double attention/FFN widths, add noise for symmetry breaking
- **Efficiency**: 37% active parameters per token

### Phase 4: Depth Scaling
- **Growth**: 12 → 15 layers (Gstack)
- **Parameters**: ~450M active (out of ~1.2B total)
- **Method**: Insert new layers (identity-initialized)
- **Memory**: ~14GB with Adam (fits 22GB GPU)

## Growth Mechanics

### Bilateral Growth (Width Scaling)
Implemented in `transfer/growth.py`:
1. **Attention**: Tile query/key/value/output projection weights
2. **FFN**: Tile up/down projection weights
3. **Embeddings**: Tile token/position embeddings
4. **Noise**: Add small noise (1e-4) for symmetry breaking

### Gstack (Depth Scaling)
Implemented in `transfer/growth.py`:
1. **Insert Layers**: Add new transformer blocks
2. **Initialize**: Near-identity (residual path dominates)
3. **Preserve**: Original layers unchanged

### Verification
Implemented in `transfer/verify_growth_mechanics.py`:
- Output equivalence checks
- Weight structure validation
- Router initialization verification
- Detailed diagnostic logging

## Hardware Requirements

- **Phase 1-2**: ~2GB GPU memory
- **Phase 3**: ~8GB GPU memory (batch size 2)
- **Phase 4**: ~14GB GPU memory (batch size 1)

Tested on:
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- CPU (slower)

## Dataset: TinyStories

- **Source**: roneneldan/TinyStories via HuggingFace
- **Size**: 2.1M short stories
- **Vocabulary**: Simple English (GPT-2 tokenizer, 50,257 tokens)
- **Context**: 256 tokens
- **Why**: Provides meaningful learning signal vs random tokens

## Output Files

```
checkpoints/
├── dense_model.pt          # Phase 1 final model
├── moe_model_init.pt       # Phase 2 initial (post-upcycling)
├── moe_model_final.pt      # Phase 2 final
├── medium_moe_final.pt     # Phase 3 final (~350M active)
├── large_moe_final.pt      # Phase 4 final (~450M active)
└── training_history.jsonl  # Loss curves for all phases
```

## Key Results

### Loss Progression
- **Phase 1**: Dense model learns language patterns
- **Phase 1→2 Transition**: No loss spike (functional equivalence verified)
- **Phase 2**: MoE continues learning with expert specialization
- **Phase 2→3 Transition**: Minimal spike (<0.5 loss increase)
- **Phase 3**: Wider model learns more efficiently
- **Phase 3→4 Transition**: Minimal spike (Gstack initialization)
- **Phase 4**: Deeper model reaches lower final loss

### Parameter Efficiency (MoE)
- **Total Parameters**: 240M → 950M → 1.2B (across phases)
- **Active per Token**: 90M → 350M → 450M (only top-2 experts)
- **Efficiency**: ~30-40% parameters active per forward pass

## Development

### Adjusting Training
Edit `config.py`:
```python
@dataclass
class TrainingConfig:
    steps_phase1: int = 1000    # Increase for better convergence
    steps_phase2: int = 1000
    steps_phase3: int = 500
    steps_phase4: int = 500
    batch_size: int = 8         # Adjust for GPU memory
```

## Implementation Details

### Weight Initialization
- **Dense→MoE**: Copy dense weights to all experts
- **Router**: Xavier uniform (balanced initial routing)
- **Growth**: Tile existing weights + small noise

### Optimization
- **Optimizer**: AdamW (weight_decay=0.01)
- **Gradient Clipping**: 1.0 norm
- **Learning Rate**: Decays across phases (3e-4 → 5e-5)
- **Batch Size**: Reduces for larger models

### Verification Strategy
1. **Pre-transition**: Save probe batch outputs
2. **Post-transition**: Compare outputs on same batch
3. **Tolerance**: <1e-4 for upcycling, <0.5 for growth (noise-adjusted)

## References

- **MoE Upcycling**: Based on functional-preserving FFN→MoE conversion
- **Bilateral Growth**: Width scaling via weight tiling
- **Gstack**: Depth scaling via layer insertion
- **TinyStories**: Dataset from Eldan & Li (2023)

## License

MIT License - See project repository for details.
