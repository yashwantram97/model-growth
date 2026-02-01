# Dense to MoE Transition Experiment

This project implements a modular framework for training dense transformer models and transitioning them to Mixture-of-Experts (MoE) models without loss spikes.

## Project Structure

```
.
├── config.py                    # Configuration dataclasses
├── models/
│   ├── __init__.py
│   ├── dense_model.py           # SmolLM-style dense transformer (RoPE, RMSNorm, SwiGLU)
│   ├── moe_model.py             # SmolLM-style MoE transformer
│   └── simple_model.py          # Simple transformer (LayerNorm, GELU) from p1_p2_ts.py
├── utils/
│   ├── __init__.py
│   └── data.py                  # TinyStories dataset loader
├── transfer/
│   ├── __init__.py
│   ├── weight_transfer.py       # Transfer for SmolLM-style models
│   └── simple_transfer.py       # Transfer for simple models
├── train.py                     # Training script for SmolLM-style models
└── train_tinystories.py         # Training script for TinyStories with simple models
```

## Features

### From p1_p2_ts.py (Now Modular)

All features from the original `p1_p2_ts.py` script are preserved in the modular version:

- ✅ **TinyStories Dataset**: Real-world dataset with GPT-2 tokenizer
- ✅ **Two-Phase Training**: Dense (Phase 1) → MoE (Phase 2)
- ✅ **Functional Equivalence Verification**: Frozen probe batch to verify identical outputs
- ✅ **Greedy Decoding Samples**: Generate text after each phase
- ✅ **MPS Support**: Apple Silicon GPU acceleration
- ✅ **Comprehensive Logging**: Loss tracking, boundary analysis
- ✅ **Simple Architecture**: LayerNorm + GELU (easy to understand)

### Additional Modular Benefits

- 🔧 **Configurable**: Easy to modify via `config.py`
- 📦 **Reusable Components**: Import models, datasets, transfer functions
- 🧪 **Multiple Architectures**: Simple vs SmolLM-style
- 💾 **Checkpointing**: Save/load models at any stage

## Quick Start

### 1. Install Dependencies

```bash
pip install torch datasets transformers tokenizers
```

### 2. Run TinyStories Experiment

```bash
python train_tinystories.py
```

This will:
1. Load and tokenize 50,000 TinyStories (~11M tokens)
2. Train a 100M parameter dense model (1000 steps)
3. Transition to MoE with 8 experts, top-2 routing
   - Total: 371.64M parameters (8 copies of FFN layers)
   - Active: 138.88M parameters per token (only top-2 experts)
4. Verify functional equivalence with frozen probe
5. Train MoE model (1000 steps)
6. Generate sample text after each phase
7. Save checkpoints and logs to `./checkpoints/`

### 3. Run SmolLM-Style Experiment (Advanced)

```bash
python train.py
```

Uses more sophisticated architecture with:
- Rotary Position Embeddings (RoPE)
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of GELU

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class ModelConfig:
    """~100M parameter dense model configuration"""
    vocab_size: int = 50_257     # GPT-2 vocabulary
    d_model: int = 832           # Hidden dimension
    n_layers: int = 7            # Number of layers
    n_heads: int = 8             # Attention heads (832/8 = 104 per head)
    d_ff: int = 3328             # FFN inner dimension (4 × d_model)
    seq_len: int = 256           # Sequence length
    num_experts: int = 8         # Experts per MoE layer
    top_k: int = 2               # Active experts per token

@dataclass
class TrainingConfig:
    steps_phase1: int = 1000     # Dense training steps
    steps_phase2: int = 1000     # MoE training steps
    lr_phase1: float = 3e-4      # Phase 1 learning rate
    lr_phase2: float = 1e-4      # Phase 2 learning rate (lower)
    batch_size: int = 8
    log_every: int = 50
    num_stories: int = 50_000    # TinyStories to load
```

### Smaller Configuration (For Testing)

If you want to test quickly with fewer resources, edit `config.py`:

```python
@dataclass
class ModelConfig:
    """~16M parameter model (faster for testing)"""
    vocab_size: int = 50_257
    d_model: int = 256           # Smaller hidden dimension
    n_layers: int = 4            # Fewer layers
    n_heads: int = 4
    d_ff: int = 1024
    seq_len: int = 256
    num_experts: int = 8
    top_k: int = 2
```

## Device Support

Automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs)
2. **MPS** (Apple Silicon M1/M2/M3)
3. **CPU** (fallback)

## Key Components

### Models

**Simple Model** (`models/simple_model.py`):
- Standard LayerNorm
- GELU activation
- PyTorch's MultiheadAttention
- Easier to understand and debug

**SmolLM-Style Model** (`models/dense_model.py` + `models/moe_model.py`):
- RoPE (Rotary Position Embeddings)
- RMSNorm (Root Mean Square Normalization)
- SwiGLU activation
- Custom attention implementation

### Weight Transfer

The key to loss-free transition is the `transition_to_moe()` function:

1. **Copy embeddings** and output head
2. **Copy attention** weights (identical in both)
3. **Copy layer norms** (identical in both)
4. **Replicate FFN** to all experts (every expert = original FFN)
5. **Zero router** → uniform softmax → MoE(x) ≈ FFN(x)

### Dataset

**TinyStoriesDataset** (`utils/data.py`):
- Loads stories from HuggingFace
- Tokenizes with GPT-2 tokenizer
- Concatenates into one long stream
- Serves random windows for training
- ~2.1M short stories in simple English

## Output

After running `train_tinystories.py`, you'll see:

```
[init] Device: mps
[init] GPU:    Apple Silicon (MPS)
[dataset] Loading tokenizer (gpt2) …
[dataset] Loading TinyStories …
[dataset] Tokenizing 50,000 stories …
[dataset] Done. 12,345,678 total tokens from 50,000 stories.

============================================================
  PHASE 1  —  Dense Model  (single FFN per block)
============================================================
  Steps: 1000  |  LR: 0.0003  |  Device: mps
  Params: 100.04 M
============================================================
  step    0 | loss 10.8234 | 0.5s
  step   50 | loss 8.4512 | 12.3s
  ...

RESULTS — Phase 1 — Dense
------------------------------------------------------------
  Loss:        10.8234  →  6.2145  (dropped 4.6089)
  Total params: 100.04 M

  Sample greedy decoding (40 new tokens each):
  --------------------------------------------------------
  Prompt:    "Once upon a time there was"
  Generated: " a little girl named Lucy. She loved to play..."
  --------------------------------------------------------

============================================================
  TRANSITION  —  Dense → MoE Upcycling
============================================================

  [Verification] Per-sequence max |logit_dense − logit_moe|:
    Seq 0: 8.34e-05  OK
    Seq 1: 7.21e-05  OK
    ...
  Overall max difference: 8.34e-05
  ✓ PASSED — functional equivalence confirmed (< 0.0001)

============================================================
  PHASE 2  —  MoE Model  (8 experts, top-2 routing)
============================================================
  ...

BOUNDARY SUMMARY
============================================================
  Phase 1 final loss  :  6.2145
  Phase 2 first loss  :  6.2234
  Phase 2 final loss  :  5.1823
  Boundary jump       :  0.0089
    (small jump is normal — it's a different random batch,
     NOT a loss spike from the conversion)
  Phase 2 total drop  :  1.0411
============================================================
```

## Checkpoints

Saved to `./checkpoints/`:
- `dense_model.pt` - Trained dense model
- `moe_model_init.pt` - MoE right after transition
- `moe_model_final.pt` - Trained MoE model
- `training_history.jsonl` - Loss log for plotting

## Next Steps

1. **Visualize Results**: Plot loss curves from `training_history.jsonl`
2. **Experiment**: Try different architectures, learning rates, expert counts
3. **Scale Up**: Increase model size, training steps, dataset size
4. **Analyze Experts**: Study expert specialization over training

## Credits

Based on the Dense → MoE transition technique with functional preservation guarantees, implemented in both simple and advanced architectures.
