# Modular Structure: p1_p2_ts.py → train_tinystories.py

## Overview

The monolithic `p1_p2_ts.py` (725 lines) has been refactored into a clean modular structure while preserving **all** features.

## What Changed

### Before: p1_p2_ts.py (Monolithic)
```
p1_p2_ts.py (725 lines)
├── Configuration (lines 50-81)
├── TinyStoriesDataset class (lines 88-178)
├── Model classes (lines 188-318)
│   ├── FeedForward
│   ├── MoELayer
│   ├── TransformerBlock
│   └── SLM
├── Weight transfer (lines 328-372)
├── Verification (lines 381-418)
├── Training loop (lines 425-476)
├── Results printer (lines 483-570)
└── Main (lines 577-707)
```

### After: Modular Structure
```
config.py (48 lines)
├── ModelConfig
└── TrainingConfig

models/
├── simple_model.py (165 lines)
│   ├── FeedForward
│   ├── MoELayer
│   ├── TransformerBlock
│   └── SLM

utils/
└── data.py (81 lines)
    └── TinyStoriesDataset

transfer/
└── simple_transfer.py (103 lines)
    ├── transition_to_moe()
    └── verify_functional_equivalence()

train_tinystories.py (328 lines)
├── get_device()
├── train_phase()
├── print_phase_results()
└── main()
```

## Feature Mapping

| Feature | p1_p2_ts.py Location | Modular Location |
|---------|---------------------|------------------|
| Configuration | Lines 50-81 | `config.py` |
| TinyStories Dataset | Lines 88-178 | `utils/data.py` |
| FeedForward | Lines 188-200 | `models/simple_model.py` |
| MoELayer | Lines 203-260 | `models/simple_model.py` |
| TransformerBlock | Lines 263-286 | `models/simple_model.py` |
| SLM Model | Lines 289-318 | `models/simple_model.py` |
| Weight Transfer | Lines 328-372 | `transfer/simple_transfer.py` |
| Verification | Lines 381-418 | `transfer/simple_transfer.py` |
| Training Loop | Lines 425-476 | `train_tinystories.py` |
| Results Printer | Lines 483-570 | `train_tinystories.py` |
| Device Selection | Lines 577-599 | `train_tinystories.py` |
| Phase 1 Training | Lines 594-613 | `train_tinystories.py` |
| Phase 2 Training | Lines 647-662 | `train_tinystories.py` |
| Boundary Summary | Lines 667-680 | `train_tinystories.py` |
| Log Saving | Lines 684-693 | `train_tinystories.py` |

## Benefits of Modular Structure

### 1. Reusability
```python
# Import just what you need
from models.simple_model import SLM
from utils.data import TinyStoriesDataset
from transfer.simple_transfer import transition_to_moe

# Use in your own script
model = SLM(vocab_size=50257, d_model=256, ...)
dataset = TinyStoriesDataset(num_stories=10000)
moe_model = transition_to_moe(dense_model, num_experts=8, top_k=2)
```

### 2. Testability
Each module can be tested independently:
```python
# Test dataset loading
from utils.data import TinyStoriesDataset
dataset = TinyStoriesDataset(num_stories=100)
dataset.tokenize()
batch = dataset.get_batch(4, 128, "cpu")
assert batch[0].shape == (4, 128)

# Test model forward pass
from models.simple_model import SLM
model = SLM(vocab_size=50257, d_model=128, n_layers=2, n_heads=2, d_ff=512)
output = model(torch.randint(0, 50257, (2, 64)))
assert output.shape == (2, 64, 50257)

# Test weight transfer
from transfer.simple_transfer import transition_to_moe
moe_model = transition_to_moe(dense_model, num_experts=4, top_k=2)
```

### 3. Maintainability
- Each file has a single responsibility
- Easy to locate and fix bugs
- Changes to one component don't affect others
- Clear imports show dependencies

### 4. Extensibility
Add new features without modifying existing code:
```python
# Add a new dataset
class WikiTextDataset:
    """utils/data.py"""
    def __init__(self, split="train"):
        ...

# Add a new model variant
class SLMWithDropout(SLM):
    """models/simple_model.py"""
    def __init__(self, ..., dropout=0.1):
        ...

# Add new verification metrics
def compute_expert_usage(moe_model, test_input):
    """transfer/simple_transfer.py"""
    ...
```

### 5. Documentation
- Each module has focused docstrings
- README explains overall structure
- Easy to understand one piece at a time

## All Features Preserved

✅ **TinyStories Dataset**
- Still uses HuggingFace datasets
- Still uses GPT-2 tokenizer
- Still concatenates into token stream
- Still samples random windows

✅ **Simple Architecture**
- Still uses LayerNorm (not RMSNorm)
- Still uses GELU (not SwiGLU)
- Still uses nn.MultiheadAttention
- Exact same forward pass

✅ **Two-Phase Training**
- Phase 1: Dense model (1000 steps)
- Phase 2: MoE model (1000 steps)
- Same loss computation
- Same optimizer settings

✅ **Weight Transfer**
- Identical upcycling logic
- Clone FFN to all experts
- Zero router for uniform distribution
- Preserves functional equivalence

✅ **Verification**
- Frozen probe batch
- Per-sequence difference checking
- 1e-4 tolerance
- Detailed logging

✅ **Greedy Decoding**
- Sample prompts after each phase
- 40 tokens generated
- Same TinyStories-style prompts
- Token-by-token generation

✅ **MPS Support**
- Apple Silicon GPU acceleration
- Automatic device detection
- Cache management

✅ **Logging**
- Loss tracking every 50 steps
- Phase boundary analysis
- JSONL format logs
- Checkpoint saving

## How to Use

### Run the modular version:
```bash
python train_tinystories.py
```

### Or use components individually:
```python
from config import ModelConfig, TrainingConfig
from models.simple_model import SLM
from utils.data import TinyStoriesDataset
from transfer.simple_transfer import transition_to_moe

# Create your custom experiment
config = ModelConfig(d_model=512, n_layers=6)
model = SLM(config.vocab_size, config.d_model, ...)
dataset = TinyStoriesDataset(num_stories=100000)
# ... your custom training loop ...
```

## Comparison

| Aspect | p1_p2_ts.py | Modular |
|--------|-------------|---------|
| Total Lines | 725 | ~725 (split across files) |
| Files | 1 | 7 |
| Largest File | 725 lines | 328 lines |
| Configuration | Hardcoded dict | Dataclass |
| Imports | All in one file | Explicit per module |
| Testing | Hard to isolate | Easy per module |
| Reusability | Copy-paste | Import |
| Documentation | One big docstring | Per-module docs |
| Git History | All changes mixed | Clear by module |

## Migration Guide

If you have custom modifications to `p1_p2_ts.py`:

1. **Config changes** → Edit `config.py`
2. **Model architecture** → Edit `models/simple_model.py`
3. **Dataset changes** → Edit `utils/data.py`
4. **Transfer logic** → Edit `transfer/simple_transfer.py`
5. **Training loop** → Edit `train_tinystories.py`

## Backward Compatibility

The original `p1_p2_ts.py` still works! You can:
- Keep using it if you prefer
- Migrate gradually
- Compare outputs between versions
- Use both for different experiments
