"""Configuration for Dense to MoE transition experiment."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base model configuration.
    
    Target: ~100M dense parameters.
    Configuration breakdown:
        Embedding          41.81 M   (50257 × 832)
        7 blocks × 8.32 M  58.22 M   (LN + Attn + LN + FFN per block)
        Final LN            0.00 M
        LM Head             0.00 M   (tied to Embedding)
        ─────────────────────────────
        Dense total       100.04 M
        
    After MoE upcycling (8 experts, top-2):
        Total params    371.64 M   (7 FFNs × 8 copies each)
        Active/token    138.88 M   (only top-2 experts fire)
    """
    vocab_size: int = 50_257
    d_model: int = 832         # hidden dim
    n_layers: int = 7          # transformer blocks
    n_heads: int = 8           # attention heads (832 / 8 = 104 per head)
    d_ff: int = 3328           # FFN inner dim (4 × d_model)
    seq_len: int = 256         # context window
    
    # MoE specific
    num_experts: int = 8
    top_k: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training phases
    steps_phase1: int = 1000  # Dense training steps
    steps_phase2: int = 1000  # MoE training steps
    
    # Optimization
    lr_phase1: float = 3e-4
    lr_phase2: float = 1e-4  # Lower LR after upcycling
    weight_decay: float = 0.01
    batch_size: int = 8
    
    # Logging
    log_every: int = 50
    
    # Data
    dataset_name: str = "tinystories"
    num_stories: int = 50_000  # Number of TinyStories to load
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    
    # Sample generation
    n_sample_prompts: int = 3  # Number of prompts for greedy decoding
    sample_tokens: int = 40    # Tokens to generate per prompt
