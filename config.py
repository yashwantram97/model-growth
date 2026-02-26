"""Configuration for Dense to MoE transition experiment."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base model configuration.
    
    Target: ~40M dense parameters (Option A2: wider, shallower, standard 4× FFN).
    Configuration breakdown:
        Embedding          25.7 M    (50257 × 512)
        5 blocks × 3.3 M   16.5 M    (LN + Attn + LN + FFN per block)
        Final LN            0.00 M
        LM Head             0.00 M   (tied to Embedding)
        ─────────────────────────────
        Dense total       ~42 M
        
    After MoE upcycling (8 experts, top-2):
        Total params    ~170 M   (5 FFNs × 8 copies each)
        Active/token    ~62 M    (only top-2 experts fire)
    """
    vocab_size: int = 50_257
    d_model: int = 512         # hidden dim (512 -> 512 -> 1024 -> 1024 across phases)
    n_layers: int = 5          # transformer blocks (fewer layers, more width, 4× FFN)
    n_heads: int = 4           # attention heads (512 / 4 = 128 per head)
    d_ff: int = 2048           # FFN inner dim (4 × d_model, standard ratio)
    seq_len: int = 256         # context window
    
    # MoE specific
    num_experts: int = 8
    top_k: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training phases
    steps_phase1: int = 1000   # Dense training steps (~24M params, 12L × 256D)
    steps_phase2: int = 1000   # MoE training steps (~30M active, 12L × 256D)
    steps_phase3: int = 1000   # Medium MoE growth (~120M active, 12L × 512D)
    steps_phase4: int = 1000   # Large MoE growth (~150M active, 15L × 512D)
    
    # Optimization
    lr_phase1: float = 3e-4
    lr_phase2: float = 1e-4   # Lower LR after upcycling
    lr_phase3: float = 8e-5   # Lower LR for medium model
    lr_phase4: float = 5e-5   # Even lower LR for large model
    weight_decay: float = 0.01
    batch_size: int = 2
    batch_size_phase3: int = 2  # Smaller batch for medium model (~680M total, 5L × 1024D)
    batch_size_phase4: int = 2  # Smallest batch for large model (~1.1B total, 8L × 1024D)
    
    # Logging
    log_every: int = 100
    
    # Data
    dataset_name: str = "tinystories"
    num_stories: int = 50_000  # Number of TinyStories to load
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    
    # Sample generation
    n_sample_prompts: int = 3  # Number of prompts for greedy decoding
    sample_tokens: int = 40    # Tokens to generate per prompt
