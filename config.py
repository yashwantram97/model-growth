"""Configuration for Dense to MoE transition experiment."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base model configuration.
    
    Target: ~70M dense parameters (starting smaller for memory efficiency).
    Configuration breakdown:
        Embedding          25.7 M    (50257 × 512)
        12 blocks × 3.7 M  44.4 M    (LN + Attn + LN + FFN per block)
        Final LN            0.00 M
        LM Head             0.00 M   (tied to Embedding)
        ─────────────────────────────
        Dense total       ~70 M
        
    After MoE upcycling (8 experts, top-2):
        Total params    ~240 M   (12 FFNs × 8 copies each)
        Active/token    ~90 M    (only top-2 experts fire)
    """
    vocab_size: int = 50_257
    d_model: int = 512         # hidden dim (smaller for memory efficiency)
    n_layers: int = 12         # transformer blocks (more layers, less width)
    n_heads: int = 8           # attention heads (512 / 8 = 64 per head)
    d_ff: int = 2048           # FFN inner dim (4 × d_model)
    seq_len: int = 256         # context window
    
    # MoE specific
    num_experts: int = 8
    top_k: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training phases
    steps_phase1: int = 10   # Dense training steps (~70M params, 12L × 512D)
    steps_phase2: int = 10   # MoE training steps (~90M active, 12L × 512D)
    steps_phase3: int = 10   # Medium MoE growth (~350M active, 12L × 1024D)
    steps_phase4: int = 10   # Large MoE growth (~450M active, 15L × 1024D)
    
    # Optimization
    lr_phase1: float = 3e-4
    lr_phase2: float = 1e-4   # Lower LR after upcycling
    lr_phase3: float = 8e-5   # Lower LR for medium model
    lr_phase4: float = 5e-5   # Even lower LR for large model
    weight_decay: float = 0.01
    batch_size: int = 8
    batch_size_phase3: int = 2  # Smaller batch for medium model (~950M total, 12L × 1024D)
    batch_size_phase4: int = 1  # Smallest batch for large model (~1.2B total, 15L × 1024D)
    
    # Logging
    log_every: int = 1
    
    # Data
    dataset_name: str = "tinystories"
    num_stories: int = 50_000  # Number of TinyStories to load
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    
    # Sample generation
    n_sample_prompts: int = 3  # Number of prompts for greedy decoding
    sample_tokens: int = 40    # Tokens to generate per prompt
