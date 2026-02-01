"""
Bilateral Growth Implementation for model.py
Scales Width (HyperCloning) and Depth (Gstack) while preserving function.
"""

import torch
import torch.nn as nn
from models.model import SLM, TransformerBlock, MoELayer


def hyperclone_weight(weight, scale_factor=2, noise_std=0.0, dim_mode="both"):
    """
    Expands a weight matrix using HyperCloning with Symmetry Breaking.
    
    Args:
        weight: Original weight tensor
        scale_factor: How much to expand (must be integer, usually 2)
        noise_std: Standard deviation for symmetry-breaking noise
        dim_mode: 
            - 'both': Expand input and output (Linear layers, Attention)
            - 'in': Expand input only (Router, Output Head)
            - 'out': Expand output only (Embeddings, RMSNorm)
    
    Returns:
        Expanded weight tensor
    """
    # Ensure scale_factor is an integer for repeat operations
    scale_factor = int(scale_factor)
    if scale_factor < 1:
        raise ValueError(f"scale_factor must be >= 1, got {scale_factor}")
    
    with torch.no_grad():
        w = weight.data
        
        # 1. Tile the weights based on mode
        if dim_mode == "both":
            # Shape: [Out, In] -> [Out*S, In*S]
            w_new = w.repeat(scale_factor, scale_factor)
            # Divide by scale to preserve dot product sum
            w_new = w_new / scale_factor
            
        elif dim_mode == "in":
            # Shape: [Out, In] -> [Out, In*S] (e.g., Router)
            w_new = w.repeat(1, scale_factor)
            # Divide by scale because input 'x' magnitude effectively doubled
            w_new = w_new / scale_factor
            
        elif dim_mode == "out":
            # Shape: [Out, In] -> [Out, In*S] (e.g., Embeddings)
            # Or [Dim] -> [Dim*S] (RMSNorm)
            if w.dim() == 1:
                w_new = w.repeat(scale_factor)
            else:
                w_new = w.repeat(1, scale_factor)
            # No division needed for output expansion (magnitude preserved by Norm later)
            
        # 2. Symmetry Breaking (Noise)
        if noise_std > 0.0 and dim_mode == "both":
            noise = torch.randn_like(w_new) * noise_std
            
            # Split and apply opposing noise to create distinct "heads/experts"
            # Example: Head A gets +noise, Head B gets -noise
            split = w_new.shape[0] // 2
            w_new[:split] += noise[:split]
            w_new[split:] -= noise[:split]
            
        return w_new


@torch.no_grad()
def scale_bilaterally(old_model, scale_factor=2, extra_layers=4, noise_std=1e-5):
    """
    Grows an SLM model in width and depth.
    
    Args:
        old_model: The trained small MoE model.
        scale_factor: Multiplier for width (d_model, n_heads, d_ff). Must be integer >= 1.
        extra_layers: Number of new layers to stack (Gstack).
        noise_std: Amount of noise to break symmetry.
        
    Returns:
        new_model: The expanded, function-preserving model.
    """
    # Validate scale_factor is an integer
    if not isinstance(scale_factor, int) and scale_factor != int(scale_factor):
        raise ValueError(f"scale_factor must be an integer (got {scale_factor}). "
                        f"HyperCloning uses tensor.repeat() which requires integer repetitions.")
    scale_factor = int(scale_factor)
    
    print(f"\n[Bilateral Growth] Scaling x{scale_factor} width, +{extra_layers} layers...")
    
    # --- 1. Calculate New Config ---
    config = {
        'vocab_size': old_model.embed.num_embeddings,
        'd_model': int(old_model.embed.embedding_dim * scale_factor),
        'n_layers': len(old_model.blocks) + extra_layers,
        'n_heads': int(old_model.blocks[0].attention.n_heads * scale_factor),  # Scale heads to keep head_dim constant (RoPE safe)
        'd_ff': int(old_model.blocks[0].ffn.experts[0].w1.out_features * scale_factor),
        'use_moe': True,
        'num_experts': old_model.blocks[0].ffn.num_experts,
        'top_k': old_model.blocks[0].ffn.top_k,
        'dropout': old_model.blocks[0].attention.dropout.p,
        'max_seq_len': old_model.blocks[0].attention.rotary_emb.max_seq_len
    }
    
    new_model = SLM(**config)
    
    # --- 2. Expand Shared Components ---
    
    # Embeddings (Case 1: Output only)
    new_model.embed.weight.copy_(
        hyperclone_weight(old_model.embed.weight, scale_factor, 0, "out")
    )
    
    # Final Norm (Case 1: Output only)
    new_model.norm_f.weight.copy_(
        hyperclone_weight(old_model.norm_f.weight, scale_factor, 0, "out")
    )
    
    # Final Head (Case 2: Input only - Logic: d_model grows, vocab stays same)
    # Note: model ties weights, but if they weren't tied, we'd do this:
    if not isinstance(new_model.head, nn.Identity):  # Just in case logic changes
        new_model.head.weight.copy_(
            hyperclone_weight(old_model.head.weight, scale_factor, 0, "in")
        )

    # --- 3. Expand Existing Blocks (HyperCloning) ---
    n_old = len(old_model.blocks)
    
    for i in range(n_old):
        src = old_model.blocks[i]
        dst = new_model.blocks[i]
        
        # RMSNorms (Output expansion)
        dst.norm1.weight.copy_(hyperclone_weight(src.norm1.weight, scale_factor, 0, "out"))
        dst.norm2.weight.copy_(hyperclone_weight(src.norm2.weight, scale_factor, 0, "out"))
        
        # Attention (Q, K, V, O)
        # Q, K, V: d_model -> d_model (Both expand). 
        # By doubling n_heads, we effectively "tile" the heads.
        dst.attention.q_proj.weight.copy_(hyperclone_weight(src.attention.q_proj.weight, scale_factor, noise_std, "both"))
        dst.attention.k_proj.weight.copy_(hyperclone_weight(src.attention.k_proj.weight, scale_factor, noise_std, "both"))
        dst.attention.v_proj.weight.copy_(hyperclone_weight(src.attention.v_proj.weight, scale_factor, noise_std, "both"))
        dst.attention.o_proj.weight.copy_(hyperclone_weight(src.attention.o_proj.weight, scale_factor, noise_std, "both"))
        
        # MoE Layer
        # 1. Experts (SwiGLU)
        for e_idx in range(len(src.ffn.experts)):
            s_exp = src.ffn.experts[e_idx]
            d_exp = dst.ffn.experts[e_idx]
            
            # w1 (Gate) & w3 (Up): d_model -> d_ff (Both expand)
            d_exp.w1.weight.copy_(hyperclone_weight(s_exp.w1.weight, scale_factor, noise_std, "both"))
            d_exp.w3.weight.copy_(hyperclone_weight(s_exp.w3.weight, scale_factor, noise_std, "both"))
            
            # w2 (Down): d_ff -> d_model (Both expand)
            d_exp.w2.weight.copy_(hyperclone_weight(s_exp.w2.weight, scale_factor, noise_std, "both"))
            
        # 2. Router (Linear: d_model -> num_experts)
        # Input dim expands, output (num_experts) stays constant.
        # We must divide by scale because input x magnitude is larger.
        dst.ffn.router.weight.copy_(hyperclone_weight(src.ffn.router.weight, scale_factor, 0, "in"))
        # Bias doesn't change because output dim didn't change
        if src.ffn.router.bias is not None:
            dst.ffn.router.bias.copy_(src.ffn.router.bias)

    # --- 4. Expand Depth (Gstack) ---
    # Copy the last trained block to the new layers
    # This acts as an Identity function (or close to it)
    last_block_state = new_model.blocks[n_old - 1].state_dict()
    
    for i in range(n_old, config['n_layers']):
        new_model.blocks[i].load_state_dict(last_block_state)
        print(f"  > Initialized Layer {i} as Gstack copy of Layer {n_old-1}")

    print(f"✓ Growth Complete. New Config: {config['d_model']} dim, {config['n_layers']} layers, {config['n_heads']} heads.")
    return new_model
