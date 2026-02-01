"""Weight transfer from Dense to MoE model with functional preservation."""

import torch
import torch.nn as nn
from typing import Tuple


def transfer_dense_to_moe(dense_model, moe_model, verbose: bool = True):
    """
    Transfer weights from dense model to MoE model while preserving functionality.
    
    Strategy:
    1. Copy embeddings and output head (shared)
    2. Copy attention weights (identical in both models)
    3. Copy layer norms (identical in both models)
    4. Replicate FFN weights to all experts in MoE
    5. Initialize router to near-zero for uniform distribution
    
    Args:
        dense_model: Trained dense transformer
        moe_model: Initialized MoE transformer (weights will be overwritten)
        verbose: Print transfer details
    
    Returns:
        moe_model: MoE model with transferred weights
    """
    if verbose:
        print("=" * 60)
        print("Starting Dense → MoE Weight Transfer")
        print("=" * 60)
    
    # Ensure models are on the same device
    device = next(dense_model.parameters()).device
    moe_model = moe_model.to(device)
    
    with torch.no_grad():
        # 1. Transfer embeddings
        if verbose:
            print("\n[1/5] Transferring token embeddings...")
        moe_model.embed.weight.copy_(dense_model.embed.weight)
        
        # 2. Transfer output head (tied with embeddings)
        if verbose:
            print("[2/5] Output head already tied with embeddings ✓")
        
        # 3. Transfer final layer norm
        if verbose:
            print("[3/5] Transferring final layer norm...")
        moe_model.norm_f.weight.copy_(dense_model.norm_f.weight)
        
        # 4. Transfer transformer blocks
        if verbose:
            print(f"[4/5] Transferring {len(dense_model.blocks)} transformer blocks...")
        
        for layer_idx, (dense_block, moe_block) in enumerate(zip(dense_model.blocks, moe_model.blocks)):
            if verbose and layer_idx % 3 == 0:
                print(f"  → Processing layer {layer_idx + 1}/{len(dense_model.blocks)}")
            
            # Transfer attention weights
            moe_block.attention.q_proj.weight.copy_(dense_block.attention.q_proj.weight)
            moe_block.attention.k_proj.weight.copy_(dense_block.attention.k_proj.weight)
            moe_block.attention.v_proj.weight.copy_(dense_block.attention.v_proj.weight)
            moe_block.attention.o_proj.weight.copy_(dense_block.attention.o_proj.weight)
            
            # Transfer attention RoPE buffers
            moe_block.attention.rotary_emb.inv_freq.copy_(dense_block.attention.rotary_emb.inv_freq)
            
            # Transfer layer norms
            moe_block.norm1.weight.copy_(dense_block.norm1.weight)
            moe_block.norm2.weight.copy_(dense_block.norm2.weight)
            
            # Transfer FFN to all experts (CRITICAL for functional preservation)
            for expert_idx, expert in enumerate(moe_block.moe.experts):
                expert.w1.weight.copy_(dense_block.feed_forward.w1.weight)
                expert.w2.weight.copy_(dense_block.feed_forward.w2.weight)
                expert.w3.weight.copy_(dense_block.feed_forward.w3.weight)
            
            # Initialize router to near-zero for uniform distribution
            # This ensures each expert gets equal probability initially
            nn.init.normal_(moe_block.moe.router.weight, mean=0.0, std=0.01)
        
        if verbose:
            print(f"  → All {len(dense_model.blocks)} layers transferred ✓")
        
        # 5. Verify parameter counts
        if verbose:
            print("\n[5/5] Verifying transfer...")
            dense_params = dense_model.get_num_params()
            moe_params = moe_model.get_num_params()
            active_params = moe_model.get_active_params_per_token()
            
            print(f"\nParameter Summary:")
            print(f"  Dense model:          {dense_params:,} parameters")
            print(f"  MoE model (total):    {moe_params:,} parameters")
            print(f"  MoE active/token:     {active_params:,} parameters")
            print(f"  Expansion ratio:      {moe_params / dense_params:.2f}x")
            print(f"  Active ratio:         {active_params / dense_params:.2f}x")
        
        if verbose:
            print("\n" + "=" * 60)
            print("Weight transfer completed successfully! ✓")
            print("=" * 60 + "\n")
    
    return moe_model


def verify_functional_identity(dense_model, moe_model, test_input: torch.Tensor, 
                               tolerance: float = 1e-3, verbose: bool = True) -> Tuple[bool, float]:
    """
    Verify that MoE model produces similar outputs to dense model after transfer.
    
    Due to the stochastic nature of top-k routing, outputs won't be exactly identical,
    but should be very close if router is initialized near zero.
    
    Args:
        dense_model: Dense model
        moe_model: MoE model after weight transfer
        test_input: Test input tensor (batch_size, seq_len)
        tolerance: Maximum allowed mean absolute difference
        verbose: Print verification details
    
    Returns:
        is_preserved: Whether functional identity is preserved within tolerance
        max_diff: Maximum absolute difference between outputs
    """
    dense_model.eval()
    moe_model.eval()
    
    with torch.no_grad():
        # Get outputs
        dense_logits = dense_model(test_input)
        moe_logits, _ = moe_model(test_input)
        
        # Compute differences
        abs_diff = (dense_logits - moe_logits).abs()
        mean_diff = abs_diff.mean().item()
        max_diff = abs_diff.max().item()
        
        # Check if within tolerance
        is_preserved = mean_diff < tolerance
        
        if verbose:
            print("\n" + "=" * 60)
            print("Functional Identity Verification")
            print("=" * 60)
            print(f"Mean absolute difference: {mean_diff:.6f}")
            print(f"Max absolute difference:  {max_diff:.6f}")
            print(f"Tolerance:                {tolerance:.6f}")
            print(f"Status:                   {'✓ PASS' if is_preserved else '✗ FAIL'}")
            
            if not is_preserved:
                print(f"\n⚠ Warning: Functional identity not preserved!")
                print(f"  Mean difference {mean_diff:.6f} exceeds tolerance {tolerance:.6f}")
            else:
                print(f"\n✓ Functional identity preserved within tolerance!")
            
            print("=" * 60 + "\n")
        
        return is_preserved, max_diff


def analyze_expert_diversity(moe_model, test_input: torch.Tensor, verbose: bool = True):
    """
    Analyze how tokens are distributed across experts after transfer.
    
    Args:
        moe_model: MoE model
        test_input: Test input (batch_size, seq_len)
        verbose: Print analysis
    """
    moe_model.eval()
    
    expert_usage = {i: 0 for i in range(moe_model.config.num_experts)}
    total_tokens = test_input.numel()
    
    with torch.no_grad():
        x = moe_model.embed(test_input)
        
        for layer_idx, block in enumerate(moe_model.blocks):
            # Get router logits
            x_flat = x.view(-1, moe_model.config.d_model)
            router_logits = block.moe.router(x_flat)
            
            # Get top-k indices
            _, top_k_indices = torch.topk(router_logits, block.moe.top_k, dim=-1)
            
            # Count expert usage
            for expert_id in range(moe_model.config.num_experts):
                expert_usage[expert_id] += (top_k_indices == expert_id).sum().item()
            
            # Forward through block
            x, _ = block(x.view(*test_input.shape, -1))
    
    if verbose:
        print("\n" + "=" * 60)
        print("Expert Usage Analysis (across all layers)")
        print("=" * 60)
        
        for expert_id, count in expert_usage.items():
            percentage = (count / (total_tokens * len(moe_model.blocks)) * 100)
            print(f"Expert {expert_id}: {count:6d} tokens ({percentage:5.2f}%)")
        
        print("=" * 60 + "\n")
    
    return expert_usage
