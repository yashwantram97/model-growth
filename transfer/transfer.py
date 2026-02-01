"""Weight transfer for models (SmolLM-style architecture)."""

import torch
import torch.nn as nn


@torch.no_grad()
def transition_to_moe(dense_model, num_experts: int, top_k: int):
    """
    Upcycle a trained dense SLM into a sparse MoE SLM.
    Returns a NEW model; dense_model is unchanged (kept for verification).
    
    Works with SmolLM-style architecture (RMSNorm, RoPE, SwiGLU).
    
    Args:
        dense_model: Trained dense model
        num_experts: Number of experts per MoE layer
        top_k: Number of experts to activate per token
    
    Returns:
        moe_model: New MoE model with transferred weights
    """
    from models.model import SLM
    
    vocab_size = dense_model.embed.num_embeddings
    d_model = dense_model.embed.embedding_dim
    n_layers = len(dense_model.blocks)
    n_heads = dense_model.blocks[0].attention.n_heads
    d_ff = dense_model.blocks[0].ffn.w1.out_features
    
    # Get dropout and max_seq_len if available
    dropout = getattr(dense_model.blocks[0].attention, 'dropout', nn.Dropout(0.1)).p
    max_seq_len = getattr(dense_model.blocks[0].attention.rotary_emb, 'max_seq_len', 2048)

    print(f"\n  [transition_to_moe]")
    print(f"    Input:  Dense  (d_model={d_model}, layers={n_layers})")
    print(f"    Output: MoE    ({num_experts} experts/block, top-{top_k})")

    moe_model = SLM(vocab_size, d_model, n_layers, n_heads, d_ff,
                    use_moe=True, num_experts=num_experts, top_k=top_k,
                    dropout=dropout, max_seq_len=max_seq_len)

    # Shared layers
    moe_model.embed.load_state_dict(dense_model.embed.state_dict())
    moe_model.norm_f.load_state_dict(dense_model.norm_f.state_dict())

    # Per-block transfer
    for idx, (src, dst) in enumerate(
            zip(dense_model.blocks, moe_model.blocks)):

        # Transfer attention (custom MultiHeadAttention with RoPE)
        dst.attention.load_state_dict(src.attention.state_dict())
        dst.norm1.load_state_dict(src.norm1.state_dict())
        dst.norm2.load_state_dict(src.norm2.state_dict())

        # Clone FFN → every expert (SwiGLU with w1, w2, w3)
        ffn_state = src.ffn.state_dict()
        for expert in dst.ffn.experts:
            expert.load_state_dict(ffn_state)

        # Zero router
        nn.init.zeros_(dst.ffn.router.weight)
        nn.init.zeros_(dst.ffn.router.bias)

    print(f"    ✓ {n_layers} blocks converted  "
          f"({num_experts} experts each, router zeroed)")
    return moe_model


def verify_functional_equivalence(dense_model, moe_model, probe, device):
    """
    probe: (B, seq_len) — frozen token batch from TinyStories.

    Runs both models on the same probe, prints per-sequence diffs,
    asserts overall max diff < tolerance.
    
    Args:
        dense_model: Dense model
        moe_model: MoE model after weight transfer
        probe: Test input tensor
        device: Device to run on
    
    Returns:
        max_diff: Maximum absolute difference
    """
    dense_model.eval()
    moe_model.eval()

    with torch.no_grad():
        logits_dense = dense_model(probe)  # (B, S, V)
        logits_moe = moe_model(probe)      # (B, S, V)

    # Per-sequence max absolute difference (collapse seq & vocab dims)
    diff_per_seq = (logits_dense - logits_moe).abs().amax(dim=(1, 2))  # (B,)

    print("\n  [Verification] Per-sequence max |logit_dense − logit_moe|:")
    for i in range(diff_per_seq.shape[0]):
        status = "OK" if diff_per_seq[i].item() < 1e-4 else "FAIL"
        print(f"    Seq {i}: {diff_per_seq[i].item():.2e}  {status}")

    overall = diff_per_seq.max().item()
    print(f"\n  Overall max difference: {overall:.2e}")

    tol = 1e-4
    if overall < tol:
        print(f"  ✓ PASSED — functional equivalence confirmed (< {tol})")
    else:
        print(f"  ✗ FAILED — diff {overall:.2e} exceeds tolerance {tol}")

    assert overall < tol, \
        f"Functional equivalence broken! max_diff={overall}"
    return overall
