"""Simple Transformer Models (SmolLM-style architecture)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in SmolLM)."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMSNorm: x / sqrt(mean(x^2) + eps) * weight
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) as used in SmolLM."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies for half the dimensions
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len: int):
        """Apply rotary embeddings to input."""
        # x: (batch, seq_len, n_heads, head_dim)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)
        # Return cos and sin without concatenation
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    # x: (batch, seq_len, n_heads, head_dim)
    # cos, sin: (seq_len, head_dim//2)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
    # Reshape cos and sin for broadcasting: (1, seq_len, 1, head_dim//2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    
    # Rotate
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    return rotated


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with RoPE."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(out)


class FeedForward(nn.Module):
    """
    SwiGLU Feed-forward network (SmolLM-style).
    x → (SiLU(xW1) ⊙ xW3)W2 → out
    This entire module becomes ONE expert inside MoELayer after upcycling.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SwiGLU: (Swish(xW1) ⊙ xW3)W2
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer. Replaces a single FeedForward.

        input x  (B, S, d_model)
             │
             ▼
        ┌─────────┐
        │ Router  │  Linear(d_model → num_experts)
        │         │  logits → softmax → pick top_k → re-normalize
        └────┬────┘
             │
        ┌────┼────────────┐
        ▼    ▼            ▼
     Exp0  Exp1  …  Exp(N-1)     ← each is a FeedForward (SwiGLU)
        │    │            │
        └────┼────────────┘
             │  weighted sum of selected experts' outputs
             ▼
          output  (B, S, d_model)

    UPCYCLING GUARANTEE:
      All experts start as identical copies of the original FFN.
      Router is zeroed → softmax is uniform → any weighted combination
      of identical functions equals the original function.
      Therefore MoE(x) == FFN(x) at the moment of conversion.
    """
    def __init__(self, d_model: int, d_ff: int,
                 num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)]
        )
        self.router = nn.Linear(d_model, num_experts)
        self.top_k = top_k
        self.num_experts = num_experts

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*S, D)

        # ── 1. Routing ────────────────────────────────────────────────
        logits = self.router(x_flat)  # (B*S, N)
        probs = F.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)  # renorm

        # ── 2. Dispatch & combine ─────────────────────────────────────
        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (topk_idx == i).any(dim=-1)  # which tokens use expert i?
            if not mask.any():
                continue
            w = (topk_idx[mask] == i).float() * topk_w[mask]
            w = w.sum(dim=-1, keepdim=True)
            out[mask] += w * self.experts[i](x_flat[mask])

        return out.reshape(B, S, D)


class TransformerBlock(nn.Module):
    """
    Pre-norm block: x → RMSNorm → Attn → +residual → RMSNorm → FFN/MoE → +residual
    The use_moe flag swaps FFN ↔ MoE; everything else is shared.
    Uses SmolLM-style architecture with RMSNorm and RoPE attention.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 use_moe: bool = False,
                 num_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.norm2 = RMSNorm(d_model)

        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        else:
            self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class SLM(nn.Module):
    """
    Decoder-only transformer with SmolLM-style architecture.

        Token IDs → Embedding → [Block × L] → RMSNorm → LM Head → Logits

    Embedding and Head share one weight matrix (tied weights).
    Uses RMSNorm, RoPE attention, and SwiGLU feed-forward.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int,
                 use_moe: bool = False,
                 num_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff,
                             use_moe, num_experts, top_k,
                             dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tie weights
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch_size, seq_len) token IDs
            mask: Optional attention mask
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.embed(x)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm_f(x)
        return self.head(x)  # (B, S, vocab_size)
    
    def get_num_params(self):
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
