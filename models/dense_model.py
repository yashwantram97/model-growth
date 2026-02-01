"""Dense Transformer Model (SmolLM-style architecture)."""

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
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
    def forward(self, x, seq_len: int):
        """Apply rotary embeddings to input."""
        # x: (batch, seq_len, n_heads, head_dim)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_emb(x, cos, sin):
    """Apply rotary embeddings."""
    # x: (batch, seq_len, n_heads, head_dim)
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    
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
    """Feed-forward network with SwiGLU activation (SmolLM-style)."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # Gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=False)  # Down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # SwiGLU: (Swish(xW1) ⊙ xW3)W2
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Single transformer block."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.feed_forward(self.norm2(x))
        return x


class DenseTransformer(nn.Module):
    """Dense Transformer Language Model (SmolLM-style)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model, 
                config.n_heads, 
                config.d_ff, 
                config.dropout,
                config.max_seq_len
            )
            for _ in range(config.n_layers)
        ])
        
        # Final norm and output projection
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings (weight sharing)
        self.head.weight = self.embed.weight
        
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
    
    def forward(self, input_ids, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            mask: Optional attention mask
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.embed(input_ids)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = input_ids.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm and projection
        x = self.norm_f(x)
        logits = self.head(x)
        
        return logits
    
    def get_num_params(self):
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters())
