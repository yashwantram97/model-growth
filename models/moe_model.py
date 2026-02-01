"""Mixture of Experts (MoE) Transformer Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dense_model import RMSNorm, MultiHeadAttention, FeedForward


class MoELayer(nn.Module):
    """Sparse Mixture of Experts layer with top-k routing."""
    
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create expert networks (each is a standard FFN)
        self.experts = nn.ModuleList([
            FeedForward(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Linear(d_model, num_experts, bias=False)
        
        # Load balancing loss weight
        self.load_balance_loss_weight = 0.01
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
            aux_loss: Load balancing auxiliary loss
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch*seq, d_model)
        
        # Router logits
        router_logits = self.router(x_flat)  # (batch*seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)  # (batch*seq,)
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]  # (num_tokens, d_model)
                
                # Get expert weights for these tokens
                expert_weights = top_k_probs[expert_mask]  # (num_tokens, top_k)
                expert_idx = (top_k_indices[expert_mask] == i).float()  # (num_tokens, top_k)
                expert_weight = (expert_weights * expert_idx).sum(dim=-1, keepdim=True)  # (num_tokens, 1)
                
                # Apply expert
                expert_output = expert(expert_input)  # (num_tokens, d_model)
                
                # Weighted contribution
                output[expert_mask] += expert_weight * expert_output
        
        # Reshape output
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balancing loss
        aux_loss = self._compute_load_balance_loss(router_probs)
        
        return output, aux_loss
    
    def _compute_load_balance_loss(self, router_probs):
        """
        Compute load balancing auxiliary loss to encourage uniform expert usage.
        
        Args:
            router_probs: (batch*seq, num_experts)
        Returns:
            loss: scalar
        """
        # Average probability per expert
        expert_usage = router_probs.mean(dim=0)  # (num_experts,)
        
        # Encourage uniform distribution (1/num_experts for each expert)
        target_usage = 1.0 / self.num_experts
        
        # Mean squared error from uniform distribution
        loss = ((expert_usage - target_usage) ** 2).sum()
        
        return self.load_balance_loss_weight * loss


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE feed-forward layer."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int = 8, 
                 top_k: int = 2, dropout: float = 0.1, max_seq_len: int = 2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k, dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x), mask)
        
        # MoE layer
        moe_out, aux_loss = self.moe(self.norm2(x))
        x = x + moe_out
        
        return x, aux_loss


class MoETransformer(nn.Module):
    """MoE Transformer Language Model."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        
        # MoE Transformer blocks
        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                config.d_model,
                config.n_heads,
                config.d_ff,
                config.num_experts,
                config.top_k,
                config.dropout,
                config.max_seq_len
            )
            for _ in range(config.n_layers)
        ])
        
        # Final norm and output projection
        self.norm_f = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie embeddings
        self.head.weight = self.embed.weight
        
    def forward(self, input_ids, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            mask: Optional attention mask
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            total_aux_loss: Load balancing loss across all layers
        """
        x = self.embed(input_ids)
        
        # Create causal mask if not provided
        if mask is None:
            seq_len = input_ids.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Accumulate auxiliary losses
        total_aux_loss = 0.0
        
        # Pass through transformer blocks
        for block in self.blocks:
            x, aux_loss = block(x, mask)
            total_aux_loss += aux_loss
        
        # Final norm and projection
        x = self.norm_f(x)
        logits = self.head(x)
        
        return logits, total_aux_loss
    
    def get_num_params(self, only_trainable=False):
        """Get number of parameters."""
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_active_params_per_token(self):
        """Calculate active parameters per token (considering top-k routing)."""
        # Attention parameters (always active)
        attn_params = sum(p.numel() for block in self.blocks 
                         for p in block.attention.parameters())
        
        # MoE parameters (only top_k experts active)
        expert_params = sum(p.numel() for expert in self.blocks[0].moe.experts[:1])
        active_moe_params = expert_params * self.config.top_k * self.config.n_layers
        
        # Embeddings and other parameters
        other_params = (self.embed.weight.numel() + 
                       sum(p.numel() for block in self.blocks for p in block.norm1.parameters()) +
                       sum(p.numel() for block in self.blocks for p in block.norm2.parameters()) +
                       self.norm_f.weight.numel())
        
        return attn_params + active_moe_params + other_params
