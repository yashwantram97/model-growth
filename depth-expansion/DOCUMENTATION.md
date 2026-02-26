# Depth-Expansion Codebase — Complete Documentation

> A research codebase for training language models with **adaptive compute depth**,
> **linear attention**, **sparse mixture-of-experts**, and **reversible integration**.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Conceptual Foundations](#2-conceptual-foundations)
3. [File: `models/reversible_ops_midpoint.py`](#3-models-reversible_ops_midpointpy)
4. [File: `models/recurrence_model_3b.py`](#4-models-recurrence_model_3bpy)
5. [File: `new-fourier-test.py` — SmolLM Training Loop](#5-new-fourier-testpy--smollm-training-loop)
6. [File: `new-fourier-test-new.py` — Model3B Training Loop](#6-new-fourier-test-newpy--model3b-training-loop)
7. [How the Files Connect](#7-how-the-files-connect)
8. [Key Hyperparameter Reference](#8-key-hyperparameter-reference)
9. [Production Blockers & TODOs](#9-production-blockers--todos)

---

## 1. System Overview

This codebase trains **decoder-only language models** with two experimental ideas on top of standard transformer training:

| Idea | What it does | Why |
|---|---|---|
| **Adaptive Depth (Pondering)** | Each transformer layer decides *how many passes* to run on each token (1–4) | Lets easy tokens be cheap, hard tokens get more compute |
| **Reversible Midpoint Integration** | Layers are stacked as a physics-inspired ODE integrator | Memory-efficient backprop — no need to store all activations |

Two model sizes exist:

- **SmolLM** (`new-fourier-test.py`): 86M-param model with standard MHA attention and Fourier embeddings. Trained on Mac M1.
- **Model3B** (`new-fourier-test-new.py`): ~450M-param (local config) / ~3.9B (full) model with hybrid DeltaNet + Sparse Attention and Kronecker embeddings. Trained on Mac M4 Pro.

---

## 2. Conceptual Foundations

### 2.1 Adaptive Depth / Pondering

Traditional transformers always run every token through every layer exactly once. **Pondering** (also called "adaptive computation") lets the model decide to run a token through additional processing passes.

**How it works:**
```
Pass 1: run layer → compute "should I keep going?" → probability p₁
Pass 2: run layer again → compute p₂
...
Stop when cumulative halt probability falls below a threshold
```

The **survival weight** `s` represents how much the token is still "alive":
- `s₀ = 1.0` (full weight at start)
- `s₁ = p₁` (after pass 1, carry over)
- `s₂ = p₁ × p₂` (multiply probabilities)

Each pass's contribution to the hidden state is gated by `s`, so tokens that halt early contribute less in later passes.

This is trained with a **ponder loss** (`λ_p × avg_depth`) that taxes compute, plus a **coupon reward** that refunds the tax when extra passes actually improve predictions.

### 2.2 Reversible Midpoint / Leapfrog Integration

Imagine a ball bouncing — its position and velocity evolve according to physics equations. The **leapfrog integrator** is a standard numerical method for this:

```
position_next = position_prev + 2h × velocity(position_current)
```

In this codebase, transformer layers are treated as computing a "force" (residual update `delta`), and hidden states are "positions" in a high-dimensional space:

```
p_next = a × p_prev + (1-a) × p_cur + 2h × f(p_cur)
```

**Why is this useful?**
- **Reversibility**: You can reconstruct earlier hidden states from later ones, which means you don't need to save all intermediate activations for backprop → less memory.
- **Stability**: Leapfrog conserves energy in physical systems, which translates to stable gradient flow in neural networks.

### 2.3 Linear Attention (GatedDeltaNet)

Standard attention is O(T²) — it compares every token to every other token, which becomes prohibitive at long sequences (e.g. 256k tokens). **Linear attention** replaces this with a **recurrent state** that is O(N):

```
State S_t = α_t × S_{t-1} × (I - β_t × k_t × k_t^T) + β_t × v_t × k_t^T
Output o_t = q_t × S_t
```

Think of it as a "memory matrix" that is updated at each token position. New tokens can read from and write to this matrix.

- **α (alpha)**: Decay — controls how fast old information fades. α close to 1 = long memory.
- **β (beta)**: Writing strength — controls how strongly a token updates the memory.

### 2.4 Mixture of Experts (MoE)

Instead of one big FFN layer, use many smaller expert networks and route each token to only 1–2 of them:

```
Token → Router → Select top-2 experts → Run those experts → Weighted sum
```

**Null Experts**: Half the expert slots are "null" (do nothing). A token routed to a null expert simply passes through unchanged. This enforces sparsity — on average only 1 real expert fires per token.

**Why this helps**: Total parameter count is large (more experts = more capacity), but compute per token is small (only 2 experts run).

### 2.5 Kronecker Embeddings

Standard token embeddings are lookup tables: `token_id → vector`. Kronecker embeddings instead *compute* the embedding from the token's bytes:

```
token "hello" → UTF-8 bytes [104, 101, 108, 108, 111]
            → Kronecker product of byte embeddings and position embeddings
            → 8192-dim vector
```

**Why**: The embedding captures *structure* in the token (what characters it contains, where they appear). Two tokens with similar spelling get similar embeddings by construction, not by learning.

---

## 3. `models/reversible_ops_midpoint.py`

This file implements the **reversible midpoint integrator** — the memory-efficient layer stack used when ponder mode is off.

### 3.1 `_ForceWrapper` (lines 7–18)

```python
class _ForceWrapper(nn.Module):
    def forward(self, x, attention_mask=None):
        return self.layer.force(x, attention_mask=attention_mask)
```

**What it does**: A thin wrapper that redirects `forward()` calls to `force()`. This is needed because `functional_call` (used in the custom autograd function) calls `forward`, but we want it to call `force` (which returns `delta, aux` instead of a full forward pass).

**Why it's critical**: Without this wrapper, `functional_call` would call the layer's `forward()` and get the wrong output format.

---

### 3.2 `MidpointFunction` (lines 21–145)

This is a **custom PyTorch autograd function** — it manually defines both the forward pass and the backward pass (gradient computation).

#### Forward pass (lines 23–53)

```python
p_next = a × p_prev + (1-a) × p_cur + 2h × f(p_cur)
```

Where:
- `p_prev`, `p_cur` = two consecutive hidden states (like leapfrog needs two positions)
- `f(p_cur)` = the layer's `force()` output = the residual delta
- `a` = stabilizing blend coefficient (0.85–0.98 typically)
- `2h` = twice the step size

**What gets saved for backward**: `p_prev`, `p_cur`, and all trainable parameters of the layer. Buffers (non-trainable) are not saved — they're retrieved from the live module at backward time.

#### Backward pass (lines 55–145)

Manual gradient computation using the chain rule:

```
∂L/∂p_prev = ∂L/∂p_next × a
∂L/∂p_cur  = ∂L/∂p_next × (1-a) + ∂L/∂delta × ∂delta/∂p_cur
∂L/∂params = ∂L/∂delta × ∂delta/∂params
```

The backward pass **recomputes** `f(p_cur)` (the forward pass through the force function) to get gradients — this is the "reversible" trick. Instead of storing `delta` from the forward pass, we recompute it, saving memory.

**Key detail**: `torch.autograd.grad` is used instead of `.backward()` because we need gradients with respect to specific inputs, not all parameters. The `allow_unused=True` flag handles layers where some parameters don't contribute to `delta`.

**aux handling**: If the auxiliary loss (e.g., MoE router loss) has a gradient (`aux.grad_fn is not None`), it's included in the gradient computation. Dense FFN returns a constant `0.0` with no grad_fn, so it's excluded.

---

### 3.3 `MidpointBlock` (lines 148–176)

A wrapper module that holds one layer and applies the midpoint update formula.

```python
class MidpointBlock(nn.Module):
    def forward(self, p_prev, p_cur, attention_mask=None):
        # Collects layer parameters and calls MidpointFunction.apply(...)
```

**Key initialization**:
- `self.wrapper` = a `_ForceWrapper` around the block
- `self.param_keys` / `self.buffer_keys` = stable lists of parameter/buffer names (cached at init so the order doesn't change during training)
- `self.two_h` = 2 × step_size (used in the recurrence formula)

**Why collect param_values at forward time?** Because `functional_call` requires tensor references, not just names. By passing tensors as `*flat_tensors` to `MidpointFunction.apply()`, PyTorch's autograd can track gradients through them.

---

### 3.4 `ReversibleMidpointStack` (lines 179–257)

The main entry point — stacks all layers and runs the leapfrog recurrence.

#### Initialization

```python
self.bootstrap_layer = blocks[0]      # First layer (special treatment)
self.mid_layers = [MidpointBlock(b, ...) for b in blocks[1:]]  # Rest
```

The first layer needs special "bootstrap" handling because leapfrog requires **two** initial states but we only have one at the start.

#### Forward pass

**Step 1 — Bootstrap**: Create two initial states `(p_prev, p_cur)` from the single input `x`.

Two modes:
- `"no_kick"`: `p_cur = p_prev = x` (both start identical; simple and aligned with baseline)
- `"euler"`: `p_cur = x + 0.5h × f(x)` (half-step Euler to "kick" the system into motion)

**Step 2 — Midpoint recurrence**: For each remaining layer:
```python
p_next, aux = layer(p_prev, p_cur)
p_prev, p_cur = p_cur, p_next  # Shift window
```

**Output**: `(p_cur, total_aux)` — the final hidden state and sum of auxiliary losses.

**gradient_checkpoint**: The bootstrap layer uses `torch.utils.checkpoint.checkpoint` for gradient checkpointing — this trades compute for memory by recomputing activations during backward.

---

## 4. `models/recurrence_model_3b.py`

The main model file. Contains all architectural components for Model3B.

### 4.1 Embedding: `KroneckerConfig` + `KroneckerEmbeddings` (lines 196–395)

#### `KroneckerConfig`
```python
@dataclass
class KroneckerConfig:
    CHAR_DIM: int = 256   # Number of possible byte values (0–255)
    POS_DIM: int = 32     # Max token length in bytes
    D: int = 8192         # Total dimension = 256 × 32
```

#### `KroneckerEmbeddings.encode_word(word: str) → np.ndarray`

**Step-by-step**:
1. Convert string to UTF-8 bytes: `"hello" → [104, 101, 108, 108, 111]`
2. Truncate if > 32 bytes (UTF-8 safe — won't split a multi-byte character)
3. Build a 256×32 matrix `M` where `M[byte_value, position] = 1.0`
4. Length-normalize: `M *= 1/√L`
5. Flatten to a 8192-dim vector

**Example for "hi"** (bytes [104, 105]):
```
M[104, 0] = 1/√2     # 'h' at position 0
M[105, 1] = 1/√2     # 'i' at position 1
→ flatten → 8192-dim vector
```

#### `KroneckerEmbeddings.decode_word(pf_vec) → str`

Invert the encoding:
1. Reshape 8192-dim vector to 256×32 matrix
2. Find positions (columns) with non-zero norm = "active" positions
3. At each active position, take `argmax` over 256 bytes
4. Convert byte sequence back to UTF-8 string

This is **lossless** for tokens ≤ 32 bytes.

---

### 4.2 Model Configurations

#### `ModelConfig` — Full 3B Model
| Parameter | Value | Meaning |
|---|---|---|
| `vocab_size` | 131,072 (2^17) | TSAI tokenizer |
| `hidden_size` | 4096 | Main embedding dimension |
| `num_layers` | 8 | Total transformer layers |
| `num_deltanet_layers` | 6 | 75% linear attention |
| `num_gsa_layers` | 2 | 25% sparse attention |
| `delta_v_heads` | 32 | DeltaNet heads |
| `delta_head_dim` | 128 | Dimension per head |
| `num_real_experts` | 20 | Active MoE experts |
| `num_null_experts` | 20 | Null (skip) experts |
| `top_k` | 2 | Experts selected per token |
| `max_seq_len` | 262,144 | 256k context |

**Total params: ~3.9B total, ~1.74B active**

#### `LocalModelConfig` — Mac M4 Pro (24 GB)
| Parameter | Value | Why smaller |
|---|---|---|
| `hidden_size` | 512 | 8× smaller |
| `delta_v_heads` | 16 | Halved |
| `delta_head_dim` | 32 | 4× smaller |
| `num_real_experts` | 4 | 5× fewer |
| `n_streams` | 1 | Eliminates stream factor |
| `enable_mtp` | False | Saves memory |

**Memory math**: `H × head_dim²` is the expensive term in DeltaNet. Old: `8×64²=32768`. New: `16×32²=16384`. This 2× reduction is what allows training on 24 GB RAM.

---

### 4.3 `PureHybridEmbeddingTorch` (lines 529–576)

The PyTorch module that serves Kronecker embeddings at runtime.

```python
def forward(self, token_ids):  # token_ids: [B, T]
    PF = self.PF_table[token_ids]  # [B, T, 8192] — lookup from precomputed table
    # Normalize: zero-mean, unit std per token
    PF = (PF - PF.mean(dim=-1, keepdim=True)) / (PF.std(dim=-1, keepdim=True) + 1e-6)
    return PF
```

**Key design choice**: `PF_table` is registered as `persistent=False`. This means it's NOT saved in checkpoints — it's recomputed from the vocab list when loading. This saves ~2 GB per checkpoint.

---

### 4.4 `RMSNorm` (lines 651–684)

Root Mean Square Layer Normalization — a simpler, faster alternative to LayerNorm.

```
RMSNorm(x) = x / RMS(x) × weight
where RMS(x) = √(mean(x²))
```

**Why RMS instead of LayerNorm?** LayerNorm subtracts mean then divides by std. RMSNorm skips the mean subtraction — faster and empirically just as good.

**Critical detail (FIX #43)**: All normalization math is done in `float32` even during `bfloat16` training. This prevents rare NaN spikes at very long sequences where `bfloat16` loses precision.

```python
x_f = x.float()           # Cast to float32
norm = x_f.pow(2).mean()  # All math in float32
return (x_f / sqrt(norm + eps) * weight).to(in_dtype)  # Cast back at end
```

---

### 4.5 `RotaryEmbedding` (lines 687–750)

Rotary Position Embeddings (RoPE) — encodes position information by *rotating* query and key vectors.

**Standard formula**: `inv_freq[i] = 1 / (base^(2i/dim))` for i = 0..dim/2-1

**Application**:
```python
x1, x2 = x[..., ::2], x[..., 1::2]  # Split even/odd dimensions
output = cat(x1*cos - x2*sin, x1*sin + x2*cos)
```

This rotation has the property that the dot product of Q and K only depends on their *relative* positions, which is exactly what attention needs.

**Memory optimization**: cos/sin tables are computed on-the-fly rather than cached. Caching would cost: `262,144 positions × 128 dims × 2 (cos/sin) × 4 bytes × 20 layers ≈ 5.4 GB`. On-the-fly is ~5–10% slower but saves 5.4 GB of VRAM.

---

### 4.6 `GatedDeltaNet` (lines 801–1070)

The main workhorse — 75% of layers use this.

#### Architecture

```
Input x [B, T, D]
   ↓
Q, K, V projections
   ↓
Short convolution (kernel_size=4)  ← local context
   ↓
Reshape to heads [B, T, H, head_dim]
   ↓
RoPE (position encoding)
   ↓
L2 normalize Q, K (not softmax!)
   ↓
Compute α (decay) and β (write strength)
   ↓
DeltaNet recurrence (O(N))
   ↓
Output gate (Swish × RMSNorm)
   ↓
Output projection
```

#### The DeltaNet Recurrence

State update formula:
```
S_t = α_t × S_{t-1} × (I - β_t × k_t × k_t^T) + β_t × v_t × k_t^T
o_t = q_t × S_t + D × (q_t · k_t) × v_t
```

In plain English:
1. **Forget**: Decay old state by α, also "erase" the direction of the new key (orthogonal projection)
2. **Write**: Add the new value-key outer product, scaled by β
3. **Read**: Dot the query against the state to get output

**The orthogonal projection** `(I - β_t × k_t × k_t^T)` is key: it removes the current key's direction from the state before writing the new value. This prevents interference between similar tokens.

#### α (decay) computation
```python
alpha = exp(-exp(A_log) × softplus(gk + dt_bias))
```
This ensures α ∈ (0, 1) — the exponential maps to full range, while `exp(A_log)` keeps A positive.

#### Short Convolutions
```python
class ShortConvolution(nn.Module):
    # Depthwise conv1d with kernel_size=4 and causal padding
```
Applied to Q, K, V before the recurrence. Purpose: captures local context (last 4 tokens) which linear attention struggles with. The causal padding ensures no future tokens leak in.

#### Python loop vs Triton kernel
The recurrence is currently a Python `for t in range(T):` loop. At 256k tokens, this launches 256,000 sequential GPU kernels — catastrophically slow. A Triton kernel would fuse all iterations into one launch, achieving 500–2000× speedup.

---

### 4.7 `GatedSparseAttention` (lines 1077–1301)

Used for 25% of layers (layers 3 and 7). Full-quality attention but with sparse selection.

#### Lightning Indexer

Instead of attending to all T keys, the indexer selects the top-k most relevant keys:

1. Compute low-dim query/key projections `q_I, k_I` (dim=32 instead of 64+)
2. Compute importance scores: `match_logits = q_I @ k_I^T` → [B, indexer_heads, T, T]
3. Apply causal mask
4. Adaptively determine k based on importance variance:
   - High variance → more keys needed → larger k
   - Low variance → most keys similar → smaller k
5. Keep only top-k indices per query position

**Adaptive k formula**:
```python
k_t = k_base × variance(importance) / variance_ema
k_t = clamp(k_t, k_min, k_max)
```

This means: if the importance scores are very spread out (high variance), the model needs more tokens to make a decision, so k is larger.

#### Attention Sinks

The first 4 tokens are always included in the sparse set (set to `+inf` importance). Empirically, transformers attend to early tokens regardless of content — these are "sink" tokens that absorb excess attention.

#### Dual Gating

Two gates:
- `g_v = sigmoid(W_gv(x))`: Applied to values before attention
- `g_o = sigmoid(W_go(x))`: Applied to output after attention

This gives the model two points of control over how much information flows through.

**Current limitation**: The O(T²) `match_logits` tensor still gets materialized during selection. At 256k tokens: `B × 16 heads × 256k × 256k × 4 bytes ≈ 1.1 TB`. Needs Triton sparse kernels to fix.

---

### 4.8 `MoEGate` + `MoEFFN` (lines 1308–1510)

#### Gate (`MoEGate`)

```python
real_logits = gate_linear(x)        # [B, T, num_experts]
null_logits = null_logit.expand(...)  # [B, T, num_null_copies]
logits = cat([real_logits, null_logits])
topk_idx, topk_weight = topk(softmax(logits), k=2)
is_null = topk_idx >= num_experts   # True if selected a null expert
```

**Null expert mechanism**: Half the "slots" are null experts with a shared learnable logit. If a token selects a null expert, that slot contributes zero to the output. The gate normalizes weights over only the real expert selections.

**Auxiliary losses**:
- `L_bal`: Load balancing — penalizes expert collapse (one expert getting all tokens)
- `L_z`: Z-loss — penalizes large logits for numerical stability
- `L_null`: Null rate regularizer — keeps null selection rate near target ρ=0.5

Total: `aux_loss = 0.02 × L_bal + 0.001 × L_z + 0.01 × L_null`

#### FFN (`MoEFFN`)

```python
# Shared expert (always runs)
shared_out = SwiGLU(x)  # shared_gate * shared_up, then shared_down

# Routed experts
topk_idx, topk_weight, is_null, aux = gate(x)
# Sort tokens by expert assignment (for batched matmul)
# Run each expert on its assigned tokens
# Scatter results back to original positions
y = shared_out + routed_out
```

**SwiGLU**: `SiLU(gate_proj(x)) × up_proj(x)` then `down_proj()`. This is the gated FFN used in LLaMA/DeepSeek.

---

### 4.9 Multi-Head Composition (mHC) — `MHCSublayer` (lines 1608–1638)

Each layer operates on multiple "streams" simultaneously. mHC routes information between these streams adaptively.

```
Input: x_stream [B, T, n_streams, D]

1. Compute routing coefficients (H_pre, H_post, H_res) from all streams combined
2. Pre-mix: x_in = sum(x_stream × H_pre)  → [B, T, D]
3. Run sublayer (attention or FFN): y = sublayer(x_in)
4. Post-expand: y_stream = y × H_post    → [B, T, n_streams, D]
5. Residual: x_res = H_res ⊗ x_stream   (Sinkhorn-normalized routing matrix)
6. Output: x_res + y_stream
```

**Sinkhorn-Knopp normalization** (`sinkhorn_knopp`): Makes the routing matrix doubly stochastic (rows and columns all sum to 1). This ensures information is neither created nor destroyed in the routing step — it's purely redistributed.

With `n_streams=1` (LocalModelConfig), mHC becomes a simple residual connection — no stream routing overhead.

---

### 4.10 `LightningDecoderLayer` (lines 1645–1875)

The complete decoder layer combining attention + FFN + mHC + optional pondering.

#### `force(x, attention_mask)` method

```python
def force(self, x, attention_mask=None):
    h, aux1 = self.attn_block(x, ...)
    out, aux2 = self.mlp_block(h, ...)
    delta = out - x   # Just the residual!
    return delta, aux
```

Returns only the **delta** (change), not the full output. This is what the reversible midpoint integrator calls.

#### `_ponder_forward` — Adaptive Depth Loop

```python
s = ones(B, T)          # Survival weight per token
for k in range(1, Kmax+1):
    delta, aux = self.force(x_stream)          # Compute update
    f_k = compute_features(delta, x_stream)    # [B, T, 2] features
    p_k = continue_head(cat([f_k, layer_emb])) # [B, T] continue prob

    gated_delta = s.unsqueeze(-1,-1) * alpha[k-1] * delta
    x_stream = x_stream + gated_delta          # Apply weighted update

    s = s * p_k                                # Decay survival
    if (s >= halt_thr).mean() <= 0.02:         # 98% tokens halted
        break
```

**`alpha` buffer**: `[1.0, 0.7, 0.5, 0.35]` — scales down the delta at each subsequent pass. This prevents the later passes from dominating and ensures the first pass is most impactful.

**2% early break**: Rather than waiting for 100% of tokens to halt, stops when only 2% are still alive. One stubborn token shouldn't force the whole batch through extra passes.

**Features for ContinueHead**:
```python
delta_rel = |dh| / |h|  # How large is the update relative to the state?
cos_dir = cosine_sim(dh, dh_prev)  # Are consecutive updates in the same direction?
```
These two features + 16 layer embedding dims = 18-dim input to ContinueHead.

---

### 4.11 `MTPTransformerBlock` — Multi-Token Prediction (lines 1882–1968)

Predicts token at position `t+2` (not just `t+1`) by combining:
- `h_t`: hidden state after processing token t
- `emb_{t+1}`: embedding of token t+1

```python
x = fusion_proj(cat([h_t, emb_{t+1}], dim=-1))  # Fuse 2D → D
# Run GSA + MoE blocks
logits_mtp = lm_head(norm(x))  # Predict token t+2
```

**Why GSA for MTP?** The MTP block runs once per step (not 8× like backbone layers), so the O(T²) cost of GSA is acceptable. And GSA provides better gradient signal than linear attention for this single-pass prediction task.

The MTP loss weight is 0.3 — it's an auxiliary teacher that improves gradient quality without dominating the main NTP loss.

---

### 4.12 `Model3B.forward()` (lines 2169–2268)

The complete forward pass:

```
1. Embed: Kronecker lookup + project to hidden_size + tok_embed residual
2. Expand to streams: x_stream [B, T, n_streams, D], only stream 0 filled
3. Cache RoPE cos/sin for all layers
4. Memory stream injection (if prev_memory_stream provided)
5. Layer pass:
   - use_ponder=True: direct loop, collect per-layer stats
   - use_ponder=False: reversible midpoint stack (memory-efficient)
6. Extract memory stream: last token's stream 3 vector
7. Collapse streams: mean over n_streams
8. Apply norm + lm_head → logits_ntp
9. MTP block (if enabled): embed next tokens, predict t+2 → logits_mtp
10. Clear RoPE cache
11. Return: logits_ntp, logits_mtp, aux_loss, memory_stream, layer_stats
```

**Memory stream recurrence**: The last token's stream 3 is extracted and passed to the next chunk. This enables processing documents longer than max_seq_len by chunking — the memory stream carries context across chunk boundaries.

---

## 5. `new-fourier-test.py` — SmolLM Training Loop

Training script for the **SmolLM** model (86M params) with Fourier embeddings and adaptive depth.

### 5.1 Architecture: `SmolLM`

```python
embed: HybridEmbeddingTorch(bpe_vocab, pf_codec, K=1536)  # Fourier embedding
pf_proj: Linear(2048 → 768)                                # Project to hidden_size
embed_ln: RMSNorm(768)
lambda_e: Parameter(768)                                    # Baton injection weights
layers: [AdaptiveDecoderLayer × 8]
norm: RMSNorm(768)
lm_head: Linear(768 → 50272)
```

**Baton Injection**: A form of sequential position embedding where each token's embedding is influenced by the previous token's embedding:
```python
sigmoid_lambda = sigmoid(lambda_e)        # [D], scales 0–1
x[:, 1:] += sigmoid_lambda × x[:, :-1]  # Token t gets info from t-1
```
This creates a learned "carry-over" effect between adjacent tokens.

### 5.2 `AdaptiveDecoderLayer`

The SmolLM-specific adaptive layer. Uses standard Multi-Head Latent Attention (MLA) and MoE FFN.

**Ponder loop** (runs up to `Kmax=4` times):
1. Compute attention weights explicitly (to extract entropy features)
2. Compute `compute_adaptive_features()` — 5 features including attention entropy, head agreement, relative update magnitude
3. Concatenate features with 16-dim `layer_emb` → 21-dim input to `ContinueHead`
4. Gate update by survival weight `s`

**Key difference from Model3B**: SmolLM uses 5 richer features including attention entropy, while Model3B uses 2 simpler features (since DeltaNet doesn't expose attention weights).

### 5.3 Data Pipeline: `SYNTHStream`

```python
class SYNTHStream(IterableDataset):
    # Loads PleIAs/SYNTH dataset from local disk
    # Instant resume: skips start_step × batch_size examples
    # Packs examples into fixed-length blocks (seq_len=512)
    # Yields: {"ids": tensor, "targets": tensor}
```

**Instant resume**: Rather than iterating through examples to skip them (slow), uses Arrow's `select(range(skip_n, len))` to immediately jump to the right position.

**Buffer overflow prevention**: Keeps at most `4 × seq_len` tokens in the buffer to prevent unbounded memory growth.

### 5.4 `SYNTHPromptSampler`

Used for generation during training (every 500 steps). Samples English-language queries from the SYNTH dataset, formatted as chat messages:
```
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
```

### 5.5 `sample_generate_single_fast`

Autoregressive generation with a pre-allocated buffer (no `torch.cat` per step):

```python
buf = empty(1, max_seq_len)      # Pre-allocate full buffer
buf[:, :prompt_len] = prompt     # Copy prompt in
for step in range(max_new_tokens):
    logits = model(buf[:, :cur_len])
    next_id = sample(logits[:, -1, :])  # Top-p sampling
    buf[0, cur_len] = next_id
    cur_len += 1
```

**Top-p (nucleus) sampling**:
1. Sort tokens by probability (descending)
2. Find the smallest set whose cumulative probability ≥ p
3. Zero out tokens outside this set
4. Sample from renormalized distribution

### 5.6 Training Loop Variables

| Variable | Value | Purpose |
|---|---|---|
| `D=768, L=8, HEADS=12` | — | Model dims |
| `TARGET_K=1.30` | depth target | Dual-ascent setpoint |
| `DUAL_LR=5e-5` | — | Lambda controller learning rate |
| `LAMBDA_MIN=1e-4` | — | Floor on compute price |
| `LAMBDA_MAX=5e-2` | — | Ceiling on compute price |
| `AUDIT_PROB=0.05` | 5% | Fraction of steps forcing k=2 |
| `REWARD_ETA=2.0` | — | Coupon multiplier for ΔCE |
| `CREDIT_BETA=0.995` | — | EMA decay for credit baseline |

### 5.7 The Dual-Ascent Controller

The compute price `λ_p` is updated each step like a PI controller:

```python
policy_avg_k = average expected depth across all layers
error = policy_avg_k - TARGET_K
lambda_p_state += DUAL_LR × error
lambda_p_state = clamp(lambda_p_state, LAMBDA_MIN, LAMBDA_MAX)
```

If the model is computing more than `TARGET_K` layers on average → raise `λ_p` → ponder loss increases → model learns to halt earlier.

If the model is computing less than `TARGET_K` → lower `λ_p` → compute becomes "cheaper" → model learns to use more passes.

**Deadband mode**: Once `λ_p` has been high (> 50% of range) for 100 consecutive steps, switch to an asymmetric deadband:
- Only raise `λ_p` if `avg_k > TARGET_K × 1.05` (5% above target)
- Only lower `λ_p` if `avg_k < TARGET_K × 0.98` (2% below target)
- Inside the band: freeze `λ_p`

This prevents whipsaw oscillations once the controller has converged.

### 5.8 Audit Forcing

5% of steps are "audit" steps where `force_2=True` (exactly 2 passes, regardless of what the gate says). These measure:
- What would happen if we always used 2 passes?
- The `ΔCE` (change in cross-entropy) tells us whether pass 2 actually helped.

**Critical**: Audit steps do NOT update the dual-ascent controller (they're counterfactual — not what the policy actually chose).

### 5.9 The Coupon Reward (Economic Fix)

```python
# After forward pass:
delta_ce = (CE_loss_pass1 - CE_loss_pass2).clamp(min=0)  # How much did pass 2 help?
ponder_loss = relu(raw_ponder - REWARD_ETA × delta_ce)   # Subtract the "coupon"
```

If pass 2 reduced the cross-entropy by 0.01, and `REWARD_ETA=2.0`, the coupon is 0.02. Any ponder loss below 0.02 is fully canceled.

This creates an economic incentive: the model only pays the ponder tax when it can't justify the compute through improved predictions.

### 5.10 Checkpoint System

```python
save_checkpoint():
    # Saves model_state_dict + optimizer_state_dict + step + loss + lambda_p_state
    # Saves both timestamped file AND "latest.pt" (atomic write via tmp file in Model3B)

load_checkpoint():
    # Auto-discovers highest step number checkpoint (not just "latest")
    # Uses strict=False to handle architecture changes
    # Restores lambda_p_state for continuity of the compute controller
```

**Auto-resume**: On startup, if a checkpoint exists, automatically resumes without needing `--resume` flag. Prevents accidental data loss.

---

## 6. `new-fourier-test-new.py` — Model3B Training Loop

Updated training script for **Model3B** with Kronecker embeddings and TSAI 131K tokenizer. Largely the same structure as the SmolLM loop with these key differences:

### 6.1 Model and Tokenizer Setup

```python
# TSAI 131K tokenizer (local)
tokenizer = AutoTokenizer.from_pretrained("./tokenizer/")
vocab_size = 131075  # 131072 + 3 special tokens

# Kronecker embeddings
kron_cfg = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192)
pf_codec = KroneckerEmbeddings(kron_cfg)

# Model (local config: ~450M params for 24 GB)
cfg = LocalModelConfig()
model = Model3B(cfg, embedding_type="kronecker", bpe_vocab=bpe_vocab,
                pf_codec=pf_codec, use_ponder=True)
```

**BPE vocab list**: All `vocab_size` tokens are decoded to strings and passed to Model3B. The model precomputes the Kronecker table for the entire vocabulary at initialization.

### 6.2 Model3B Forward Call

```python
logits_ntp, logits_mtp, aux_loss, _, stats = model(
    inp_seq,
    next_token_ids=tgt_seq,   # For MTP
    lambda_p=lambda_p,
    force_2=force_2_effective,
    halt_thr=HALT_THR,
    capture_h1=do_credit,
    return_loss=True,
    return_memory=False,
)
```

**Returns 5 values** (vs SmolLM's 3):
- `logits_ntp`: Next-token prediction [B, T, vocab]
- `logits_mtp`: Two-step-ahead prediction [B, T, vocab] (or None)
- `aux_loss`: MoE router balance loss
- `memory_stream_out`: Cross-chunk memory (None here)
- `layer_stats`: Per-layer ponder statistics

### 6.3 Loss Computation

```python
lm_loss  = CrossEntropy(logits_ntp, tgt_seq)
mtp_loss = CrossEntropy(logits_mtp, tgt_seq)  # Same targets! (shifted by construction)
total    = lm_loss + 0.3 × mtp_loss + aux_loss + ponder_loss
```

The 0.3 weight on MTP was empirically chosen to prevent the auxiliary prediction task from dominating gradients.

### 6.4 Recurrence Quality from h_1

```python
# Collect h_1 tensors (hidden state after exactly 1 ponder pass) from all layers
h1_tensors = [s.get('h_1') for s in stats if s.get('h_1') is not None]
if h1_tensors:
    h1_last  = h1_tensors[-1]                    # Last layer's h_1
    logits_1 = lm_head(norm(h1_last))            # Project to vocab
    nll_1    = CrossEntropy(logits_1, tgt_seq)   # Loss with just 1 pass
    delta_ce = (nll_1 - lm_loss).clamp(min=0)   # Improvement from more passes
```

### 6.5 `Tee` stdout Mirror

```python
class Tee:
    def write(self, data):
        self._stdout.write(data)   # Print to terminal
        self._log_file.write(data) # Also write to log file
```

All `print()` calls automatically go to both terminal AND the log file. No need to explicitly write to the log file in the training loop.

### 6.6 Dual-Ascent with Higher Target

```python
TARGET_K = 2.4   # vs 1.30 in SmolLM
```

Model3B has `Kmax=4` passes. A target of 2.4 means the model should use ~2–3 passes on average, leaving room for both easy (1 pass) and hard (4 passes) tokens.

### 6.7 Dataset: Smaller Batch

```python
dataset = SYNTHStream(tokenizer, seq_len=128, batch_size=2, ...)
train_loader = DataLoader(dataset, batch_size=2, num_workers=0)
```

`seq_len=128` and `batch_size=2` to fit within 24 GB memory. With `hidden_size=512` and 8 layers, even 128-token sequences generate substantial activation memory through the ponder loop.

---

## 7. How the Files Connect

```
new-fourier-test-new.py (training loop)
    │
    ├── imports Model3B from models/recurrence_model_3b.py
    │       │
    │       ├── LightningDecoderLayer (each layer)
    │       │       ├── GatedDeltaNet (75% of layers)
    │       │       ├── GatedSparseAttention (25% of layers)
    │       │       ├── MoEFFN (all layers)
    │       │       └── ContinueHead (when use_ponder=True)
    │       │
    │       ├── PureHybridEmbeddingTorch → KroneckerEmbeddings
    │       │
    │       └── ReversibleMidpointStack (when use_ponder=False)
    │               └── imported from models/reversible_ops_midpoint.py
    │                       ├── MidpointBlock (per layer)
    │                       └── MidpointFunction (custom autograd)
    │
    ├── SYNTHStream → PleIAs/SYNTH dataset
    ├── TSAI 131K tokenizer
    └── Dual-ascent controller (inline code)
```

**When `use_ponder=True`** (default in new-fourier-test-new.py):
- `Model3B.forward()` loops through layers directly
- Each `LightningDecoderLayer._ponder_forward()` runs the adaptive depth loop
- `ReversibleMidpointStack` is NOT used (memory efficiency traded for per-layer stats)

**When `use_ponder=False`**:
- `Model3B.stack` (a `ReversibleMidpointStack`) processes all layers
- Memory-efficient backprop via custom autograd
- No per-layer ponder statistics available

---

## 8. Key Hyperparameter Reference

### Ponder Controller

| Param | SmolLM | Model3B | Meaning |
|---|---|---|---|
| `Kmax` | 4 | 4 | Max recurrence passes per layer |
| `TARGET_K` | 1.30 | 2.40 | Target average depth |
| `DUAL_LR` | 5e-5 | 5e-5 | Lambda controller step size |
| `LAMBDA_MIN` | 1e-4 | 1e-4 | Minimum ponder penalty |
| `LAMBDA_MAX` | 5e-2 | 5e-2 | Maximum ponder penalty |
| `HALT_THR` | 0.10 | 0.10 | Survival threshold to stop |
| `REWARD_ETA` | 2.0 | 2.0 | Coupon multiplier |
| `AUDIT_PROB` | 0.05 | 0.05 | Fraction of audit steps |
| `PONDER_START` | 1200 | 1200 | Step to enable ponder loss |

### Learning Rate Schedule

```
Steps 0–500:    Linear warmup 0 → 3e-4
Steps 500–10k:  Cosine decay 3e-4 → 3e-5
```

### Deadband Controller

| Param | Value | Meaning |
|---|---|---|
| `DEADBAND_TRIGGER_LAM` | 0.50 | Lambda must exceed 50% of range |
| `DEADBAND_TRIGGER_WINDOW` | 100 | For 100 consecutive steps |
| `DEADBAND_HIGH_PCT` | 0.05 | +5% above target → raise lambda |
| `DEADBAND_LOW_PCT` | 0.02 | -2% below target → lower lambda |
| `DEADBAND_LR_SCALE` | 0.50 | Slow adjustments in deadband |

### Reversible Midpoint

| Param | Value | Meaning |
|---|---|---|
| `step_size (h)` | 0.25 | Integration step size |
| `a` | 0.5 | Blend coefficient (0=pure midpoint, 1=pure leapfrog) |
| `bootstrap` | "euler" | Half-step Euler for initialization |

---

## 9. Production Blockers & TODOs

### Critical (Blocking 256k Context)

1. **Triton kernel for GatedDeltaNet recurrence** — the Python for-loop is 500–2000× slower than needed. Requires implementing a fused parallel associative scan kernel. Estimated 2–3 weeks.

2. **Triton kernel for GatedSparseAttention** — the O(T²) match_logits tensor causes OOM at 256k. Requires block-sparse indexing that avoids materializing the full T×T matrix. Estimated 1–2 weeks.

### Recommended (For Production Quality)

3. **MoE dispatch fusion** — Python loop over experts adds overhead at scale. Use Tutel/Megablocks-style batched dispatch.

4. **RoPE benchmark** — determine whether on-the-fly or cached is better at production batch sizes.

5. **Production monitoring hooks** — track aux_loss ratio, expert load balance, memory stream gradient norms, activation statistics.

### Architecture Notes

- `n_streams=1` (LocalModelConfig) effectively disables mHC — a good simplification for local training
- `enable_mtp=False` (LocalModelConfig) disables Multi-Token Prediction — reduces memory by ~15%
- `persistent=False` on `PF_table` means Kronecker embeddings are recomputed at checkpoint load — saves ~2 GB per checkpoint file

### Known Issues / Fragile Points

- `_saved_selection` in `GatedSparseAttention` — stores sparse indices as module state, which is fragile under DDP/torch.compile. Works for single-GPU training.
- SmolLM's `new-fourier-test.py` has a dead code block after `return buf[0, :cur_len]` (lines 273–282) — unreachable but harmless.
- `device.type` check in SmolLM generation block (line 1543) uses string comparison instead of `device.type` — works but inconsistent with the rest of the code.

---

*Documentation generated from reading all source files. Last updated: 2026-02-22.*
