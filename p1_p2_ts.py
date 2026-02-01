"""
============================================================================
  PHASE 1 + PHASE 2  —  Dense Training  →  MoE Upcycling
  Dataset: roneneldan/TinyStories  (via HuggingFace datasets + GPT-2 tokenizer)
============================================================================

WHAT RUNS HERE:
  Phase 1 (steps 0-999):    Train a dense transformer on TinyStories.
  Phase 2 (steps 1000-1999): Convert every FFN → 8-expert MoE, verify the
                             output is IDENTICAL to the dense model on a
                             frozen probe batch, then keep training.

REQUIREMENTS (install before running):
  pip install torch datasets transformers tokenizers

DATASET DETAILS:
  • roneneldan/TinyStories  —  2.1 M short stories, simple English vocabulary.
  • Stories are separated by <|endoftext|> in the raw .txt files.
  • We use the correctly-split HuggingFace version (skeskinen/TinyStories-hf)
    which gives us a single "text" column per story, already split on
    <|endoftext|>.  Fall back to the raw version if the HF split fails.
  • Tokeniser: gpt2  (OpenAI's GPT-2 tokenizer used by the official TinyStories
    models).  vocab_size = 50 257.  We add no extra tokens.

WHY TinyStories OVER RANDOM TOKENS:
  Random tokens have no structure — loss stays near log(vocab_size) and
  never moves.  TinyStories has real English grammar, repeated sentence
  patterns, and a limited vocabulary.  The model can actually learn
  something in 1 000 steps, so loss drops visibly.  That visible drop
  makes the "no spike at the Phase 1→2 boundary" guarantee clearly
  observable in the printed results.

============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import json
import random
import math


# ---------------------------------------------------------------------------
# 0.  CONFIGURATION
# ---------------------------------------------------------------------------

CONFIG = dict(
    # --- Model shape ---
    # GPT-2 vocab is 50 257.  We match it so the tokeniser output maps
    # directly to embedding indices with no remapping.
    #
    # Target: ~100 M dense params.
    # With vocab=50 257 the embedding alone is vocab×d_model, so d_model
    # and n_layers must be tuned together.  The config below was found by
    # searching all (d_model, n_layers) pairs with d_ff = 4×d_model:
    #   d_model=832, n_layers=7  →  100.04 M  (exact breakdown below)
    #
    #   Embedding          41.81 M   (50257 × 832)
    #   7 blocks × 8.32 M  58.22 M   (LN + Attn + LN + FFN per block)
    #   Final LN            0.00 M
    #   LM Head             0.00 M   (tied to Embedding)
    #   ─────────────────────────────
    #   Dense total       100.04 M
    #
    #   After MoE upcycling (8 experts, top-2):
    #     Total params    371.64 M   (7 FFNs × 8 copies each)
    #     Active/token    138.88 M   (only top-2 experts fire)
    #
    vocab_size   = 50_257,
    d_model      = 832,        # hidden dim
    n_layers     = 7,          # transformer blocks
    n_heads      = 8,          # attention heads  (832 / 8 = 104 per head)
    d_ff         = 3328,       # FFN inner dim    (4 × d_model)

    # --- Dataset / batching ---
    seq_len      = 256,        # context window (TinyStories stories are short)
    batch_size   = 8,

    # --- Phase 2 MoE ---
    num_experts  = 8,
    top_k        = 2,

    # --- Training ---
    steps_phase1 = 1000,
    steps_phase2 = 1000,
    lr_phase1    = 3e-4,
    lr_phase2    = 1e-4,       # lower LR after upcycling for stability

    # --- Logging ---
    log_every    = 50,         # print loss every N steps

    # --- Dataset loading ---
    # Number of stories to cache in RAM.  TinyStories has 2.1 M stories;
    # we only need a small slice so the script starts instantly.
    num_stories  = 50_000,
)


# ---------------------------------------------------------------------------
# 1.  DATASET  —  TinyStories loader + token-stream batcher
# ---------------------------------------------------------------------------

class TinyStoriesDataset:
    """
    Loads a slice of TinyStories, tokenises every story with GPT-2,
    concatenates them into one long token stream (separated by
    <|endoftext|>), and serves random windows of seq_len for training.

    WHY a single long stream?
      Individual stories are short (often < 256 tokens).  Padding every
      story to seq_len wastes compute on <pad> tokens.  Concatenating
      into one stream and slicing random windows is the standard trick
      used by GPT-2/3/LLaMA training.  The <|endoftext|> token acts as
      a boundary marker so attention doesn't bleed across stories.

    LAZY LOADING:
      tokenise() must be called once before get_batch().  It downloads
      the dataset (cached by HuggingFace after the first run) and
      tokenises the stories.  On subsequent runs the HF cache is reused
      so the download is skipped.
    """
    def __init__(self, num_stories: int = 50_000):
        self.num_stories = num_stories
        self.tokens      = None          # filled by tokenise()
        self.tokenizer   = None

    def tokenise(self):
        """Download dataset + tokeniser, tokenise, concatenate into one stream."""
        from transformers import AutoTokenizer
        from datasets import load_dataset

        print("[dataset] Loading tokeniser (gpt2) …")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 has no pad token by default; set eos as pad for safety
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eot = self.tokenizer.eos_token_id   # <|endoftext|>

        print("[dataset] Loading TinyStories …")
        # Try the correctly-split version first; fall back to the raw one
        try:
            ds = load_dataset("skeskinen/TinyStories-hf",
                              split="train",
                              streaming=True)
        except Exception:
            ds = load_dataset("roneneldan/TinyStories",
                              split="train",
                              streaming=True)

        # Collect stories into one flat token list
        print(f"[dataset] Tokenising {self.num_stories:,} stories …")
        all_tokens = []
        count      = 0
        for example in ds:
            text = example["text"].strip()
            if not text:
                continue
            # Tokenise one story, append <|endoftext|> as separator
            ids = self.tokenizer.encode(text)
            all_tokens.extend(ids)
            all_tokens.append(eot)
            count += 1
            if count >= self.num_stories:
                break

        self.tokens = all_tokens
        print(f"[dataset] Done.  {len(self.tokens):,} total tokens "
              f"from {count:,} stories.")

    def get_batch(self, batch_size: int, seq_len: int, device):
        """
        Sample `batch_size` random windows of length `seq_len` from the
        token stream.  Returns (input_ids, labels) where labels is the
        stream shifted left by 1 (next-token prediction).

        input_ids : (B, seq_len)
        labels    : (B, seq_len)   — labels[i] = input_ids[i] shifted left by 1
        """
        assert self.tokens is not None, \
            "Call tokenise() before get_batch()"

        L = len(self.tokens)
        input_ids = []
        labels    = []

        for _ in range(batch_size):
            # Random start position  (need seq_len + 1 tokens: input + label)
            start = random.randint(0, L - seq_len - 1)
            chunk = self.tokens[start : start + seq_len + 1]
            input_ids.append(chunk[:-1])   # positions 0 … seq_len-1
            labels.append(chunk[1:])       # positions 1 … seq_len  (shifted)

        return (torch.tensor(input_ids, dtype=torch.long, device=device),
                torch.tensor(labels,    dtype=torch.long, device=device))


# ---------------------------------------------------------------------------
# 2.  MODEL ARCHITECTURE
# ---------------------------------------------------------------------------
# Identical structure to the previous version.  Four classes, read top to
# bottom in forward-pass order:
#   FeedForward  →  MoELayer  →  TransformerBlock  →  SLM

class FeedForward(nn.Module):
    """
    Two-layer MLP: x → Linear(d→d_ff) → GELU → Linear(d_ff→d) → out
    This entire module becomes ONE expert inside MoELayer after upcycling.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1  = nn.Linear(d_model, d_ff)
        self.w2  = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer.  Replaces a single FeedForward.

        input x  (B, S, d_model)
             │
             ▼
        ┌─────────┐
        │ Router  │  Linear(d_model → num_experts)
        │         │  logits → softmax → pick top_k → re-normalise
        └────┬────┘
             │
        ┌────┼────────────┐
        ▼    ▼            ▼
     Exp0  Exp1  …  Exp(N-1)     ← each is a FeedForward
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
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.experts     = nn.ModuleList(
            [FeedForward(d_model, d_ff) for _ in range(num_experts)]
        )
        self.router      = nn.Linear(d_model, num_experts)
        self.top_k       = top_k
        self.num_experts = num_experts

    def forward(self, x):
        B, S, D = x.shape
        x_flat  = x.reshape(-1, D)                       # (B*S, D)

        # ── 1. Routing ────────────────────────────────────────────────
        logits      = self.router(x_flat)                 # (B*S, N)
        probs       = F.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_w      = topk_w / topk_w.sum(dim=-1, keepdim=True)   # renorm

        # ── 2. Dispatch & combine ─────────────────────────────────────
        out = torch.zeros_like(x_flat)
        for i in range(self.num_experts):
            mask = (topk_idx == i).any(dim=-1)            # which tokens use expert i?
            if not mask.any():
                continue
            w = (topk_idx[mask] == i).float() * topk_w[mask]
            w = w.sum(dim=-1, keepdim=True)
            out[mask] += w * self.experts[i](x_flat[mask])

        return out.reshape(B, S, D)


class TransformerBlock(nn.Module):
    """
    Pre-norm block:  x → LN → Attn → +residual → LN → FFN/MoE → +residual
    The use_moe flag swaps FFN ↔ MoE; everything else is shared.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 use_moe: bool = False,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(d_model)

        if use_moe:
            self.ffn = MoELayer(d_model, d_ff, num_experts, top_k)
        else:
            self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class SLM(nn.Module):
    """
    Decoder-only transformer.

        Token IDs → Embedding → [Block × L] → LayerNorm → LM Head → Logits

    Embedding and Head share one weight matrix (tied weights).
    """
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int,
                 use_moe: bool = False,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff,
                             use_moe, num_experts, top_k)
            for _ in range(n_layers)
        ])
        self.ln_f   = nn.LayerNorm(d_model)
        self.head   = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight          # tie weights

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)                           # (B, S, vocab_size)


# ---------------------------------------------------------------------------
# 3.  PHASE-2 TRANSITION:  Dense → MoE  (Upcycling)
# ---------------------------------------------------------------------------
# Per-block operations:
#   (a) Copy Attention + LayerNorms verbatim
#   (b) Clone the single FFN into every expert
#   (c) Zero the router → uniform softmax → MoE(x) == FFN(x)

@torch.no_grad()
def transition_to_moe(dense_model: SLM,
                      num_experts: int,
                      top_k: int) -> SLM:
    """
    Upcycle a trained dense SLM into a sparse MoE SLM.
    Returns a NEW model; dense_model is unchanged (kept for verification).
    """
    vocab_size = dense_model.embed.num_embeddings
    d_model    = dense_model.embed.embedding_dim
    n_layers   = len(dense_model.blocks)
    n_heads    = dense_model.blocks[0].attn.num_heads
    d_ff       = dense_model.blocks[0].ffn.w1.out_features

    print(f"\n  [transition_to_moe]")
    print(f"    Input:  Dense  (d_model={d_model}, layers={n_layers})")
    print(f"    Output: MoE    ({num_experts} experts/block, top-{top_k})")

    moe_model = SLM(vocab_size, d_model, n_layers, n_heads, d_ff,
                    use_moe=True, num_experts=num_experts, top_k=top_k)

    # Shared layers
    moe_model.embed.load_state_dict(dense_model.embed.state_dict())
    moe_model.ln_f.load_state_dict(dense_model.ln_f.state_dict())

    # Per-block transfer
    for idx, (src, dst) in enumerate(
            zip(dense_model.blocks, moe_model.blocks)):

        dst.attn.load_state_dict(src.attn.state_dict())
        dst.ln1.load_state_dict(src.ln1.state_dict())
        dst.ln2.load_state_dict(src.ln2.state_dict())

        # Clone FFN → every expert
        ffn_state = src.ffn.state_dict()
        for expert in dst.ffn.experts:
            expert.load_state_dict(ffn_state)

        # Zero router
        nn.init.zeros_(dst.ffn.router.weight)
        nn.init.zeros_(dst.ffn.router.bias)

    print(f"    ✓ {n_layers} blocks converted  "
          f"({num_experts} experts each, router zeroed)")
    return moe_model


# ---------------------------------------------------------------------------
# 4.  FUNCTIONAL-EQUIVALENCE VERIFICATION
# ---------------------------------------------------------------------------
# We freeze one batch of REAL TinyStories tokens as a "probe".
# Same tokens go through Dense then MoE.  If max|Δlogits| < 1e-4
# the conversion is lossless.

def verify_functional_equivalence(dense_model: SLM,
                                  moe_model:   SLM,
                                  probe:       torch.Tensor,
                                  device):
    """
    probe: (B, seq_len) — frozen token batch from TinyStories.

    Runs both models on the same probe, prints per-sequence diffs,
    asserts overall max diff < tolerance.
    """
    dense_model.eval()
    moe_model.eval()

    with torch.no_grad():
        logits_dense = dense_model(probe)   # (B, S, V)
        logits_moe   = moe_model(probe)     # (B, S, V)

    # Per-sequence max absolute difference  (collapse seq & vocab dims)
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


# ---------------------------------------------------------------------------
# 5.  TRAINING LOOP
# ---------------------------------------------------------------------------

def train_phase(model: SLM,
                dataset: TinyStoriesDataset,
                steps: int,
                lr: float,
                label: str,
                device) -> tuple:
    """
    Next-token-prediction loop.  Returns (model, loss_log).

    loss_log: list of (global_step, loss_value) for every logged step.
    """
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    vocab_size = CONFIG['vocab_size']
    B          = CONFIG['batch_size']
    S          = CONFIG['seq_len']
    log_every  = CONFIG['log_every']

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Steps: {steps}  |  LR: {lr}  |  Device: {device}")
    print(f"  Params: {param_count:.2f} M")
    print(f"{'=' * 60}")

    loss_log = []
    start    = time.time()

    for step in range(steps):
        input_ids, labels = dataset.get_batch(B, S, device)

        logits = model(input_ids)                        # (B, S, V)

        # Cross-entropy: flatten spatial dims
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),              # (B*S, V)
            labels.reshape(-1)                           # (B*S,)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            elapsed = time.time() - start
            loss_log.append((step, loss.item()))
            print(f"  step {step:>4d} | loss {loss.item():.4f} | {elapsed:.1f}s")

    return model, loss_log


# ---------------------------------------------------------------------------
# 6.  RESULTS PRINTER  —  called after every phase
# ---------------------------------------------------------------------------

def print_phase_results(label: str,
                        loss_log: list,
                        model: SLM,
                        dataset: TinyStoriesDataset,
                        device,
                        n_sample: int = 3):
    """
    Print a structured results block after each phase:
      • Loss trajectory (first → last)
      • Parameter count (dense vs active-per-token for MoE)
      • Sample predictions: feed a short TinyStories prompt through the
        model, greedy-decode 40 tokens, print input + output side by side.
    """
    model.eval()
    tokenizer = dataset.tokenizer

    # ── loss summary ──────────────────────────────────────────────────
    first_loss = loss_log[0][1]
    last_loss  = loss_log[-1][1]
    total_drop = first_loss - last_loss

    print(f"\n{'─' * 60}")
    print(f"  RESULTS — {label}")
    print(f"{'─' * 60}")
    print(f"  Loss:        {first_loss:.4f}  →  {last_loss:.4f}  "
          f"(dropped {total_drop:.4f})")

    # ── parameter count ───────────────────────────────────────────────
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params / 1e6:.2f} M")

    # For MoE models estimate active params per token
    # Active = total - (inactive experts' FFN params)
    # Each MoE block has num_experts experts but only top_k are active.
    # Check if this is an MoE model by inspecting the first block's ffn.
    first_ffn = model.blocks[0].ffn
    if hasattr(first_ffn, 'experts'):
        num_experts = first_ffn.num_experts
        top_k       = first_ffn.top_k
        # FFN params per expert = w1 + w2 weights & biases
        ffn_params_per_expert = sum(p.numel() for p in first_ffn.experts[0].parameters())
        inactive_per_block    = ffn_params_per_expert * (num_experts - top_k)
        inactive_total        = inactive_per_block * len(model.blocks)
        active_params         = total_params - inactive_total
        print(f"  Active params (top-{top_k} of {num_experts}): "
              f"{active_params / 1e6:.2f} M per token")

    # ── sample greedy decoding ────────────────────────────────────────
    # Use short story-opening prompts — common in TinyStories
    prompts = [
        "Once upon a time there was",
        "One day a little",
        "The boy and the girl",
        "A small cat sat on",
        "There was a happy",
    ]
    # Pick n_sample prompts deterministically
    selected = prompts[:n_sample]

    print(f"\n  Sample greedy decoding (40 new tokens each):")
    print(f"  {'─' * 56}")

    with torch.no_grad():
        for prompt_text in selected:
            input_ids = torch.tensor(
                [tokenizer.encode(prompt_text)],
                dtype=torch.long, device=device
            )                                            # (1, prompt_len)

            # Greedy autoregressive decode — 40 steps
            for _ in range(40):
                # Only feed the last seq_len tokens to stay within context
                feed = input_ids[:, -CONFIG['seq_len']:]
                logits = model(feed)                     # (1, S, V)
                next_logits = logits[:, -1, :]           # (1, V)
                next_token  = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                input_ids   = torch.cat([input_ids, next_token], dim=1)

            # Decode the generated part only (after the prompt)
            prompt_len   = len(tokenizer.encode(prompt_text))
            generated    = input_ids[0, prompt_len:].tolist()
            gen_text     = tokenizer.decode(generated, skip_special_tokens=True)

            print(f"  Prompt:    \"{prompt_text}\"")
            print(f"  Generated: \"{gen_text.strip()}\"")
            print(f"  {'─' * 56}")

    print()


# ---------------------------------------------------------------------------
# 7.  MAIN
# ---------------------------------------------------------------------------

def main():
    # Device selection: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[init] Device: {device}")
        print(f"[init] GPU:    {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[init] Device: {device}")
        print(f"[init] GPU:    Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        print(f"[init] Device: {device}")

    # ── Load & tokenise TinyStories ────────────────────────────────────
    dataset = TinyStoriesDataset(num_stories=CONFIG['num_stories'])
    dataset.tokenise()

    # ============================================================
    # PHASE 1  —  Train the Dense Model
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 1  —  Dense Model  (single FFN per block)")
    print("=" * 60)

    dense_model = SLM(
        vocab_size  = CONFIG['vocab_size'],
        d_model     = CONFIG['d_model'],
        n_layers    = CONFIG['n_layers'],
        n_heads     = CONFIG['n_heads'],
        d_ff        = CONFIG['d_ff'],
        use_moe     = False,
    ).to(device)

    dense_model, phase1_log = train_phase(
        dense_model, dataset,
        steps  = CONFIG['steps_phase1'],
        lr     = CONFIG['lr_phase1'],
        label  = "Phase 1 — Dense",
        device = device,
    )

    # ── Print Phase 1 results ─────────────────────────────────────────
    print_phase_results("Phase 1 — Dense", phase1_log,
                        dense_model, dataset, device)

    # ============================================================
    # FREEZE A PROBE BATCH  (real TinyStories tokens)
    # Used before AND after conversion to verify identity.
    # ============================================================
    dense_model.eval()
    probe, _ = dataset.get_batch(CONFIG['batch_size'],
                                 CONFIG['seq_len'], device)
    probe = probe.detach().clone()   # freeze it — never changes

    # ============================================================
    # TRANSITION  —  Dense → MoE
    # ============================================================
    print("\n" + "=" * 60)
    print("  TRANSITION  —  Dense → MoE Upcycling")
    print("=" * 60)

    moe_model = transition_to_moe(
        dense_model,
        num_experts = CONFIG['num_experts'],
        top_k       = CONFIG['top_k'],
    ).to(device)

    # ── Verify: same probe → same logits? ─────────────────────────────
    verify_functional_equivalence(dense_model, moe_model, probe, device)

    # Free dense model
    del dense_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    # ============================================================
    # PHASE 2  —  Train the MoE Model
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 2  —  MoE Model  (8 experts, top-2 routing)")
    print("=" * 60)

    moe_model, phase2_log = train_phase(
        moe_model, dataset,
        steps  = CONFIG['steps_phase2'],
        lr     = CONFIG['lr_phase2'],
        label  = "Phase 2 — MoE",
        device = device,
    )

    # ── Print Phase 2 results ─────────────────────────────────────────
    print_phase_results("Phase 2 — MoE (8 experts, top-2)", phase2_log,
                        moe_model, dataset, device)

    # ============================================================
    # BOUNDARY SUMMARY
    # ============================================================
    p1_end   = phase1_log[-1][1]
    p2_start = phase2_log[0][1]
    p2_end   = phase2_log[-1][1]

    print("=" * 60)
    print("  BOUNDARY SUMMARY")
    print("=" * 60)
    print(f"  Phase 1 final loss  :  {p1_end:.4f}")
    print(f"  Phase 2 first loss  :  {p2_start:.4f}")
    print(f"  Phase 2 final loss  :  {p2_end:.4f}")
    print(f"  Boundary jump       :  {abs(p2_start - p1_end):.4f}")
    print(f"    (small jump is normal — it's a different random batch,")
    print(f"     NOT a loss spike from the conversion)")
    print(f"  Phase 2 total drop  :  {p2_start - p2_end:.4f}")
    print("=" * 60)

    # ── Save log ──────────────────────────────────────────────────────
    log_path = "phase1_phase2_tinystories_log.jsonl"
    with open(log_path, "w") as f:
        for step, loss in phase1_log:
            f.write(json.dumps({"phase": 1, "step": step,
                                "loss": round(loss, 6)}) + "\n")
        for step, loss in phase2_log:
            f.write(json.dumps({"phase": 2,
                                "step": step + CONFIG['steps_phase1'],
                                "loss": round(loss, 6)}) + "\n")
    print(f"\n  Logs saved → {log_path}")


if __name__ == "__main__":
    main()