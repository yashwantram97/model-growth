"""
============================================================================
  PHASE 1 + PHASE 2  —  Dense Training  →  MoE Upcycling (Modular Version)
  Dataset: roneneldan/TinyStories  (via HuggingFace datasets + GPT-2 tokenizer)
============================================================================

WHAT RUNS HERE:
  Phase 1 (steps 0-999):    Train a dense transformer on TinyStories.
  Phase 2 (steps 1000-1999): Convert every FFN → 8-expert MoE, verify the
                             output is IDENTICAL to the dense model on a
                             frozen probe batch, then keep training.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import time
import json
from pathlib import Path

from config import ModelConfig, TrainingConfig
from models.simple_model import SLM
from utils.data import TinyStoriesDataset
from transfer.simple_transfer import transition_to_moe, verify_functional_equivalence


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
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
    return device


def train_phase(model, dataset, steps, lr, label, device, model_config, training_config):
    """
    Next-token-prediction loop. Returns (model, loss_log).

    loss_log: list of (global_step, loss_value) for every logged step.
    """
    model.train()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    vocab_size = model_config.vocab_size
    B = training_config.batch_size
    S = model_config.seq_len
    log_every = training_config.log_every

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"  Steps: {steps}  |  LR: {lr}  |  Device: {device}")
    print(f"  Params: {param_count:.2f} M")
    print(f"{'=' * 60}")

    loss_log = []
    start = time.time()

    for step in range(steps):
        input_ids, labels = dataset.get_batch(B, S, device)

        logits = model(input_ids)  # (B, S, V)

        # Cross-entropy: flatten spatial dims
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),  # (B*S, V)
            labels.reshape(-1)               # (B*S,)
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


def print_phase_results(label, loss_log, model, dataset, device, model_config, training_config):
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
    last_loss = loss_log[-1][1]
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
    first_ffn = model.blocks[0].ffn
    if hasattr(first_ffn, 'experts'):
        num_experts = first_ffn.num_experts
        top_k = first_ffn.top_k
        # FFN params per expert = w1 + w2 weights & biases
        ffn_params_per_expert = sum(p.numel() for p in first_ffn.experts[0].parameters())
        inactive_per_block = ffn_params_per_expert * (num_experts - top_k)
        inactive_total = inactive_per_block * len(model.blocks)
        active_params = total_params - inactive_total
        print(f"  Active params (top-{top_k} of {num_experts}): "
              f"{active_params / 1e6:.2f} M per token")

    # ── sample greedy decoding ────────────────────────────────────────
    prompts = [
        "Once upon a time there was",
        "One day a little",
        "The boy and the girl",
        "A small cat sat on",
        "There was a happy",
    ]
    selected = prompts[:training_config.n_sample_prompts]

    print(f"\n  Sample greedy decoding ({training_config.sample_tokens} new tokens each):")
    print(f"  {'─' * 56}")

    with torch.no_grad():
        for prompt_text in selected:
            input_ids = torch.tensor(
                [tokenizer.encode(prompt_text)],
                dtype=torch.long, device=device
            )  # (1, prompt_len)

            # Greedy autoregressive decode
            for _ in range(training_config.sample_tokens):
                # Only feed the last seq_len tokens to stay within context
                feed = input_ids[:, -model_config.seq_len:]
                logits = model(feed)  # (1, S, V)
                next_logits = logits[:, -1, :]  # (1, V)
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

            # Decode the generated part only (after the prompt)
            prompt_len = len(tokenizer.encode(prompt_text))
            generated = input_ids[0, prompt_len:].tolist()
            gen_text = tokenizer.decode(generated, skip_special_tokens=True)

            print(f"  Prompt:    \"{prompt_text}\"")
            print(f"  Generated: \"{gen_text.strip()}\"")
            print(f"  {'─' * 56}")

    print()


def main():
    """Main training pipeline."""
    device = get_device()

    # ── Configuration ──────────────────────────────────────────────────
    model_config = ModelConfig()
    training_config = TrainingConfig()

    # ── Load & tokenize TinyStories ────────────────────────────────────
    dataset = TinyStoriesDataset(num_stories=training_config.num_stories)
    dataset.tokenize()

    # Create checkpoint directory
    Path(training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # PHASE 1  —  Train the Dense Model
    # ============================================================
    print("\n" + "=" * 60)
    print("  PHASE 1  —  Dense Model  (single FFN per block)")
    print("=" * 60)

    dense_model = SLM(
        vocab_size=model_config.vocab_size,
        d_model=model_config.d_model,
        n_layers=model_config.n_layers,
        n_heads=model_config.n_heads,
        d_ff=model_config.d_ff,
        use_moe=False,
    ).to(device)

    dense_model, phase1_log = train_phase(
        dense_model, dataset,
        steps=training_config.steps_phase1,
        lr=training_config.lr_phase1,
        label="Phase 1 — Dense",
        device=device,
        model_config=model_config,
        training_config=training_config,
    )

    # ── Print Phase 1 results ─────────────────────────────────────────
    print_phase_results("Phase 1 — Dense", phase1_log,
                        dense_model, dataset, device, model_config, training_config)

    # Save dense model
    checkpoint_path = Path(training_config.checkpoint_dir) / "dense_model.pt"
    torch.save({
        'model_state_dict': dense_model.state_dict(),
        'config': model_config,
        'loss_log': phase1_log,
    }, checkpoint_path)
    print(f"✓ Saved dense model to {checkpoint_path}")

    # ============================================================
    # FREEZE A PROBE BATCH  (real TinyStories tokens)
    # Used before AND after conversion to verify identity.
    # ============================================================
    dense_model.eval()
    probe, _ = dataset.get_batch(training_config.batch_size,
                                 model_config.seq_len, device)
    probe = probe.detach().clone()  # freeze it — never changes

    # ============================================================
    # TRANSITION  —  Dense → MoE
    # ============================================================
    print("\n" + "=" * 60)
    print("  TRANSITION  —  Dense → MoE Upcycling")
    print("=" * 60)

    moe_model = transition_to_moe(
        dense_model,
        num_experts=model_config.num_experts,
        top_k=model_config.top_k,
    ).to(device)

    # ── Verify: same probe → same logits? ─────────────────────────────
    verify_functional_equivalence(dense_model, moe_model, probe, device)

    # Save initial MoE model
    checkpoint_path = Path(training_config.checkpoint_dir) / "moe_model_init.pt"
    torch.save({
        'model_state_dict': moe_model.state_dict(),
        'config': model_config,
    }, checkpoint_path)
    print(f"✓ Saved initial MoE model to {checkpoint_path}")

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
        steps=training_config.steps_phase2,
        lr=training_config.lr_phase2,
        label="Phase 2 — MoE",
        device=device,
        model_config=model_config,
        training_config=training_config,
    )

    # ── Print Phase 2 results ─────────────────────────────────────────
    print_phase_results("Phase 2 — MoE (8 experts, top-2)", phase2_log,
                        moe_model, dataset, device, model_config, training_config)

    # Save final MoE model
    checkpoint_path = Path(training_config.checkpoint_dir) / "moe_model_final.pt"
    torch.save({
        'model_state_dict': moe_model.state_dict(),
        'config': model_config,
        'loss_log': phase2_log,
    }, checkpoint_path)
    print(f"✓ Saved final MoE model to {checkpoint_path}")

    # ============================================================
    # BOUNDARY SUMMARY
    # ============================================================
    p1_end = phase1_log[-1][1]
    p2_start = phase2_log[0][1]
    p2_end = phase2_log[-1][1]

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
    log_path = Path(training_config.checkpoint_dir) / "training_history.jsonl"
    with open(log_path, "w") as f:
        for step, loss in phase1_log:
            f.write(json.dumps({"phase": 1, "step": step,
                                "loss": round(loss, 6)}) + "\n")
        for step, loss in phase2_log:
            f.write(json.dumps({"phase": 2,
                                "step": step + training_config.steps_phase1,
                                "loss": round(loss, 6)}) + "\n")
    print(f"\n  Logs saved → {log_path}")
    print("\n✓ Experiment completed successfully!")


if __name__ == "__main__":
    main()
