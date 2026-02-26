#!/usr/bin/env python3
"""
train.py  — Diagnostic Training Run
====================================
Purpose: Verify that loss decreases properly over a few thousand steps.

Config:
  - Model:     Model3B (LocalModelConfig, ~450M params, fits M4 Pro 24 GB)
  - Dataset:   synth_local  (PleIAs/SYNTH, pre-downloaded Arrow)
  - Device:    CUDA  |  CPU fallback
  - Loss:      NTP + 0.3×MTP + aux_loss + ponder_loss
  - Logging:   Every step → training_log.jsonl  (one JSON line per step)
               Every 50 steps → human-readable summary to stdout + training_log.txt
  - Checkpoint: saves to checkpoints/train_latest.pt every 500 steps

Run:
  python train.py [--steps 3000] [--batch 2] [--seq 128] [--lr 3e-4]
"""

import os
import sys
import json
import math
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Sys-path: make  `models/`  importable from wherever we're run from
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS = os.path.join(_HERE, "models")
for _p in (_HERE, _MODELS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.recurrence_model_3b_unmodified import (   # noqa: E402
    KroneckerEmbeddings,
    KroneckerConfig,
    Model3B,
    LocalModelConfig,
)

logging.basicConfig(level=logging.WARNING)   # keep model chatter quiet


# ===========================================================================
# Helpers
# ===========================================================================

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def cosine_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * max(step, 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * progress))


# ===========================================================================
# SYNTH streaming dataset  (packs tokens into fixed-length blocks)
# ===========================================================================

class SYNTHStream(IterableDataset):
    """
    Streams examples from synth_local, packs tokens into seq_len blocks.
    Yields dict with keys 'ids' and 'targets' (both LongTensors of length seq_len).
    """

    def __init__(
        self,
        tokenizer,
        local_path: str = "synth_local",
        seq_len: int = 128,
        seed: int = 42,
        start_step: int = 0,
        batch_size: int = 2,
    ):
        super().__init__()
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.seed       = seed
        self.start_step = start_step
        self.batch_size = batch_size

        # Resolve dataset path
        candidates = [
            os.path.join(os.getcwd(), local_path),
            os.path.join(_HERE, local_path),
        ]
        self.full_path = next((p for p in candidates if os.path.exists(p)), None)
        if self.full_path:
            print(f"[DATA] Loading synth_local from: {self.full_path}")
        else:
            print(f"[DATA] ⚠️  synth_local not found — will use dummy data")

    def _build_text(self, ex: dict) -> Optional[str]:
        lang = ex.get("language", "")
        if not lang or str(lang).lower() != "en":
            return None
        # Use plain-text separators — no special tokens.
        # TSAI 131K tokenizer maps <|im_start|>, <think>, etc. to IDs > 65535,
        # which all clamp to the same token (65535) and inject noise at every
        # dialogue boundary. Plain ASCII headers avoid this completely.
        parts = []
        query     = (ex.get("query")               or "").strip()
        reasoning = (ex.get("synthetic_reasoning") or "").strip()
        answer    = (ex.get("synthetic_answer")    or "").strip()
        if query:
            parts.append(f"Question: {query}")
        if reasoning:
            parts.append(f"Thinking: {reasoning}")
        if answer:
            parts.append(f"Answer: {answer}")
        return "\n\n".join(parts) if parts else None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if not self.full_path:
            # Dummy fallback so training can still run
            while True:
                ids = torch.randint(0, 65536, (self.seq_len,), dtype=torch.long)
                yield {"ids": ids, "targets": ids.clone()}
            return

        try:
            full_ds = load_from_disk(self.full_path).shuffle(seed=self.seed)
        except Exception as e:
            print(f"[DATA] ❌ Failed to load dataset: {e} — using dummy data")
            while True:
                ids = torch.randint(0, 65536, (self.seq_len,), dtype=torch.long)
                yield {"ids": ids, "targets": ids.clone()}
            return

        # Skip ahead for resuming
        skip = self.start_step * self.batch_size
        if 0 < skip < len(full_ds):
            full_ds = full_ds.select(range(skip, len(full_ds)))

        it: Iterator = iter(full_ds)
        buf: List[int] = []

        while True:
            # Fill buffer
            while len(buf) < self.seq_len:
                try:
                    ex = next(it)
                except StopIteration:
                    full_ds = load_from_disk(self.full_path).shuffle(seed=self.seed + 1)
                    it = iter(full_ds)
                    ex = next(it)

                text = self._build_text(dict(ex))
                if not text:
                    continue
                enc = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors=None,
                    max_length=self.seq_len * 4,
                    truncation=True,
                    padding=False,
                )
                ids = enc.get("input_ids", [])
                if ids:
                    buf.extend(ids)

                # Keep buffer bounded
                if len(buf) > 8 * self.seq_len:
                    buf = buf[-(4 * self.seq_len):]

            block = buf[:self.seq_len]
            del buf[:self.seq_len]
            t = torch.tensor(block, dtype=torch.long)
            yield {"ids": t, "targets": t.clone()}


# ===========================================================================
# TinyStories streaming dataset
# ===========================================================================

class TinyStoriesStream(IterableDataset):
    """
    Streams examples from roneneldan/TinyStories (HuggingFace hub).
    Each example has a plain 'text' field — no special tokens, no chat format.
    Packs tokens into seq_len blocks identical to SYNTHStream.
    """

    def __init__(self, tokenizer, seq_len: int = 256, seed: int = 42,
                 start_step: int = 0, batch_size: int = 4):
        super().__init__()
        self.tokenizer  = tokenizer
        self.seq_len    = seq_len
        self.seed       = seed
        self.start_step = start_step
        self.batch_size = batch_size
        print("[DATA] TinyStories — will stream from roneneldan/TinyStories")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from datasets import load_dataset
        try:
            ds = load_dataset(
                "roneneldan/TinyStories",
                split="train",
                streaming=True,
                trust_remote_code=False,
            )
            ds = ds.shuffle(seed=self.seed, buffer_size=10_000)
        except Exception as e:
            print(f"[DATA] ❌ TinyStories load failed: {e} — using dummy data")
            while True:
                ids = torch.randint(0, 65536, (self.seq_len,), dtype=torch.long)
                yield {"ids": ids, "targets": ids.clone()}
            return

        buf: List[int] = []
        skip = self.start_step * self.batch_size
        skipped = 0

        for ex in ds:
            text = (ex.get("text") or "").strip()
            if not text:
                continue

            if skipped < skip:
                skipped += 1
                continue

            enc = self.tokenizer(
                text,
                add_special_tokens=False,
                return_tensors=None,
                max_length=self.seq_len * 4,
                truncation=True,
                padding=False,
            )
            ids = enc.get("input_ids", [])
            if ids:
                buf.extend(ids)
            if len(buf) > 8 * self.seq_len:
                buf = buf[-(4 * self.seq_len):]

            while len(buf) >= self.seq_len:
                block = buf[:self.seq_len]
                del buf[:self.seq_len]
                t = torch.tensor(block, dtype=torch.long)
                yield {"ids": t, "targets": t.clone()}


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(model, optimizer, step: int, loss: float,
                    lr: float, save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    ckpt = {
        "step":                 step,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
        "lr":                   lr,
    }
    path = os.path.join(save_dir, "train_latest.pt")
    tmp  = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    print(f"[CKPT] 💾  step={step}  loss={loss:.4f}  lr={lr:.2e}  → {path}")


def load_checkpoint(model, optimizer, save_dir: str = "checkpoints"):
    path = os.path.join(save_dir, "train_latest.pt")
    if not os.path.exists(path):
        return 0
    try:
        print(f"[CKPT] 🔄  Loading {path}")
        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if optimizer and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        step = ckpt.get("step", 0)
        print(f"[CKPT] ✅  Resumed at step {step}  (loss={ckpt.get('loss', '?'):.4f})")
        return step
    except Exception as e:
        print(f"[CKPT] ❌  Failed to load checkpoint: {e}")
        return 0


# ===========================================================================
# Logger — writes structured JSONL + human txt
# ===========================================================================

class TrainLogger:
    """Writes one JSON line per step to .jsonl and periodic summaries to .txt."""

    def __init__(self, jsonl_path: str, txt_path: str, print_every: int = 50):
        self.jsonl_path  = jsonl_path
        self.txt_path    = txt_path
        self.print_every = print_every
        self._jsonl = open(jsonl_path, "a", buffering=1, encoding="utf-8")
        self._txt   = open(txt_path,   "a", buffering=1, encoding="utf-8")
        self._window: List[float] = []   # rolling loss for smoothing

        header = f"\n{'='*70}\n[RUN START] {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n"
        self._txt.write(header)
        self._txt.flush()

    def log(self, row: dict):
        """Write a single step record.  row must contain at least 'step' and 'loss'."""
        self._jsonl.write(json.dumps(row) + "\n")

        loss = row.get("loss", float("nan"))
        self._window.append(loss)
        if len(self._window) > self.print_every:
            self._window.pop(0)

        step = row["step"]
        if step % self.print_every == 0 or step == 0:
            smooth = sum(self._window) / len(self._window) if self._window else loss
            perp   = math.exp(min(smooth, 20))
            lr     = row.get("lr", float("nan"))
            dt     = row.get("step_ms", float("nan"))
            gnorm  = row.get("grad_norm", float("nan"))
            aux    = row.get("aux_loss", float("nan"))
            mtp    = row.get("mtp_loss", 0.0)
            ponder = row.get("ponder_loss", 0.0)

            line = (
                f"step={step:6d}  "
                f"loss={loss:.4f}  "
                f"smooth={smooth:.4f}  "
                f"ppl={perp:.1f}  "
                f"aux={aux:.4f}  "
                f"mtp={mtp:.4f}  "
                f"ponder={ponder:.4f}  "
                f"gnorm={gnorm:.3f}  "
                f"lr={lr:.2e}  "
                f"ms/step={dt:.0f}"
            )
            print(line)
            self._txt.write(line + "\n")
            self._txt.flush()

    def close(self):
        self._jsonl.close()
        self._txt.close()


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="train.py — diagnostic training run")
    parser.add_argument("--steps",   type=int,   default=5000,  help="Total training steps")
    parser.add_argument("--batch",   type=int,   default=4,     help="Batch size")
    parser.add_argument("--seq",     type=int,   default=256,   help="Sequence length")
    parser.add_argument("--lr",      type=float, default=3e-4,  help="Peak learning rate")
    parser.add_argument("--warmup",  type=int,   default=100,   help="LR warmup steps")
    parser.add_argument("--ckpt",    type=int,   default=250,   help="Save checkpoint every N steps")
    parser.add_argument("--resume",  action="store_true",        help="Auto-resume from latest checkpoint")
    parser.add_argument("--no_kronecker", action="store_true",   help="Use standard token embeddings (faster)")
    parser.add_argument("--dataset", type=str,  default="tinystories",
                        choices=["synth", "tinystories"],    help="Training dataset")
    parser.add_argument("--accum",   type=int,   default=4,     help="Gradient accumulation steps")
    parser.add_argument("--seed",    type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = detect_device()
    print(f"\n{'='*60}")
    print(f"  train.py — diagnostic run")
    print(f"  Device  : {device}")
    print(f"  Steps   : {args.steps}")
    print(f"  Batch   : {args.batch}   Seq : {args.seq}")
    print(f"  LR      : {args.lr}   Warmup : {args.warmup}")
    print(f"{'='*60}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = os.path.join(_HERE, "tokenizer")
    if os.path.exists(tok_path):
        print(f"[INIT] Loading TSAI tokenizer from {tok_path}")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)
    else:
        # Fallback for Colab / environments without the local TSAI tokenizer.
        # GPT-2 vocab (50257) is a subset of cfg.vocab_size (65536) — all IDs valid.
        print("[INIT] Local tokenizer not found — falling back to GPT-2 tokenizer.")
        print("       (download: pip install transformers; uses HuggingFace cache)")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"[INIT] Vocab size: {vocab_size:,}")

    # ── Model config (created here so Kronecker table uses cfg.vocab_size) ────
    cfg = LocalModelConfig()

    # ── Kronecker embeddings ───────────────────────────────────────────────────
    use_kronecker = not args.no_kronecker
    bpe_vocab = None
    pf_codec  = None
    if use_kronecker:
        print("[INIT] Building Kronecker vocab table (may take ~30s)…")
        from tqdm import tqdm
        kron_cfg = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192)
        pf_codec = KroneckerEmbeddings(kron_cfg)
        bpe_vocab = []
        for i in tqdm(range(cfg.vocab_size), desc="  vocab", ncols=70):
            try:
                bpe_vocab.append(tokenizer.decode([i]))
            except Exception:
                bpe_vocab.append(f"<TOKEN_{i}>")
        print("[INIT] Kronecker table ready.")
    else:
        print("[INIT] Using standard token embeddings (--no_kronecker)")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("[INIT] Building Model3B (LocalModelConfig)…")
    model = Model3B(
        cfg,
        embedding_type="kronecker" if use_kronecker else "standard",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
    ).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Parameters : {total_p:,}  ({total_p/1e6:.1f} M)")
    print(f"[INIT] Memory est : weights≈{total_p*2/1e9:.2f} GB (bf16)  "
          f"Adam≈{total_p*8/1e9:.2f} GB  "
          f"KP-table≈{cfg.vocab_size*8192*2/1e9:.2f} GB")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=False,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.dataset == "tinystories":
        dataset = TinyStoriesStream(
            tokenizer,
            seq_len=args.seq,
            seed=args.seed,
            start_step=start_step,
            batch_size=args.batch,
        )
    else:
        dataset = SYNTHStream(
            tokenizer,
            local_path="synth_local",
            seq_len=args.seq,
            seed=args.seed,
            start_step=start_step,
            batch_size=args.batch,
        )
    loader   = DataLoader(dataset, batch_size=args.batch, num_workers=0)
    data_it  = iter(loader)

    criterion = nn.CrossEntropyLoss()

    # ── Loggers ───────────────────────────────────────────────────────────────
    logger = TrainLogger(
        jsonl_path  = os.path.join(_HERE, "training_log.jsonl"),
        txt_path    = os.path.join(_HERE, "training_log.txt"),
        print_every = 50,
    )
    print(f"[INIT] Logging to: training_log.jsonl  +  training_log.txt")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[TRAIN] 🚀  Starting from step {start_step}  →  {args.steps}\n")
    model.train()

    loss_ema   = None
    EMA_ALPHA  = 0.02       # smoothing for loss EMA (printed at end)
    REWARD_ETA = 2.0

    accum_steps = max(1, args.accum)   # gradient accumulation

    try:
        for step in range(start_step, args.steps):

            # LR schedule
            lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr * 0.1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            sync(device)
            t0 = time.perf_counter()

            # Gradient accumulation: sum gradients over accum_steps micro-batches
            optimizer.zero_grad()
            lm_loss_accum  = 0.0
            mtp_loss_accum = 0.0
            aux_val_accum  = 0.0

            for micro in range(accum_steps):
                # Fetch batch
                try:
                    batch = next(data_it)
                except StopIteration:
                    data_it = iter(loader)
                    batch   = next(data_it)

                input_ids  = batch["ids"].to(device)      # [B, L]
                target_ids = batch["targets"].to(device)  # [B, L]

                # Build input/target (shift by 1 for NTP); clamp to model vocab
                inp = input_ids[:, :-1].clamp(0, cfg.vocab_size - 1)   # [B, L-1]
                tgt = target_ids[:, 1:].clamp(0, cfg.vocab_size - 1)   # [B, L-1]

                # Forward
                # Returns: logits_ntp, logits_mtp, aux_loss
                out = model(
                    inp,
                    next_token_ids=tgt,
                    return_loss=True,
                    return_memory=False,
                )

                logits_ntp, logits_mtp, aux_loss = out

                vocab    = logits_ntp.shape[-1]
                lm_loss  = criterion(logits_ntp.reshape(-1, vocab), tgt.reshape(-1))
                mtp_loss = criterion(logits_mtp.reshape(-1, vocab), tgt.reshape(-1)) if logits_mtp is not None else torch.tensor(0.0, device=device)

                # Scale loss by 1/accum_steps so gradients average (not sum) over microbatches
                total_loss = (lm_loss + 0.3 * mtp_loss + aux_loss) / accum_steps
                total_loss.backward()

                lm_loss_accum  += lm_loss.item()
                mtp_loss_accum += mtp_loss.item() if logits_mtp is not None else 0.0
                aux_val_accum  += aux_loss.item() if isinstance(aux_loss, torch.Tensor) else float(aux_loss)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0).item()
            optimizer.step()

            sync(device)
            dt_ms = (time.perf_counter() - t0) * 1000

            loss_val = lm_loss_accum  / accum_steps
            mtp_val  = mtp_loss_accum / accum_steps
            aux_val  = aux_val_accum  / accum_steps
            loss_ema = loss_val if loss_ema is None else (
                EMA_ALPHA * loss_val + (1 - EMA_ALPHA) * loss_ema
            )

            # Log
            row = {
                "step":        step,
                "loss":        round(loss_val, 5),
                "aux_loss":    round(aux_val, 5),
                "mtp_loss":    round(mtp_val, 5),
                "grad_norm":   round(grad_norm, 4),
                "lr":          lr,
                "step_ms":     round(dt_ms, 1),
                "loss_ema":    round(loss_ema, 5),
            }
            logger.log(row)

            # Checkpoint
            if step > 0 and step % args.ckpt == 0:
                save_checkpoint(model, optimizer, step, loss_val, lr)

    except KeyboardInterrupt:
        print(f"\n[TRAIN] ⚠️   Interrupted at step {step}")
        save_checkpoint(model, optimizer, step, loss_val, lr)

    finally:
        logger.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Steps     : {step + 1}")
    print(f"  Final loss: {loss_val:.4f}  (EMA: {loss_ema:.4f})")
    print(f"  Log files : training_log.jsonl  |  training_log.txt")
    print(f"{'='*60}\n")

    # Save final checkpoint
    save_checkpoint(model, optimizer, step + 1, loss_val, lr)


if __name__ == "__main__":
    main()
