#!/usr/bin/env python3
"""
new-fourier-test-new.py
- Training loop for Model3B (local config, ~450M params) with adaptive depth pondering.
- Features: k_exec / per-layer recurrence stats, dual-ascent λ_p controller,
            MTP loss, MoE aux loss, Dashboard Logging
- Hardware: Optimised for Mac M4 Pro 24 GB (MPS)

Architecture:
  - Model     : Model3B (LocalModelConfig) from models/recurrence_model_3b.py
  - Layers    : LightningDecoderLayer (GatedDeltaNet × 6 + GSA × 2) + ContinueHead
  - Embedding : Kronecker (D=8192) + learnable tok_embed residual (Fix 1)
  - Tokenizer : TSAI 131K (AutoTokenizer from ./tokenizer/)
  - Loss      : NTP + 0.3×MTP + aux_loss(MoE router) + ponder_loss

Changes vs new-fourier-test.py (SmolLM / Fourier):
  1. Model     : SmolLM (86M, MHA+SwiGLU) → Model3B/LocalModelConfig (~450M,
                 DeltaNet+GSA+MoE)
  2. Tokenizer : GPT2 (50k) → TSAI 131K
  3. Embedding : Fourier PFCodec → Kronecker + tok_embed residual
  4. Loss      : LM-only → LM + 0.3×MTP + aux + ponder
  5. Inline classes removed; all components live in models/recurrence_model_3b.py
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import argparse
import signal
import logging
import random
import gc
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from tqdm import tqdm
from datasets import load_from_disk
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer

# ── Kronecker embedding (replaces fourier_se_decoder) ────────────────────────
# Two sys.path entries needed:
#   1. depth-expansion/         → makes `models` a top-level importable package
#   2. depth-expansion/models/  → satisfies the absolute import
#      `from reversible_ops_midpoint import …` inside Model3B (unused here but
#      present in the module; the try/except catches it gracefully).
_script_dir = os.path.dirname(os.path.abspath(__file__))
_models_dir  = os.path.join(_script_dir, "models")
for _p in (_script_dir, _models_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.recurrence_model_3b import (
    KroneckerEmbeddings, KroneckerConfig, PureHybridEmbeddingTorch,
    Model3B, LocalModelConfig,
)

# --- Environment Setup ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"]  = "0.9"
os.environ["PYTORCH_MPS_PREFER_METAL"]         = "1"

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(1.0)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Pause/Resume Handler  (unchanged)
# =============================================================================

class PauseHandler:
    """Handles pause/resume via signals and file flag"""
    def __init__(self, pause_flag_path: str = "checkpoints/.pause"):
        self.pause_flag_path = Path(pause_flag_path)
        if self.pause_flag_path.parent:
            self.pause_flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.paused       = False
        self.in_pause_wait = False
        self.should_save  = False

        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        if self.in_pause_wait:
            print(f"\n[EXIT] Received signal {signum} while paused. Exiting...")
            raise KeyboardInterrupt
        print(f"\n[PAUSE] Received signal {signum}. Pausing after current step...")
        self.paused      = True
        self.should_save = True

    def check_pause_flag(self) -> bool: return self.pause_flag_path.exists()
    def clear_pause_flag(self):
        if self.pause_flag_path.exists(): self.pause_flag_path.unlink()
    def should_pause(self) -> bool: return self.paused or self.check_pause_flag()
    def resume(self):
        self.paused = False
        self.clear_pause_flag()


# =============================================================================
# Device Sync  (unchanged)
# =============================================================================

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


# =============================================================================
# SYNTHPromptSampler  (unchanged)
# =============================================================================

class SYNTHPromptSampler:
    """STRICT Sampler: Checks multiple locations for synth_local."""
    def __init__(self, dataset_name="PleIAs/SYNTH", local_path="synth_local",
                 tokenizer=None, seed=42):
        self.tokenizer = tokenizer
        self.seed      = seed

        cwd_path    = os.path.join(os.getcwd(), local_path)
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, local_path)
        parent_path = os.path.join(os.path.dirname(script_dir), local_path)

        if os.path.exists(cwd_path):     self.full_path = cwd_path
        elif os.path.exists(script_path): self.full_path = script_path
        elif os.path.exists(parent_path): self.full_path = parent_path
        else:
            print(f"[PROMPTS] ⚠️ Local dataset not found — trying HuggingFace stream for prompt pool...")
            self.dataset = self._load_hf_prompt_pool(dataset_name, n=500)
            return

        print(f"[PROMPTS] Initializing sampler from: {self.full_path}")
        try:
            self.dataset = load_from_disk(self.full_path)
            print(f"[PROMPTS] ✅ Loaded {len(self.dataset)} examples locally")
        except Exception as e:
            print(f"[PROMPTS] ❌ Failed to load local: {e} — trying HuggingFace stream...")
            self.dataset = self._load_hf_prompt_pool(dataset_name, n=500)

    def _load_hf_prompt_pool(self, dataset_name: str, n: int = 500):
        """Download a small pool of examples from HF streaming for prompt sampling."""
        try:
            from datasets import load_dataset as _load_hf
            stream = _load_hf(dataset_name, streaming=True, split="train")
            pool = []
            for ex in stream:
                if ex.get("language", "").lower() == "en" and ex.get("query", "").strip():
                    pool.append(ex)
                if len(pool) >= n:
                    break
            print(f"[PROMPTS] ✅ Loaded {len(pool)} examples from HuggingFace stream")
            return pool
        except Exception as e:
            print(f"[PROMPTS] ❌ HuggingFace stream also failed: {e}")
            return None

    def sample_token_ids(self, n: int = 5, step: int = 0) -> List[torch.Tensor]:
        if self.dataset is None:
            return []
        prompts_t   = []
        rng         = random.Random(self.seed + step)
        total_rows  = len(self.dataset)
        attempts    = 0

        while len(prompts_t) < n and attempts < n * 10:
            idx      = rng.randint(0, total_rows - 1)
            # Support both HF Dataset (dict-like indexing) and plain list
            raw = self.dataset[idx]
            ex  = dict(raw) if not isinstance(raw, dict) else raw
            attempts += 1
            lang     = ex.get("language", "")
            if lang is None or str(lang).lower() != "en":
                continue
            query = ex.get("query", "").strip()
            if not query:
                continue
            formatted = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded   = self.tokenizer(
                formatted, add_special_tokens=False, return_tensors=None,
                max_length=512, truncation=True, padding=False
            )
            prompts_t.append(torch.tensor(encoded["input_ids"], dtype=torch.long))
        return prompts_t


# =============================================================================
# Generation  (unchanged)
# =============================================================================

@torch.no_grad()
def sample_generate_single_fast(
    model: nn.Module,
    tokenizer,
    prompt_ids:     torch.Tensor,
    max_new_tokens: int   = 128,
    temperature:    float = 0.8,
    top_p:          float = 0.95,
    max_seq_len:    int   = 512,
    lambda_p:       float = 0.0,
    force_2:        bool  = False,
    step:           int   = 0,
    **kwargs,
) -> torch.Tensor:
    """
    Single-prompt autoregressive generation.
    Returns: (T_out,) LongTensor on same device.

    Model3B returns (logits_ntp, logits_mtp, aux_loss, memory_out, layer_stats)
    when use_ponder=True; output[0] is always logits_ntp.
    """
    device       = prompt_ids.device
    vocab_size   = model.lm_head.out_features
    max_valid_id = vocab_size - 1

    prompt_len = min(int(prompt_ids.size(0)), max_seq_len - 1)
    prompt_ids = prompt_ids[:prompt_len].clamp(0, max_valid_id)

    buf = torch.empty((1, max_seq_len), dtype=torch.long, device=device)
    buf[:, :prompt_len] = prompt_ids.unsqueeze(0)

    cur_len       = prompt_len
    max_total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for _ in range(max_new_tokens):
        if cur_len >= max_total_len:
            break
        output = model(
            buf[:, :cur_len],
            lambda_p=lambda_p,
            force_2=force_2,
            return_memory=False,
        )
        logits      = output[0] if isinstance(output, tuple) else output
        next_logits = logits[:, -1, :]

        if next_logits.size(-1) > vocab_size:
            next_logits = next_logits[:, :vocab_size]

        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)
        else:
            next_logits = next_logits / float(temperature)
            probs       = torch.softmax(next_logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf  = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[:, 0] = True
                filtered = sorted_probs * mask.float()
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)
                sampled_in_sorted = torch.multinomial(filtered, num_samples=1)
                next_id = sorted_idx.gather(-1, sampled_in_sorted).squeeze(-1)
            else:
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

        next_id = torch.clamp(next_id, min=0, max=max_valid_id)
        buf[0, cur_len] = next_id[0]
        cur_len += 1

    return buf[0, :cur_len]


# =============================================================================
# SYNTHStream  (dummy fallback vocab_size updated to 131072)
# =============================================================================

class SYNTHStream(IterableDataset):
    """STRICT Loader: Instant resume using Arrow slicing."""
    def __init__(self, tokenizer, dataset_name="PleIAs/SYNTH", local_path="synth_local",
                 seq_len=512, batch_size=16, shuffle_buffer=10000, seed=42,
                 include_query=True, include_reasoning=True, include_answer=True,
                 combine_separator="\n\n", filter_language="en", start_step=0):
        super().__init__()
        self.tokenizer          = tokenizer
        self.seq_len            = seq_len
        self.batch_size         = batch_size
        self.seed               = seed
        self.start_step         = start_step
        self.combine_separator  = combine_separator
        self.include_query      = include_query
        self.include_reasoning  = include_reasoning
        self.include_answer     = include_answer
        self.filter_language    = filter_language

        possible_paths = [
            os.path.join(os.getcwd(), local_path),
            os.path.join(os.getcwd(), "..", local_path),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), local_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), local_path)
        ]
        self.full_path = None
        for p in possible_paths:
            if os.path.exists(p):
                self.full_path = p
                break
        if not self.full_path:
            print(f"⚠️ Warning: Could not find '{local_path}'. Dataset loading might fail.")
            self.full_path = local_path
        else:
            print(f"📂 SYNTHStream loading from: {self.full_path}")

    def _construct_text(self, ex: Dict[str, Any]) -> Optional[str]:
        if self.filter_language:
            lang = ex.get("language")
            if not lang or (isinstance(lang, str) and lang.lower() != self.filter_language.lower()):
                return None
        # Use plain-text separators — no special tokens.
        # TSAI 131K maps <|im_start|>, <think>, etc. → IDs > 65535, which all
        # clamp to the same token (65535) and inject noise at every boundary.
        # Plain ASCII headers avoid this completely (matches train.py fix).
        parts = []
        query     = (ex.get("query")               or "").strip()
        reasoning = (ex.get("synthetic_reasoning") or "").strip()
        answer    = (ex.get("synthetic_answer")    or "").strip()
        if self.include_query and query:
            parts.append(f"Question: {query}")
        if self.include_reasoning and reasoning:
            parts.append(f"Thinking: {reasoning}")
        if self.include_answer and answer:
            parts.append(f"Answer: {answer}")
        return "\n\n".join(parts) if parts else None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        try:
            full_ds  = load_from_disk(self.full_path)
            full_ds  = full_ds.shuffle(seed=self.seed)
            skip_n   = self.start_step * self.batch_size
            if skip_n > 0 and skip_n < len(full_ds):
                print(f"⏩ Instant Jump: Slicing dataset from row {skip_n} to {len(full_ds)}")
                active_ds = full_ds.select(range(skip_n, len(full_ds)))
            else:
                active_ds = full_ds
            it = iter(active_ds)
        except Exception as e:
            print(f"❌ Critical Error loading dataset from disk: {e}")
            # Fallback: stream directly from HuggingFace instead of random tokens
            try:
                from datasets import load_dataset as _load_hf
                print("⚠️ synth_local not found — streaming PleIAs/SYNTH from HuggingFace...")
                hf_stream = _load_hf("PleIAs/SYNTH", streaming=True, split="train")
                hf_buf: List[int] = []
                for hf_ex in hf_stream:
                    hf_text = self._construct_text(hf_ex)
                    if not hf_text:
                        continue
                    hf_enc = self.tokenizer(
                        hf_text, add_special_tokens=False, return_tensors=None,
                        max_length=self.seq_len * 2, truncation=True, padding=False,
                    )
                    hf_ids = hf_enc["input_ids"]
                    if not hf_ids:
                        continue
                    hf_buf.extend(hf_ids)
                    while len(hf_buf) >= self.seq_len:
                        block = hf_buf[:self.seq_len]
                        del hf_buf[:self.seq_len]
                        yield {"ids":     torch.tensor(block, dtype=torch.long),
                               "targets": torch.tensor(block, dtype=torch.long)}
                return
            except Exception as e2:
                print(f"❌ HuggingFace streaming also failed: {e2}")
                print("⚠️ Switching to dummy data generator (vocab_size=65536)")
                while True:
                    yield {"ids":     torch.randint(0, 65536, (self.seq_len,), dtype=torch.long),
                           "targets": torch.randint(0, 65536, (self.seq_len,), dtype=torch.long)}

        buf: List[int] = []
        while True:
            while len(buf) < self.seq_len:
                try:
                    ex = next(it)
                except StopIteration:
                    print("🔄 Dataset finished, restarting...")
                    it = iter(full_ds)
                    ex = next(it)
                text = self._construct_text(ex)
                if not text:
                    continue
                encoded = self.tokenizer(
                    text, add_special_tokens=False, return_tensors=None,
                    max_length=self.seq_len * 2, truncation=True, padding=False,
                )
                ids = encoded["input_ids"]
                if not ids:
                    continue
                buf.extend(ids)
                if len(buf) > 4 * self.seq_len:
                    buf[:] = buf[-(4 * self.seq_len):]

            block = buf[:self.seq_len]
            del buf[:self.seq_len]
            yield {"ids":     torch.tensor(block, dtype=torch.long),
                   "targets": torch.tensor(block, dtype=torch.long)}


# =============================================================================
# (Inline model components removed — now using Model3B from
#  models/recurrence_model_3b.py which contains LightningDecoderLayer +
#  ContinueHead + LocalModelConfig)
# =============================================================================

# =============================================================================
# Checkpoint Helpers  (unchanged except default embedding_type → "kronecker")
# =============================================================================

def save_checkpoint(model, optimizer, step, loss,
                    embedding_type="kronecker", save_dir="checkpoints",
                    lambda_p_state=None):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        'step':               step,
        'model_state_dict':   model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss':               loss,
        'embedding_type':     embedding_type,
        'lambda_p_state':     lambda_p_state,
    }
    latest_path = os.path.join(save_dir, f'{embedding_type}_latest.pt')
    tmp_path    = latest_path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, latest_path)   # atomic replace — no partial reads
    lp_str = f"{lambda_p_state:.2e}" if lambda_p_state is not None else "N/A"
    print(f"💾 Checkpoint saved: {latest_path} (step={step}, λ_p={lp_str})")


def load_checkpoint(model, optimizer,
                    embedding_type="kronecker", save_dir="checkpoints",
                    checkpoint_path=None):
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return 0, 0.0, None
        latest_path = checkpoint_path
    else:
        latest_path = os.path.join(save_dir, f'{embedding_type}_latest.pt')

    if not os.path.exists(latest_path):
        return 0, 0.0, None

    try:
        print(f"🔄 Loading checkpoint: {latest_path}")
        checkpoint   = torch.load(latest_path, map_location='cpu')
        missing, _   = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"[RESUME] Warning: Missing keys: {missing[:5]}...")
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step             = checkpoint.get('step', 0)
        loss             = checkpoint.get('loss', 0.0)
        lambda_p_state   = checkpoint.get('lambda_p_state', None)
        print(f"✅ Loaded checkpoint from {latest_path} at step {step}")
        if lambda_p_state is not None:
            print(f"   📊 Restored λ_p = {lambda_p_state:.2e}")
        return step, loss, lambda_p_state
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 0, 0.0, None


# =============================================================================
# Tee  — mirror all print() output to a log file automatically
# =============================================================================

class Tee:
    """
    Wraps sys.stdout so that every print() call goes to both the terminal
    and the log file.  Used via:  sys.stdout = Tee(sys.stdout, open(log_path, mode))
    """
    def __init__(self, stdout, log_file):
        self._stdout   = stdout
        self._log_file = log_file

    def write(self, data: str):
        self._stdout.write(data)
        self._log_file.write(data)

    def flush(self):
        self._stdout.flush()
        self._log_file.flush()

    # Needed so tools that inspect sys.stdout.fileno() (e.g. tqdm) don't crash
    def fileno(self):
        return self._stdout.fileno()

    def isatty(self):
        try:   return self._stdout.isatty()
        except Exception: return False


# =============================================================================
# Main  (MODIFIED: tokenizer + Kronecker setup; model params for M4 Pro)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ouro-Spec Training (Kronecker + TSAI 131K)")
    parser.add_argument("--hard_kill",  action="store_true",  help="Disable recurrence")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to checkpoint to resume from (or 'latest')")
    parser.add_argument("--set_lambda", type=float, default=None,
                        help="Manually set/override compute price λ_p")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for determinism")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pause_handler = PauseHandler()
    if pause_handler.check_pause_flag():
        print("⚠️ Pause flag found. Clearing for new run (or resume will handle via args).")
        pause_handler.resume()

    # ── Device ───────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"🔥 Using device: {device.type}")

    # ── Tokenizer: TSAI 131K ─────────────────────────────────────────────────
    tokenizer_path = os.path.join(_script_dir, "tokenizer")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"TSAI 131K tokenizer not found at: {tokenizer_path}\n"
            "Expected: depth-expansion/tokenizer/ with tokenizer.json etc."
        )
    print(f"📖 Loading TSAI 131K tokenizer from: {tokenizer_path}")
    tokenizer  = AutoTokenizer.from_pretrained(tokenizer_path)
    vocab_size = len(tokenizer)   # 131075 (131072 vocab + 3 special tokens)
    print(f"   Vocab size: {tokenizer.vocab_size:,}  |  with specials: {vocab_size:,}")

    # ── Model config (needed before KP table build) ───────────────────────────
    cfg = LocalModelConfig()

    # ── Kronecker Embeddings (replaces PFCodec / discover_chars) ─────────────
    # Byte-level: 256 bytes × 32 positions = D=8192 dimensional vectors.
    kron_cfg = KroneckerConfig(CHAR_DIM=256, POS_DIM=32, D=8192)
    pf_codec = KroneckerEmbeddings(kron_cfg)

    # Build KP table for model vocab only (cfg.vocab_size=65536).
    # TSAI tokenizer has 131075 entries but the model embedding/lm_head are
    # sized to cfg.vocab_size; IDs above that are clamped before the forward pass.
    print("🔄 Building BPE vocab list for Kronecker embeddings...")
    bpe_vocab = []
    for i in tqdm(range(cfg.vocab_size), desc="Vocab"):
        try:
            bpe_vocab.append(tokenizer.decode([i]))
        except Exception:
            bpe_vocab.append(f"<TOKEN_{i}>")

    # ── Model (LocalModelConfig: ~450M params, fits M4 Pro 24 GB) ────────────
    print("🧠 Initializing Model3B (local config)...")
    cfg   = LocalModelConfig()
    model = Model3B(
        cfg,
        embedding_type="kronecker",
        bpe_vocab=bpe_vocab,
        pf_codec=pf_codec,
        use_ponder=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=False)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"   Params: {total_p:,} ({total_p/1e6:.1f}M)")
    print(f"   Est. memory: weights={total_p*2/1e9:.2f}GB (bf16)  "
          f"Adam={total_p*8/1e9:.2f}GB  KP-table={cfg.vocab_size*8192*2/1e9:.2f}GB")

    # ── Checkpoint Resume ─────────────────────────────────────────────────────
    start_step     = 0
    save_dir       = "checkpoints"
    checkpoint_path = None
    loaded_lambda_p = None

    if not args.resume:
        potential_latest = os.path.join(save_dir, 'kronecker_latest.pt')
        if os.path.exists(potential_latest):
            print(f"\n[AUTO-RESUME] 🔄 Found existing checkpoint at {potential_latest}")
            print("[AUTO-RESUME] 🔄 Automatically resuming to prevent data loss.")
            args.resume = "latest"

    if args.resume:
        if args.resume != "latest":
            checkpoint_path = args.resume
        start_step, start_loss, loaded_lambda_p = load_checkpoint(
            model, optimizer, embedding_type="kronecker",
            save_dir=save_dir, checkpoint_path=checkpoint_path
        )
        if start_step > 0:
            print(f"⏩ Resuming from step {start_step} (loss: {start_loss:.4f})")

    # ── Data Loader ───────────────────────────────────────────────────────────
    print(f"📂 Initializing Data Loader (Start Step: {start_step})...")
    dataset = SYNTHStream(
        tokenizer, seq_len=512, batch_size=4,
        seed=args.seed, start_step=start_step
    )
    # num_workers=0: required on MPS (forked workers crash with Metal)
    train_loader   = DataLoader(dataset, batch_size=4, num_workers=0)
    train_iterator = iter(train_loader)

    prompt_sampler = SYNTHPromptSampler(
        dataset_name="PleIAs/SYNTH",
        tokenizer=tokenizer,
        seed=args.seed
    )

    gen_every        = 500
    gen_warmup_steps = 100

    # ── LR schedule parameters (10 k steps) ──────────────────────────────────
    target_total_steps = 10000
    warmup_steps       = 500
    max_lr_val         = 3.0e-4
    min_lr_val         = 3.0e-5

    # ── Persistent Logs ───────────────────────────────────────────────────────
    # Tee stdout → everything printed goes to both terminal AND log file.
    log_name  = "log_kronecker_adaptive.txt"
    file_mode = "a" if start_step > 0 else "w"
    _log_fh   = open(log_name, file_mode, buffering=1, encoding="utf-8")
    sys.stdout = Tee(sys.stdout, _log_fh)

    if start_step == 0:
        print(f"INIT | PARAMS: {total_p} | EMB: Kronecker(D=8192)+tok_embed | "
              f"D={cfg.hidden_size} L={cfg.num_layers} H={cfg.delta_v_heads} "
              f"EXPERTS={cfg.num_real_experts}r+{cfg.num_null_experts}n | PONDER=ON")

    criterion = nn.CrossEntropyLoss()

    # ── DUAL-ASCENT COMPUTE CONTROLLER (unchanged) ────────────────────────────
    TARGET_K = 2.4
    DUAL_LR  = 5e-5
    LAMBDA_MIN = 1e-4
    LAMBDA_MAX = 5e-2
    AUDIT_PROB = 0.05

    DEADBAND_TRIGGER_LAM    = 0.50
    DEADBAND_TRIGGER_WINDOW = 100
    DEADBAND_HIGH_PCT       = 0.05
    DEADBAND_LOW_PCT        = 0.02
    DEADBAND_LR_SCALE       = 0.5

    deadband_active               = False
    deadband_steps_above_threshold = 0
    deadband_trigger_step          = None

    lambda_p_state = 1e-4
    if loaded_lambda_p is not None:
        lambda_p_state = loaded_lambda_p
        print(f"📈 Restored compute price λ_p = {lambda_p_state:.2e}")
    if args.set_lambda is not None:
        lambda_p_state = float(args.set_lambda)
        print(f"🛠️ CLI Override: Setting compute price λ_p = {lambda_p_state:.2e}")

    credit_ema  = 0.0
    CREDIT_BETA = 0.995

    print("🚀 Starting Training Loop...")
    pbar      = range(start_step, target_total_steps)
    ckpt_every = 250

    # ── Training Loop (UNCHANGED from new-fourier-test.py) ───────────────────
    for step in pbar:

        # Check Pause
        if pause_handler.should_pause():
            print(f"\n[PAUSE] Saving checkpoint at step {step} and exiting...")
            loss_val = lm_loss.item() if 'lm_loss' in locals() else 0.0
            save_checkpoint(model, optimizer, step, loss_val,
                            "kronecker", save_dir, lambda_p_state=lambda_p_state)
            return

        # Periodic Checkpoint
        if step > 0 and step % ckpt_every == 0:
            loss_val = lm_loss.item() if 'lm_loss' in locals() else 0.0
            save_checkpoint(model, optimizer, step, loss_val,
                            "kronecker", save_dir, lambda_p_state=lambda_p_state)

        # LR Schedule (wrapped cosine, no notch)
        warmup_steps       = 500
        max_lr_val, min_lr_val = 3e-4, 3e-5
        if step < warmup_steps:
            lr = max_lr_val * step / warmup_steps
        else:
            decay_start = warmup_steps
            progress    = (step - decay_start) / (target_total_steps - decay_start)
            progress    = min(max(progress, 0.0), 1.0)
            lr = min_lr_val + 0.5 * (max_lr_val - min_lr_val) * (1 + math.cos(math.pi * progress))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Ponder & Force Schedule
        force_2_sched = (step < 800)

        PONDER_START = 1200
        HALT_THR     = 0.10
        REWARD_ETA   = 2.0

        audit_force_2     = (step >= PONDER_START) and (random.random() < AUDIT_PROB)
        force_2_effective = force_2_sched or audit_force_2
        lambda_p          = lambda_p_state if step >= PONDER_START else 0.0

        # Forward & Loss
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        input_ids  = batch['ids'].to(device)
        target_ids = batch['targets'].to(device)

        sync_device(device)
        t0 = time.time()

        B, Seq  = input_ids.shape
        # Clamp to model vocab (TSAI 131K has IDs up to 131074; model uses 65536)
        inp_seq = input_ids[:, :-1].clamp(0, cfg.vocab_size - 1)
        tgt_seq = target_ids[:, 1:].clamp(0, cfg.vocab_size - 1)

        do_credit      = (step >= PONDER_START)
        do_diagnostics = (step % 10 == 0)

        # Model3B ponder forward
        # Returns: logits_ntp, logits_mtp, aux_loss, memory_stream_out, layer_stats
        logits_ntp, logits_mtp, aux_loss, _, stats = model(
            inp_seq,
            next_token_ids=tgt_seq,
            lambda_p=lambda_p,
            force_2=force_2_effective,
            halt_thr=HALT_THR,
            capture_h1=do_credit,
            return_loss=True,
            return_memory=False,
        )

        ntp_vocab = logits_ntp.shape[-1]   # cfg.vocab_size (65536)
        lm_loss  = criterion(logits_ntp.reshape(-1, ntp_vocab), tgt_seq.reshape(-1))
        mtp_loss = (criterion(logits_mtp.reshape(-1, ntp_vocab), tgt_seq.reshape(-1))
                    if logits_mtp is not None else torch.tensor(0.0, device=device))

        # Recurrence Diagnostics — compare k=1 hidden state to final
        d_ent_global = 0.0
        d_nll_global = 0.0
        delta_h      = 0.0
        nll_1_val    = None

        if (do_diagnostics or do_credit) and stats:
            # Collect h_1 tensors from first recurrence step across all layers
            h1_tensors = [s.get('h_1') for s in stats if s.get('h_1') is not None]
            if h1_tensors:
                # Use the last layer's h_1 as the "k=1 only" logit estimate
                h1_last  = h1_tensors[-1]          # [B, T, D]
                logits_1 = model.lm_head(model.norm(h1_last))
                with torch.no_grad():
                    prob_1  = F.softmax(logits_1, dim=-1)
                    ent_1   = -(prob_1 * torch.log(prob_1 + 1e-9)).sum(dim=-1).mean()
                    prob_2  = F.softmax(logits_ntp, dim=-1)
                    ent_2   = -(prob_2 * torch.log(prob_2 + 1e-9)).sum(dim=-1).mean()
                    d_ent_global = (ent_2 - ent_1).item()
                    delta_h      = d_ent_global
                    nll_1        = criterion(logits_1.reshape(-1, ntp_vocab), tgt_seq.reshape(-1))
                    d_nll_global = (lm_loss - nll_1).item()
                    nll_1_val    = nll_1
                    del prob_1, prob_2, ent_1, ent_2, nll_1, logits_1

        # Economic Fix: Ponder Credit (The Coupon)
        total_ponder_loss = sum([s['p_loss'] for s in stats]) if stats else torch.tensor(0.0, device=device)
        delta_ce = torch.tensor(0.0, device=device)
        if nll_1_val is not None:
            delta_ce = (nll_1_val - lm_loss).clamp(min=0)
        total_ponder_loss = torch.relu(total_ponder_loss - REWARD_ETA * delta_ce)

        total_loss = lm_loss + 0.3 * mtp_loss + aux_loss + total_ponder_loss
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Dual-Ascent Controller Update
        if step >= PONDER_START and not audit_force_2:
            policy_avg_k  = sum([s['k'] for s in stats]) / len(stats) if stats else 1.0
            lam_normalized = (lambda_p_state - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)

            if not deadband_active:
                if lam_normalized >= DEADBAND_TRIGGER_LAM:
                    deadband_steps_above_threshold += 1
                else:
                    deadband_steps_above_threshold = max(0, deadband_steps_above_threshold - 1)
                if deadband_steps_above_threshold >= DEADBAND_TRIGGER_WINDOW:
                    deadband_active      = True
                    deadband_trigger_step = step
                    print(f"\n🔒 [DEADBAND ENGAGED] λ_p={lambda_p_state:.2e} stable for {DEADBAND_TRIGGER_WINDOW} steps")
                    print(f"   Asymmetric bounds: TARGET_K={TARGET_K:.2f} ± [{-DEADBAND_LOW_PCT*100:.1f}%, +{DEADBAND_HIGH_PCT*100:.1f}%]")

            if deadband_active:
                k_high = TARGET_K * (1.0 + DEADBAND_HIGH_PCT)
                k_low  = TARGET_K * (1.0 - DEADBAND_LOW_PCT)
                if policy_avg_k > k_high:
                    lambda_p_state += DUAL_LR * DEADBAND_LR_SCALE * (policy_avg_k - k_high)
                elif policy_avg_k < k_low:
                    lambda_p_state += DUAL_LR * DEADBAND_LR_SCALE * (policy_avg_k - k_low)
            else:
                lambda_p_state += DUAL_LR * (policy_avg_k - TARGET_K)
            lambda_p_state = float(max(LAMBDA_MIN, min(LAMBDA_MAX, lambda_p_state)))

        if delta_ce.item() > 0:
            credit_ema = CREDIT_BETA * credit_ema + (1 - CREDIT_BETA) * delta_ce.item()

        sync_device(device)
        dt = (time.time() - t0) * 1000.0
        tokens_per_sec = inp_seq.numel() / (dt / 1000.0)

        # ── Telemetry Log (UNCHANGED) ─────────────────────────────────────────
        if step % 1 == 0:
            p_loss_val    = total_ponder_loss.item()
            grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            loss_val      = total_loss.item()
            lm_val        = lm_loss.item()
            reward_val    = (REWARD_ETA * delta_ce).item() if delta_ce.item() > 0 else 0.0

            audit_flag = "[AUDIT] " if audit_force_2 else ""
            policy_k   = sum([s['k'] for s in stats]) / len(stats) if stats else 1.0
            ctrl_err   = policy_k - TARGET_K
            p_near_avg = sum([s.get('p_near_clamp', 0.0) for s in stats]) / len(stats) if stats else 0.0
            raw_ponder = sum([s['p_loss'].item() if hasattr(s['p_loss'], 'item') else s['p_loss']
                              for s in stats])

            mtp_val = mtp_loss.item() if isinstance(mtp_loss, torch.Tensor) else 0.0
            aux_val = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0
            log_line = (f"STEP: {step} {audit_flag}| TOT: {loss_val:.4f} | LM: {lm_val:.4f} | "
                        f"PONDER: {p_loss_val:.6f} (RAW: {raw_ponder:.6f}, RWD: {reward_val:.6f}) | "
                        f"LR: {lr:.4e} | GRAD: {grad_norm_val:.4f} | "
                        f"λ_p: {lambda_p_state:.2e}{'[DB]' if deadband_active else ''} | "
                        f"CTRL_ERR: {ctrl_err:+.3f} | P_NR: {p_near_avg:.1%} | "
                        f"MTP: {mtp_val:.4f} | AUX: {aux_val:.4f}")

            # (baton_rat / lam_mu were SmolLM-specific — not available in Model3B)

            if stats:
                k_execs  = [s.get('k_exec', 0) for s in stats]
                ks       = [s.get('k', 0)      for s in stats]
                avg_exec = sum(k_execs) / len(k_execs) if k_execs else 0
                avg_k    = sum(ks)      / len(ks)      if ks      else 0
                avg_gap  = avg_exec - avg_k
                pl_ratio = (p_loss_val / lm_val) * 100 if lm_val > 0 else 0.0
                log_line += f" | PL/LM: {pl_ratio:.2f}% | AVGS: exec={avg_exec:.2f}, k={avg_k:.2f}, gap={avg_gap:.2f}"

            log_line += f" | dt: {dt:7.2f}ms | tok/sec: {tokens_per_sec:9.2f}"
            print(log_line)   # Tee → terminal + log file

            if step % 10 == 0:
                for i, s in enumerate(stats):
                    base = (f" L{i}: k_exec={s.get('k_exec',0)}, "
                            f"expected_k={s.get('k',0):.2f}, "
                            f"s_fin={s.get('s_final',0):.3f}")

                    avg_et   = s.get('avg_exec_token', 0)
                    p95_et   = s.get('p95_exec_token', 0)
                    frac_ge2 = s.get('frac_ge2', 0)
                    frac_ge3 = s.get('frac_ge3', 0)
                    frac_ge4 = s.get('frac_ge4', 0)
                    token_exec_ext = (f" | tok: avg={avg_et:.2f}, p95={p95_et:.2f}"
                                      f" | tail: ≥2:{frac_ge2:.2%}, ≥3:{frac_ge3:.2%}, ≥4:{frac_ge4:.2%}")

                    h_diff     = s.get('h_diff', 0.0)
                    p_diff     = s.get('p_diff', 0.0)
                    ponder_sig = s.get('ponder_signal', 0)

                    ext      = (f" | ent={s.get('ent',0):.4f}"
                                f" | p_mu={s.get('p_mean',0):.2f}, p_std={s.get('p_std',0):.3f},"
                                f" d_rel={s.get('d_rel',0):.3f}")
                    rec_ext    = f" | Δh={h_diff:.3f}, Δp={p_diff:.2f}"
                    ponder_ext = f" | ponder={ponder_sig:.3f}, λ_p={lambda_p:.2e}"
                    print(base + token_exec_ext + ext + rec_ext + ponder_ext)

                    if step % 50 == 0:
                        s_trace_list = s.get('s_trace', [])
                        s_trace_str  = "[" + ", ".join([f"{x:.2f}" for x in s_trace_list]) + "]"
                        print(f"  L{i} s_trace: {s_trace_str}")

            if step % 50 == 0:
                phase = "warmup" if step < 500 else ("done" if step > 10000 else "cosine")
                print(f"LR CHECK: ({step}, {lr:.2e}, phase={phase})")

            if do_credit and nll_1_val is not None:
                ce_gain = (nll_1_val - lm_loss).item()
                rec_quality_msg = (f"  RECURRENCE QUALITY: ΔCE={ce_gain:.6f}"
                                   f" | ΔH={delta_h:.4f}"
                                   f" | COUPON: {reward_val:.6f}")
                if audit_force_2:
                    status = "[AUDIT-POS]" if ce_gain > 0 else "[AUDIT-NEG]"
                else:
                    status = "[ECONOMIC WIN]" if ce_gain > 0 else "[TAX ONLY]"
                conf = " [CONFIDENCE+]" if (ce_gain > 0 and delta_h < 0 and not audit_force_2) else ""
                print(rec_quality_msg + f" {status}{conf}")

            # Dashboard JSON
            try:
                serializable_stats = []
                for s in stats:
                    s_clean = {k: v.item() if isinstance(v, torch.Tensor) else v
                               for k, v in s.items() if k != 'h_1'}
                    serializable_stats.append(s_clean)
                with open("stats.json", "w") as f:
                    json.dump({
                        "step":         step,
                        "lm_loss":      lm_loss.detach().item(),
                        "mtp_loss":     mtp_loss.detach().item() if isinstance(mtp_loss, torch.Tensor) else 0.0,
                        "aux_loss":     aux_loss.detach().item() if isinstance(aux_loss, torch.Tensor) else 0.0,
                        "total_ponder": total_ponder_loss.detach().item(),
                        "delta_ce":     delta_ce.detach().item() if isinstance(delta_ce, torch.Tensor) else 0.0,
                        "delta_h":      delta_h,
                        "deadband_active": deadband_active,
                        "deadband_steps_since_trigger":
                            (step - deadband_trigger_step) if deadband_trigger_step else 0,
                        "layers": serializable_stats
                    }, f)
                del serializable_stats
            except Exception as e:
                print(f"JSON Dump failed: {e}")

        # Memory Cleanup (critical on MPS unified memory)
        del logits_ntp, logits_mtp, aux_loss, total_loss, lm_loss, mtp_loss, \
            total_ponder_loss, delta_ce, \
            batch, input_ids, target_ids, stats, inp_seq, tgt_seq, grad_norm
        if 'nll_1_val' in locals() and nll_1_val is not None:
            del nll_1_val
        if 'h1_tensors' in locals() and h1_tensors:
            del h1_tensors

        if device.type == "mps":
            try: torch.mps.empty_cache()
            except Exception: pass
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        # Generation
        if gen_every > 0 and step >= gen_warmup_steps and step % gen_every == 0:
            model.eval()
            try:
                prompts_t = prompt_sampler.sample_token_ids(n=3, step=step)
                prompts_t = [p.to(device) for p in prompts_t]

                if prompts_t:
                    print(f"\n{'='*20} GENERATION STEP {step} {'='*20}")

                    for gi, prompt_t in enumerate(prompts_t):
                        gen_t = sample_generate_single_fast(
                            model=model, tokenizer=tokenizer,
                            prompt_ids=prompt_t,
                            max_new_tokens=128, temperature=0.8, top_p=0.95,
                            max_seq_len=512,
                            force_2=force_2_effective, step=step, lambda_p=lambda_p
                        )
                        prompt_text = tokenizer.decode(prompt_t.tolist(), skip_special_tokens=False)
                        new_tokens  = gen_t[prompt_t.size(0):]
                        new_text    = (tokenizer.decode(new_tokens.tolist(), skip_special_tokens=False)
                                       if new_tokens.numel() > 0 else "")

                        log_msg = f"[GEN {gi+1}]\n  Prompt ({prompt_t.size(0)} tokens):\n"
                        for line in (prompt_text[:500] + ("..." if len(prompt_text) > 500 else "")).split('\n'):
                            log_msg += f"    {line}\n"
                        log_msg += f"  Output ({new_tokens.numel()} tokens):\n"
                        for line in (new_text[:1000] + ("..." if len(new_text) > 1000 else "")).split('\n'):
                            log_msg += f"    {line}\n"
                        print(log_msg)   # Tee → terminal + log file

                    print(f"{'='*60}\n")
                    del prompts_t

                gc.collect()
                if device.type == "mps":
                    try: torch.mps.empty_cache()
                    except Exception: pass

            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback; traceback.print_exc()
            finally:
                model.train()


if __name__ == "__main__":
    main()
