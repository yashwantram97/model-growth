#!/usr/bin/env python3
"""
new-fourier-test.py
- Complete Ouro-Spec Training Loop
- Features: 3-Phase Warmup, Static KV, Baton Injection, Dashboard Logging
- Hardware: Optimized for Mac M1 (MPS)
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
from transformers import GPT2Tokenizer

# --- Environment Setup ---
# MPS memory management - critical for Mac M1
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "1.0"
os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = "0.9"
os.environ["PYTORCH_MPS_PREFER_METAL"] = "1"

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    torch.mps.set_per_process_memory_fraction(1.0)
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Fourier components
try:
    from fourier_se_decoder import PFConfig, PFCodec, HybridEmbeddingTorch
    from new_fourier_test_components import MultiheadLatentAttention, MoEFFN
except ImportError:
    # Quick fix for import paths if running from different directory
    sys.path.append(os.getcwd())
    from fourier_se_decoder import PFConfig, PFCodec, HybridEmbeddingTorch
    from new_fourier_test_components import MultiheadLatentAttention, MoEFFN

# -----------------------------
# Pause/Resume Handler
# -----------------------------

class PauseHandler:
    """Handles pause/resume via signals and file flag"""
    def __init__(self, pause_flag_path: str = "checkpoints/.pause"):
        self.pause_flag_path = Path(pause_flag_path)
        if self.pause_flag_path.parent:
            self.pause_flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.paused = False
        self.in_pause_wait = False
        self.should_save = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        if self.in_pause_wait:
            print(f"\n[EXIT] Received signal {signum} while paused. Exiting...")
            raise KeyboardInterrupt
        
        print(f"\n[PAUSE] Received signal {signum}. Pausing after current step...")
        self.paused = True
        self.should_save = True
        
    def check_pause_flag(self) -> bool:
        return self.pause_flag_path.exists()
    
    def clear_pause_flag(self):
        if self.pause_flag_path.exists():
            self.pause_flag_path.unlink()
    
    def should_pause(self) -> bool:
        return self.paused or self.check_pause_flag()
    
    def resume(self):
        self.paused = False
        self.clear_pause_flag()

# -----------------------------
# Data Loading Components (Ported from FC4)
# -----------------------------

def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except:
            pass

def discover_chars_from_bpe_tokenizer(tokenizer, vocab_size=50272):
    """
    Extract all unique characters from the BPE tokenizer's vocabulary.
    This ensures we can handle any BPE token that appears in the dataset.
    """
    print("🔍 Discovering characters from BPE tokenizer...")
    all_chars = set()
    
    for token_id in tqdm(range(min(vocab_size, len(tokenizer))), desc="Extracting chars"):
        try:
            token_text = tokenizer.decode([token_id])
            all_chars.update(token_text)
        except Exception as e:
            continue
    
    chars_list = sorted(list(all_chars))
    char_to_id = {ch: i for i, ch in enumerate(chars_list)}
    
    print(f"📝 Found {len(chars_list)} unique characters in BPE vocabulary")
    print(f"📝 Sample characters: {chars_list[:20]}...")
    
    return chars_list, char_to_id

# -----------------------------
# Generation Components
# -----------------------------

class SYNTHPromptSampler:
    """
    STRICT Sampler: Checks multiple locations for synth_local.
    """
    def __init__(self, dataset_name="PleIAs/SYNTH", local_path="synth_local", tokenizer=None, seed=42):
        self.tokenizer = tokenizer
        self.seed = seed
        
        # --- SMART PATH FINDER ---
        cwd_path = os.path.join(os.getcwd(), local_path)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, local_path)
        parent_path = os.path.join(os.path.dirname(script_dir), local_path)

        if os.path.exists(cwd_path):
            self.full_path = cwd_path
        elif os.path.exists(script_path):
            self.full_path = script_path
        elif os.path.exists(parent_path):
            self.full_path = parent_path
        else:
            print(f"[PROMPTS] ⚠️ Local dataset not found in CWD, Script Dir, or Parent Dir.")
            self.dataset = None
            return

        print(f"[PROMPTS] Initializing sampler from: {self.full_path}")
        
        try:
            self.dataset = load_from_disk(self.full_path)
            print(f"[PROMPTS] ✅ Loaded {len(self.dataset)} examples locally")
        except Exception as e:
            print(f"[PROMPTS] ❌ Failed to load local: {e}")
            self.dataset = None

    def sample_token_ids(self, n: int = 5, step: int = 0) -> List[torch.Tensor]:
        if self.dataset is None:
            return []
        
        prompts_t = []
        rng = random.Random(self.seed + step)
        total_rows = len(self.dataset)
        attempts = 0
        
        while len(prompts_t) < n and attempts < n * 10:
            idx = rng.randint(0, total_rows - 1)
            ex = self.dataset[idx]
            attempts += 1
            
            lang = ex.get("language", "")
            if lang is None or str(lang).lower() != "en":
                continue
            
            query = ex.get("query", "").strip()
            if not query: continue
            
            formatted = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
            encoded = self.tokenizer.encode_plus(
                formatted, add_special_tokens=False, return_tensors=None,
                max_length=512, truncation=True, padding=False
            )
            prompts_t.append(torch.tensor(encoded["input_ids"], dtype=torch.long))
                    
        return prompts_t

@torch.no_grad()
def sample_generate_single_fast(
    model: nn.Module,
    tokenizer,
    prompt_ids: torch.Tensor,          # (T,)
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    **kwargs
) -> torch.Tensor:
    """
    Single-prompt generation with preallocated token buffer (no torch.cat per token).
    Returns: (T_out,) LongTensor on same device.
    Assumes model(input_ids) -> logits of shape (B, T, V).
    """
    device = prompt_ids.device
    # Fix: SmolLM doesn't save vocab_size, but lm_head does
    vocab_size = model.lm_head.out_features
    max_valid_id = vocab_size - 1

    # Truncate prompt to fit (leave room for at least 1 generated token, same as your old logic)
    prompt_len = min(int(prompt_ids.size(0)), max_seq_len - 1)
    prompt_ids = prompt_ids[:prompt_len]

    # Preallocate full buffer and copy prompt in
    # Shape (1, max_seq_len) so we can slice [:, :cur_len] each step
    buf = torch.empty((1, max_seq_len), dtype=torch.long, device=device)
    buf[:, :prompt_len] = prompt_ids.unsqueeze(0)

    cur_len = prompt_len
    max_total_len = min(max_seq_len, prompt_len + max_new_tokens)

    for _ in range(max_new_tokens):
        if cur_len >= max_total_len:
            break

        # Forward only on the filled prefix
        output = model(buf[:, :cur_len], **kwargs)          # (1, cur_len, V) or tuple
        # Handle model returning (logits, stats) tuple
        logits = output[0] if isinstance(output, tuple) else output
        next_logits = logits[:, -1, :]            # (1, V)

        # Defensive: truncate if model outputs extra dims
        if next_logits.size(-1) > vocab_size:
            next_logits = next_logits[:, :vocab_size]

        if temperature <= 0:
            next_id = torch.argmax(next_logits, dim=-1)  # (1,)
        else:
            next_logits = next_logits / float(temperature)
            probs = torch.softmax(next_logits, dim=-1)   # (1, V)

            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[:, 0] = True

                # Zero out tokens outside nucleus and renormalize
                filtered = sorted_probs * mask.float()
                filtered = filtered / filtered.sum(dim=-1, keepdim=True)

                # Sample in sorted space, then map back to vocab ids
                sampled_in_sorted = torch.multinomial(filtered, num_samples=1)  # (1,1)
                next_id = sorted_idx.gather(-1, sampled_in_sorted).squeeze(-1)  # (1,)
            else:
                next_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)

        # Defensive clamp
        next_id = torch.clamp(next_id, min=0, max=max_valid_id)

        # Write next token into buffer
        buf[0, cur_len] = next_id[0]
        cur_len += 1

    return buf[0, :cur_len]
    
    for token_id in tqdm(range(min(vocab_size, len(tokenizer))), desc="Extracting chars"):
        try:
            token_text = tokenizer.decode([token_id])
            all_chars.update(token_text)
        except Exception as e:
            continue
    
    chars_list = sorted(list(all_chars))
    char_to_id = {ch: i for i, ch in enumerate(chars_list)}
    return chars_list, char_to_id

def pad_char_vocab_128(chars):
    """Pad character vocabulary to exactly 128 chars"""
    base = [chr(i) for i in range(32, 127)]
    for ch in base:
        if len(chars) >= 128:
            break
        if ch not in chars:
            chars.append(ch)
    
    chars = chars[:128]
    
    seen = set()
    uniq = []
    for ch in chars:
        if ch not in seen:
            uniq.append(ch)
            seen.add(ch)
    
    i = 0
    while len(uniq) < 128:
        placeholder = f'¤{i}'
        if placeholder not in seen:
            uniq.append(placeholder)
            seen.add(placeholder)
        i += 1
    
    char_to_id = {ch: i for i, ch in enumerate(uniq)}
    return uniq, char_to_id

class SYNTHStream(IterableDataset):
    """
    STRICT Loader: Instant resume using Arrow slicing.
    """
    def __init__(self, tokenizer, dataset_name="PleIAs/SYNTH", local_path="synth_local",
                 seq_len=512, batch_size=16, shuffle_buffer=10000, seed=42, 
                 include_query=True, include_reasoning=True, include_answer=True,
                 combine_separator="\n\n", filter_language="en", start_step=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.seed = seed
        self.start_step = start_step
        self.combine_separator = combine_separator
        self.include_query = include_query
        self.include_reasoning = include_reasoning
        self.include_answer = include_answer
        self.filter_language = filter_language

        # --- SMART PATH FINDER ---
        # Assuming run from DZero root or subfolder
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
            # Fallback for when running in different contexts
            print(f"⚠️ Warning: Could not find '{local_path}'. Dataset loading might fail.")
            self.full_path = local_path # Attempt direct load
        else:
            print(f"📂 SYNTHStream loading from: {self.full_path}")

    def _construct_text(self, ex: Dict[str, Any]) -> Optional[str]:
        # Fast language filter
        if self.filter_language:
            lang = ex.get("language")
            if not lang or (isinstance(lang, str) and lang.lower() != self.filter_language.lower()):
                return None
        
        parts = []
        query = ex.get("query", "").strip()
        if self.include_query and query:
            parts.append(f"<|im_start|>user\n{query}<|im_end|>")
        
        reasoning = ex.get("synthetic_reasoning", "").strip()
        answer = ex.get("synthetic_answer", "").strip()
        
        assistant_parts = []
        if self.include_reasoning and reasoning:
            assistant_parts.append(f"`<think>`\n{reasoning}\n`</think>`")
        if self.include_answer and answer:
            assistant_parts.append(answer)
        
        if assistant_parts:
            assistant_text = self.combine_separator.join(assistant_parts)
            parts.append(f"<|im_start|>assistant\n{assistant_text}")
        
        if not parts: return None
        return "\n".join(parts)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        try:
            # 1. Load and Shuffle full dataset
            full_ds = load_from_disk(self.full_path)
            full_ds = full_ds.shuffle(seed=self.seed)
            
            # 2. INSTANT RESUME LOGIC
            # Calculate how many examples to skip
            skip_n = self.start_step * self.batch_size
            
            if skip_n > 0 and skip_n < len(full_ds):
                print(f"⏩ Instant Jump: Slicing dataset from row {skip_n} to {len(full_ds)}")
                active_ds = full_ds.select(range(skip_n, len(full_ds)))
            else:
                active_ds = full_ds

            it = iter(active_ds)
            
        except Exception as e:
            print(f"❌ Critical Error loading dataset: {e}")
            # Generate dummy data for testing if dataset fails (fallback)
            print("⚠️ Switching to dummy data generator")
            while True:
                yield {"input_ids": torch.randint(0, 50272, (self.seq_len,), dtype=torch.long), 
                       "labels": torch.randint(0, 50272, (self.seq_len,), dtype=torch.long)}
            return

        buf: List[int] = []
        
        while True:
            while len(buf) < self.seq_len:
                try:
                    ex = next(it)
                except StopIteration:
                    print("🔄 Dataset finished, restarting...")
                    it = iter(full_ds) # Restart full dataset
                    ex = next(it)

                text = self._construct_text(ex)
                if not text: continue
                
                encoded = self.tokenizer.encode_plus(
                    text, add_special_tokens=False, return_tensors=None,
                    max_length=self.seq_len * 2, truncation=True, padding=False,
                )
                ids = encoded["input_ids"]
                if not ids: continue
                
                buf.extend(ids)
                # Keep buffer reasonable size
                if len(buf) > 4 * self.seq_len:
                    buf[:] = buf[-(4 * self.seq_len):]

            block = buf[:self.seq_len]
            del buf[:self.seq_len]
            # Create a dict that matches what main loop expects
            # Note: Main loop expects 'ids' key in batch for model input, standard naming is input_ids
            # We will yield 'ids' to match the user's specific training loop snippet
            yield {"ids": torch.tensor(block, dtype=torch.long), 
                   "targets": torch.tensor(block, dtype=torch.long)} # targets same as ids for shifting later

# --- Adaptive & Architectural Components (User Provided) ---

class ContinueHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, bias_init: float = -2.0):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, bias_init)
    
    def forward_logits(self, f):
        """Return raw logits for noise injection."""
        return self.net(f.float()).squeeze(-1)
    
    def forward(self, f):
        """Backward-compatible: returns probability."""
        return torch.sigmoid(self.forward_logits(f))

# FIX: Per-token adaptive features (User Feedback #B)
def compute_adaptive_features(attn_w, dh, h_prev, dh_prev=None):
    # attn_w: [B, H, T, T] (or similar)
    # dh, h_prev: [B, T, D]
    # Return: [B, T, F]
    
    eps = 1e-8
    B, H, T, _ = attn_w.shape
    
    # 1. Entropy (per token query)
    # attn_w slices: we want entropy of the distribution over keys for each query
    # ent[b, h, t] = sum(-p * log p)
    # Note: attn_w is causal, so padded entries are 0. log(0) handled by +eps
    ent = -torch.sum(attn_w * torch.log(attn_w + eps), dim=-1) # [B, H, T]
    H_norm = (ent / math.log(T + 1e-6)).mean(dim=1).detach()   # [B, T] (mean over heads)

    # 2. Top-K Sum (Confidence)
    # sort over last dim (keys)
    topk_val, _ = torch.topk(attn_w, k=min(16, T), dim=-1)
    S_topk = topk_val.sum(dim=-1).mean(dim=1).detach() # [B, T]
    
    # 3. Agreement (Head variance)
    # std over heads (dim 1)
    # attn_w shape [B, H, T, T]. std(dim=1) -> [B, T, T]
    # mean over keys (dim -1) -> [B, T]
    # Wait, Agreement defined as: 1 - mean_over_keys(std_over_heads(w))
    Agree = (1.0 - attn_w.std(dim=1).mean(dim=-1)).clamp(0.0, 1.0).detach() # [B, T]

    # 4. Relative Update Norm
    rms_dh = torch.sqrt((dh.float()**2).mean(dim=-1)) # [B, T]
    rms_h = torch.sqrt((h_prev.float()**2).mean(dim=-1)) # [B, T]
    delta_rel = (rms_dh / (rms_h + eps)).clamp(0.0, 10.0).detach()

    # 5. Cosine Similarity with prev update
    if dh_prev is not None:
        cos_dir = F.cosine_similarity(dh.float(), dh_prev.float(), dim=-1).detach() # [B, T]
    else:
        cos_dir = torch.zeros(B, T, device=dh.device)

    # Stack features: [B, T, 5]
    f_k = torch.stack([H_norm, S_topk, Agree, delta_rel, cos_dir], dim=-1)
    
    # Return features and dh for state tracking (no flattening needed for next step if matching tokens)
    return f_k, dh, H_norm

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x): return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class AdaptiveDecoderLayer(nn.Module):
    def __init__(self, layer_idx, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.input_ln, self.post_ln = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.self_attn = MultiheadLatentAttention(hidden_size, num_heads)
        self.mlp = MoEFFN(hidden_size, intermediate_size)
        self.Kmax = 4
        self.register_buffer('alpha', torch.tensor([1.0, 0.7, 0.5, 0.35]))
        # FIX: Zero init for layer_emb (User Feedback #1)
        self.layer_emb = nn.Parameter(torch.zeros(16))
        self.continue_head = ContinueHead(in_dim=21)

    def forward(self, x, mask=None, use_rec=True, global_step=0, lambda_p=0.0, force_2=False, clamp_p=False, capture_h1=False, halt_thr=0.05):
        if not use_rec:
            attn_out, _, _ = self.self_attn(self.input_ln(x), mask=mask, return_weights=False)
            h = x + attn_out
            h = h + self.mlp(self.post_ln(h))
            return h, {"k": 1.0, "ent": 0.0, "p_loss": x.new_tensor(0.0)}

        B, T, D = x.shape
        # FIX: Per-token survival [B, T]
        h, s = x, torch.ones(B, T, device=x.device, dtype=torch.float32)
        expected_k = torch.zeros(B, T, device=x.device, dtype=torch.float32)
        dh_prev, ents = None, []
        
        # NEW: Per-token execution tracking
        # exec_per_token[b,t] = how many passes this token ACTUALLY executed
        # We set exec_per_token = k at START of pass k for ALIVE tokens
        exec_per_token = torch.zeros((B, T), device=x.device, dtype=torch.float32)
        
        _, _, kv_static = self.self_attn(self.input_ln(h), mask)

        # Diagnostics containers
        p_vals = []
        delta_rels = []
        h_diffs = [] # New: Gated State Delta
        s_trace = []  # User Fix #2: Track survival trajectory
        h1_tensor = None
        k_exec = 0

        for k in range(1, self.Kmax + 1):
            # FIX: Track execution at START of pass, not end
            # Tokens with s >= halt_thr ARE executing this pass k
            with torch.no_grad():
                is_executing = (s >= halt_thr)  # Tokens that will run pass k
                # Set their exec count to k (overwrite, don't increment)
                exec_per_token = torch.where(is_executing, torch.full_like(exec_per_token, float(k)), exec_per_token)
            
            attn_out, attn_w, _ = self.self_attn(self.input_ln(h), mask, return_weights=True, kv_cache=kv_static)
            u_k = attn_out + self.mlp(self.post_ln(h + attn_out))
            dh_k = self.alpha[k-1] * u_k
            
            # Compute stats (Per Token)
            # Pass full attn_w [B, H, T, T]
            f_k, dh_curr, h_ent = compute_adaptive_features(attn_w, dh_k, h, dh_prev)
            ents.append(h_ent.mean().item())
            delta_rels.append(f_k[:, :, 3].mean().item()) 
            
            # Expand layer_emb to [B, T, 16]
            layer_emb_exp = self.layer_emb.float().view(1, 1, -1).expand(B, T, -1)
            p_k_input = torch.cat([f_k.detach(), layer_emb_exp], dim=-1) # [B, T, 21]
            p_k = self.continue_head(p_k_input) # [B, T] (after squeeze inside head)
            
            if force_2: 
                # Force 2 means p=1 for k=1, p=0 else
                target = torch.ones_like(p_k) if k == 1 else torch.zeros_like(p_k)
                p_k = target
            if clamp_p: p_k = p_k.clamp(max=0.95) # FIX: Upper clamp only, allow p -> 0 to enable stopping at k=1
            
            # COLLAPSE FIX: Prevent absorbing state (training only, not during force_2)
            # This ensures ∂L/∂gate ≠ 0 always - keeps gradient signal alive
            if self.training and not force_2:
                p_k = p_k.clamp(min=0.01)
            
            p_vals.append(p_k)
            
            # S_TRACE: Capture survival entering this step (starts at 1.0)
            s_trace.append(s.mean().item())

            # Update h (Per Token Gating)
            # s is [B, T], dh_k is [B, T, D] -> s.unsqueeze(-1) matches [B, T, 1]
            gated_dh = s.to(h.dtype).unsqueeze(-1) * dh_k
            
            # New Metric: Gated State Delta (||s * dh|| / ||h||)
            # Measures effective change caused by this step
            with torch.no_grad():
                h_norm = h.norm(dim=-1) + 1e-6
                dh_norm = gated_dh.norm(dim=-1)
                h_diff_rel = (dh_norm / h_norm).mean().item()
                h_diffs.append(h_diff_rel)

            h = h + gated_dh
            
            # Capture h_1 (End of first pass) for Entropy Delta calculation
            if capture_h1 and k == 1:
                # OPTIMIZATION: Detach immediately to save massive activation graph
                h1_tensor = h.detach()
            expected_k = expected_k + s
            s = s * p_k.float()
            dh_prev = dh_curr
            
            # FIX: FRACTION-BASED EARLY BREAK (fixes "one stubborn token drags batch" problem)
            # Stop when only a small fraction of tokens are still alive, not when ALL are done
            alive_frac = (s >= halt_thr).float().mean().item()
            if alive_frac <= 0.02:  # 2% still alive = batch is "done enough"
                break

        # NEW: Compute per-token execution statistics
        with torch.no_grad():
            exec_flat = exec_per_token.flatten()  # [B*T]
            avg_exec_token = exec_flat.mean().item()
            # P95: 95th percentile of per-token execution
            p95_exec_token = torch.quantile(exec_flat, 0.95).item()
            # FIX: k_exec should be derived from actual token execution, not loop counter
            k_exec = int(exec_per_token.max().item())
            # Wasted compute: how much batch-max exceeds token-mean
            wasted_compute = (k_exec / avg_exec_token) - 1.0 if avg_exec_token > 0 else 0.0
            
            # NEW: Execution tail fractions (what % of tokens need ≥N passes)
            total_tokens = exec_flat.numel()
            frac_exec_ge2 = (exec_flat >= 2).sum().item() / total_tokens
            frac_exec_ge3 = (exec_flat >= 3).sum().item() / total_tokens
            frac_exec_ge4 = (exec_flat >= 4).sum().item() / total_tokens
            
            # NEW A) Hard exec prediction: simulate threshold stopping using p values
            # This predicts what tok_exec SHOULD be if we apply threshold correctly
            if len(p_vals) > 0:
                thr = halt_thr  # Use dynamic threshold for prediction
                s_pred = torch.ones(B, T, device=x.device, dtype=torch.float32)
                hard_exec = torch.ones(B, T, device=x.device, dtype=torch.float32)  # Start at 1 pass
                for k_p, p_k in enumerate(p_vals):
                    s_pred = s_pred * p_k.float()
                    # Tokens still "alive" after this pass -> increment their exec count
                    still_alive = (s_pred >= thr)
                    hard_exec = hard_exec + still_alive.float()
                hard_exec = hard_exec.clamp(max=float(self.Kmax))
                hard_exec_flat = hard_exec.flatten()
                hard_exec_pred_avg = hard_exec_flat.mean().item()
                hard_exec_pred_p95 = torch.quantile(hard_exec_flat, 0.95).item()
            else:
                hard_exec_pred_avg, hard_exec_pred_p95 = 1.0, 1.0

        # Collect extended stats
        # NORMALIZED PONDER PENALTY (User Fix: Excess-depth scaling)
        avg_k = expected_k.mean()
        ponder_signal = torch.relu(avg_k - 1.0) / (self.Kmax - 1.0)  # [0, 1]
        exec_frac = (k_exec - 1.0) / (self.Kmax - 1.0)  # Penalize actual compute
        
        # Gap Penalty REMOVED: It was pushing expected_k UP to match k_exec (gradient direction).
        # We want expected_k to drop, so k_exec can follow.
        # gap_penalty = ... (removed)
        
        # Combined Signal: density (ponder) + execution cost
        # Fix: Prioritize differentiable ponder_signal over discrete exec_frac
        combined_signal = 0.85 * ponder_signal + 0.15 * exec_frac
        p_loss = lambda_p * combined_signal
        
        mean_ent = sum(ents)/len(ents) if ents else 0.0
        
        # Compute Δh: Mean relative change between passes
        # h_diffs[k] = ||gated_dh_k|| / ||h_k|| for each pass
        # We want the AVERAGE across passes, or specifically h_diff[0] (first pass contribution)
        h_diff_mean = sum(h_diffs)/len(h_diffs) if h_diffs else 0.0
        
        # Aggregate diagnostics
        if p_vals:
            # Stats over full [B, T] tensors
            p_all = torch.stack(p_vals) # [K, B, T]
            # FIX: Log p_mu of the FIRST gate (k=1) to show "willingness to continue"
            # Averaging over K steps (0.95 and 0.05) yields 0.50, which is misleading.
            p_mean = p_vals[0].mean().item() if len(p_vals) > 0 else 0.0
            p_min, p_max = p_all.min().item(), p_all.max().item()
            p_std = p_all.std().item() # Trace variance
            
            # p_diff: difference between gate at step 2 vs step 1 (how much does gate change?)
            if len(p_vals) >= 2:
                p_diff = (p_vals[1] - p_vals[0]).mean().item()
            else:
                p_diff = 0.0
            
            # NEW: Absorbing-state proximity (% of p values at or below 0.05)
            # This tracks how close the gate is to "collapse" - hugging the lower clamp
            p_near_clamp = (p_all <= 0.05).float().mean().item()
        else:
            p_mean, p_min, p_max, p_std = 0.0, 0.0, 0.0, 0.0
            p_diff = 0.0
            p_near_clamp = 0.0
            
        stats = {
            "k": avg_k.item(),
            "ent": mean_ent,
            "p_loss": p_loss,
            "k_exec": k_exec,
            "s_final": s.mean().item(),
            "s_trace": s_trace,
            "h_1": h1_tensor,  # <--- ADD THIS LINE to wire the hidden state
            "p_mean": p_mean, "p_min": p_min, "p_max": p_max, "p_std": p_std,
            "d_rel": sum(delta_rels)/len(delta_rels) if delta_rels else 0.0,
            # Ponder diagnostics
            "ponder_signal": ponder_signal.item() if hasattr(ponder_signal, 'item') else ponder_signal,
            "exec_frac": exec_frac,
            "combined_ponder": combined_signal.item() if hasattr(combined_signal, 'item') else combined_signal,
            # Per-token execution stats
            "avg_exec_token": avg_exec_token,
            "p95_exec_token": p95_exec_token,
            "wasted_compute": wasted_compute,
            # NEW: Execution tail fractions
            "frac_ge2": frac_exec_ge2,
            "frac_ge3": frac_exec_ge3,
            "frac_ge4": frac_exec_ge4,
            # Hard exec prediction (should match tok_exec if logging is correct)
            "hard_exec_pred_avg": hard_exec_pred_avg,
            "hard_exec_pred_p95": hard_exec_pred_p95,
            # Recurrence quality metrics (fixed)
            "h_diff": h_diff_mean,  # Mean ||gated_dh|| / ||h|| across passes
            "p_diff": p_diff,       # p2 - p1: how gate changes between passes
            # NEW: Absorbing-state proximity (% of p values near min clamp)
            "p_near_clamp": p_near_clamp if 'p_near_clamp' in dir() else 0.0,
        }
        return h, stats

class SmolLM(nn.Module):
    def __init__(self, vocab_size, bpe_vocab, pf_codec, hidden_size, num_layers, num_heads, intermediate_size, K):
        super().__init__()
        self.embed = HybridEmbeddingTorch(bpe_vocab, pf_codec, K=K, gate_dim=384, hidden=1536, n_heads=num_heads)
        self.pf_proj = nn.Linear(2048, hidden_size, bias=False)
        nn.init.normal_(self.pf_proj.weight, std=0.02/math.sqrt(2048)) # Scale Match
        
        self.embed_ln = RMSNorm(hidden_size)
        self.lambda_e = nn.Parameter(torch.zeros(hidden_size)) # Baton
        self.layers = nn.ModuleList([AdaptiveDecoderLayer(i, hidden_size, num_heads, intermediate_size) for i in range(num_layers)])
        self.norm, self.lm_head = RMSNorm(hidden_size), nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, ids, mask=None, use_rec=True, step=0, global_step=0, lambda_p=0.0, force_2=False, clamp_p=False, capture_h1=False, halt_thr=0.05):
        # Support both step and global_step for compatibility
        if global_step == 0 and step > 0: global_step = step
        
        # Embed
        emb_out = self.embed(ids) # [B, T, 2048]
        x = self.pf_proj(emb_out)
        x = self.embed_ln(x)
        
        # State injection (Baton)
        sigmoid_lambda = torch.sigmoid(self.lambda_e)
        x_prev = x[:, :-1]
        x_curr = x[:, 1:]
        injection = sigmoid_lambda * x_prev
        
        # FIX: Avoid in-place modification (x[:, 1:] = ...) to prevent gradient errors
        # Reconstruct x by concatenating first token + modified rest
        x_first = x[:, 0:1]
        x_rest = x_curr + injection
        x = torch.cat([x_first, x_rest], dim=1)
        
        # Baton Diagnostics
        with torch.no_grad():
            baton_mag = torch.sqrt((injection.float()**2).mean())
            x_mag = torch.sqrt((x_curr.float()**2).mean())
            baton_ratio = (baton_mag / (x_mag + 1e-6)).item()
            lambda_mean = sigmoid_lambda.mean().item()
        
        # Forward pass through layers
        l_stats, total_ponder = [], 0
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            # Only capture h1 for the last layer to save memory
            x, stats = layer(x, mask, use_rec=use_rec, global_step=step, 
                                 lambda_p=lambda_p, force_2=force_2, clamp_p=clamp_p,
                                 capture_h1=(is_last and capture_h1),
                                 halt_thr=halt_thr)
            l_stats.append(stats)
            if isinstance(stats.get("p_loss"), torch.Tensor):
                total_ponder += stats.get("p_loss")
            else:
                total_ponder += 0
        
        # logits_1 calculation
        logits_1 = None
        if capture_h1 and l_stats:
            # We check the LAST layer's stats for 'h_1'
            last_stats = l_stats[-1]
            if last_stats.get('h_1') is not None:
                # OPTIMIZATION: Detach h1 and use no_grad to keep memory footprint low
                # logits_1 is purely a baseline for the reward credit
                with torch.no_grad():
                    h1 = last_stats['h_1'].detach()
                    # Apply same head as final output
                    logits_1 = self.lm_head(self.norm(h1))

        if l_stats:
            l_stats[0]["baton_rat"] = baton_ratio
            l_stats[0]["lam_mu"] = lambda_mean
        
        # Flatten stats: we return l_stats as the 3rd argument (stats), 
        # but main loop might expect 'total_ponder' too? 
        # Main loop signature: outputs, logits_1, stats = model(...)
        # So we should return: logits, logits_1, l_stats
        # The 'total_ponder' is now unused in return? 
        # Wait, the main loop calculates 'total_ponder_loss = sum([s['p_loss'] for s in stats])'.
        # So we don't need to return total_ponder explicitly.
        
        return self.lm_head(self.norm(x)), logits_1, l_stats

# -----------------------------
# Checkpoint Helpers
# -----------------------------

# -----------------------------
# Checkpoint Helpers
# -----------------------------

def save_checkpoint(model, optimizer, step, loss, embedding_type="fourier", save_dir="checkpoints", lambda_p_state=None):
    """Save model checkpoint with embedding_type prefix."""
    os.makedirs(save_dir, exist_ok=True)
        
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'embedding_type': embedding_type,
        'lambda_p_state': lambda_p_state,  # Compute price controller state
    }
    
    # Save with embedding_type prefix
    path = os.path.join(save_dir, f'{embedding_type}_checkpoint_step_{step:07d}.pt')
    torch.save(checkpoint, path)
    
    # Also save as latest for easy resuming
    latest_path = os.path.join(save_dir, f'{embedding_type}_latest.pt')
    torch.save(checkpoint, latest_path)
    
    lp_str = f"{lambda_p_state:.2e}" if lambda_p_state is not None else "N/A"
    print(f"💾 Checkpoint saved: {path} (λ_p={lp_str})")

def load_checkpoint(model, optimizer, embedding_type="fourier", save_dir="checkpoints", checkpoint_path=None):
    """
    Load checkpoint from a specific path or find the latest checkpoint.
    """
    # If a specific checkpoint path is provided, use it directly
    if checkpoint_path is not None:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint file not found: {checkpoint_path}")
            return 0, 0.0
        latest_path = checkpoint_path
    else:
        # Auto-discover: Prefer highest step number over 'latest.pt' (Safer against overwrites)
        import glob
        pattern = os.path.join(save_dir, f'{embedding_type}_checkpoint_step_*.pt')
        checkpoints = glob.glob(pattern)
        
        if checkpoints:
            # Sort by step number (extract from filename)
            def get_step(path):
                basename = os.path.basename(path)
                try:
                    step_str = basename.split('_step_')[1].split('.pt')[0]
                    return int(step_str)
                except:
                    return 0
            
            checkpoints.sort(key=get_step, reverse=True)
            latest_path = checkpoints[0]
            print(f"🔎 Found highest step checkpoint: {latest_path}")
        else:
            # Fallback to simple latest if no numbered files
            latest_path = os.path.join(save_dir, f'{embedding_type}_latest.pt')
    
    if not os.path.exists(latest_path):
        return 0, 0.0
    
    try:
        print(f"🔄 Loading checkpoint: {latest_path}")
        checkpoint = torch.load(latest_path, map_location='cpu')
        
        # Load model state (with strict=False to handle architecture changes)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model_state_dict'], strict=False
        )
        if missing_keys:
            print(f"[RESUME] Warning: Missing keys: {missing_keys[:5]}...")
        
        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', 0.0)
        lambda_p_state = checkpoint.get('lambda_p_state', None)
        
        print(f"✅ Loaded checkpoint from {latest_path} at step {step}")
        if lambda_p_state is not None:
            print(f"   📊 Restored λ_p = {lambda_p_state:.2e}")
        return step, loss, lambda_p_state
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 0, 0.0, None  # Return 3 values on error too

# -----------------------------
# Main Execution
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["fourier", "baseline"], help="Training mode")
    parser.add_argument("--hard_kill", action="store_true", help="Disable recurrence")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (or 'latest')")
    parser.add_argument("--set_lambda", type=float, default=None, help="Manually set/override compute price λ_p")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    args = parser.parse_args()

    # Determinism
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Pause Handler
    pause_handler = PauseHandler()
    if pause_handler.check_pause_flag():
        print("⚠️ Pause flag found. Clearing for new run (or resume will handle via args).")
        pause_handler.resume()

    # Confirmed Battle-Tested Parameters
    D = 768; L = 8; HEADS = 12; INT = 512; K_SEM = 1536
    
    # MPS Optimization
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"🔥 Using device: {device.type}")

    # Tokenizer & Dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Add special tokens if needed (from FC4 logic)
    special_tokens = {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|assistant|>", "<|user|>"]}
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad_0|>'})
        tokenizer.pad_token = '<|pad_0|>'
        
    vocab_size = 50272 # len(tokenizer) might be smaller, but we target this for model
    
    # Prepare Fourier configs
    if args.mode == "fourier":
        chars, char_to_id = discover_chars_from_bpe_tokenizer(tokenizer, vocab_size)
        chars128, char_to_id = pad_char_vocab_128(chars)
        pf_cfg = PFConfig(
            char_vocab=chars128, char_to_id=char_to_id, 
            CHAR_DIM=128, POS_DIM=16, D=2048, 
            length_normalize=True, truncate_long_words=True
        )
        pf_codec = PFCodec(pf_cfg)
        
        # BPE vocab list for embedding
        print("🔄 Preparing BPE vocab for Fourier...")
        bpe_vocab = []
        for i in range(min(vocab_size, len(tokenizer))):
            try:
                bpe_vocab.append(tokenizer.decode([i]))
            except:
                bpe_vocab.append(f"<TOKEN_{i}>")
        # Pad bpe_vocab if needed
        while len(bpe_vocab) < vocab_size:
            bpe_vocab.append("<PAD>")
    else:
        # Baseline stub if needed, though this script focuses on Fourier Ouro
        bpe_vocab = [""] * vocab_size
        pf_codec = None # Will error if used

    print("🧠 Initializing Model...")
    model = SmolLM(vocab_size, bpe_vocab, pf_codec, D, L, HEADS, INT, K_SEM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Load Checkpoint logic BEFORE dataset init to get start_step
    start_step = 0
    save_dir = "checkpoints"
    
    # Check resume flag
    checkpoint_path = None
    
    # Auto-Resume Logic (User Request): Check if latest exists even if not flagged
    if not args.resume:
        potential_latest = os.path.join(save_dir, f'{args.mode}_latest.pt')
        if os.path.exists(potential_latest):
            print(f"\n[AUTO-RESUME] 🔄 Found existing checkpoint at {potential_latest}")
            print("[AUTO-RESUME] 🔄 Automatically resuming to prevent data loss.")
            args.resume = "latest"

    if args.resume:
        if args.resume == "latest":
             # Logic handled inside load_checkpoint to find latest
             pass 
        else:
            checkpoint_path = args.resume

    # Always try to load if resume is requested (explicitly or via auto-detect)
    loaded_lambda_p = None  # Will be set if checkpoint has it
    if args.resume:
        start_step, start_loss, loaded_lambda_p = load_checkpoint(
            model, optimizer, embedding_type=args.mode, 
            save_dir=save_dir, checkpoint_path=checkpoint_path
        )
        if start_step > 0:
             print(f"⏩ Resuming from step {start_step} (loss: {start_loss:.4f})")
    
    # Data Loader (Deterministic & Resumable)
    print(f"📂 Initializing Data Loader (Start Step: {start_step})...")
    dataset = SYNTHStream(
        tokenizer, seq_len=512, batch_size=16, 
        seed=args.seed, start_step=start_step
    )
    train_loader = DataLoader(
        dataset, 
        batch_size=16, 
        num_workers=0 # MPS often prefers main thread
    )
    train_iterator = iter(train_loader)
    
    # Prompt sampler for generation (Deterministic & Aligned with Training)
    prompt_sampler = SYNTHPromptSampler(
        dataset_name="PleIAs/SYNTH",
        tokenizer=tokenizer,
        seed=args.seed  # Same seed as training data = deterministic prompts
    )
    
    gen_every = 500
    gen_warmup_steps = 100
    
    # LR schedule parameters (10k steps)
    target_total_steps = 10000
    warmup_steps = 500
    max_lr_val = 3.0e-4
    min_lr_val = 3.0e-5
    
    # Persistent Logs
    log_name = "log_fourier_adaptive.txt" if args.mode == "fourier" else f"log_{args.mode}_adaptive.txt"
    # Only overwrite if starting fresh; append if resuming
    mode = "a" if start_step > 0 else "w"
    with open(log_name, mode) as f:
        if start_step == 0:
            f.write(f"INIT | PARAMS: {sum(p.numel() for p in model.parameters())} | K: {K_SEM}\n")

    criterion = nn.CrossEntropyLoss()
    
    # --- DUAL-ASCENT COMPUTE CONTROLLER (Collapse Fix) ---
    # λ_p is now a *market price* that adjusts based on actual compute usage
    # Key invariant: only policy-induced compute feeds the controller, never audit-forced compute
    TARGET_K = 1.30           # Desired average depth (start modest)
    DUAL_LR = 5e-5            # Controller learning rate (slow = stable)
    LAMBDA_MIN = 1e-4         # Floor (never free compute)
    LAMBDA_MAX = 5e-2         # Ceiling (prevent runaway tax)
    AUDIT_PROB = 0.05         # 5% counterfactual audits
    
    # --- ASYMMETRIC DEADBAND CONTROLLER (Stability Fix) ---
    # Once λ_p stabilizes at high value, switch to conservative deadband mode
    # to prevent whipsaw oscillations around TARGET_K
    DEADBAND_TRIGGER_LAM = 0.50      # λ_p threshold to trigger deadband (0-1 normalized)
    DEADBAND_TRIGGER_WINDOW = 100    # Steps λ_p must exceed threshold to trigger
    DEADBAND_HIGH_PCT = 0.05         # +5% above TARGET_K → penalize (raise λ_p)
    DEADBAND_LOW_PCT = 0.02          # -2% below TARGET_K → relax (lower λ_p)
    DEADBAND_LR_SCALE = 0.5          # Slower adjustments once in deadband
    
    # Deadband state tracking
    deadband_active = False
    deadband_steps_above_threshold = 0
    deadband_trigger_step = None     # Step when deadband was triggered
    
    lambda_p_state = 1e-4     # Initialize to minimum (not zero!)
    
    # Restore lambda_p_state from checkpoint if available
    if loaded_lambda_p is not None:
        lambda_p_state = loaded_lambda_p
        print(f"📈 Restored compute price λ_p = {lambda_p_state:.2e}")
    
    # CLI Override (User Feature)
    if args.set_lambda is not None:
        lambda_p_state = float(args.set_lambda)
        print(f"🛠️ CLI Override: Setting compute price λ_p = {lambda_p_state:.2e}")
    
    # Tracking for separated audit vs policy stats
    credit_ema = 0.0          # EMA baseline for delta_ce
    CREDIT_BETA = 0.995       # Slow EMA to prevent "learning away improvements"
    
    print("🚀 Starting Training Loop...")
    
    # Fix #1: Training length 10k (Progress bar removed for cleaner logs)
    pbar = range(start_step, target_total_steps)
    
    ckpt_every = 250  # More frequent to prevent data loss (was 1000)
    
    # Memory cleanup frequencies (matching deepscreen exactly)
    gc_every = 1  # gc.collect() every 2 steps (aggressive)
    mem_every = 5  # Cache clearing every 100 steps
    
    for step in pbar:
        # Check Pause
        if pause_handler.should_pause():
            print(f"\n[PAUSE] Saving checkpoint at step {step} and exiting...")
            loss_val = lm_loss.item() if 'lm_loss' in locals() else 0.0
            save_checkpoint(model, optimizer, step, loss_val, args.mode, save_dir, lambda_p_state=lambda_p_state)
            return

        # Check Periodic Checkpoint
        if step > 0 and step % ckpt_every == 0:
             loss_val = lm_loss.item() if 'lm_loss' in locals() else 0.0
             save_checkpoint(model, optimizer, step, loss_val, args.mode, save_dir, lambda_p_state=lambda_p_state)

        # Fix #1: Official LR Schedule (Wrapped Cosine, NO NOTCH)
        warmup_steps = 500
        max_lr_val, min_lr_val = 3e-4, 3e-5
        
        if step < warmup_steps:
            lr = max_lr_val * step / warmup_steps
        else:
            # Shift cosine to start AFTER warmup
            decay_start = warmup_steps
            progress = (step - decay_start) / (target_total_steps - decay_start)
            progress = min(max(progress, 0.0), 1.0)
            lr = min_lr_val + 0.5 * (max_lr_val - min_lr_val) * (1 + math.cos(math.pi * progress))
        
        for param_group in optimizer.param_groups: param_group['lr'] = lr

        # 2. Ponder & Force Schedule
        force_2_sched = (step < 800)  # Extended stabilization
        
        # --- COLLAPSE FIX: Dual-Ascent λ_p + Audit Forcing ---
        PONDER_START = 1200   # When to start ponder penalty
        CLAMP_END = 4000      # Safety rails off at 40% of training
        HALT_THR = 0.10       # Raised to fix soft/hard mismatch (was 0.03)
        REWARD_ETA = 2.0      # The "Coupon" multiplier for credit
        
        # Audit forcing: counterfactual k=2 to measure ΔCE even when gate stops
        # CRITICAL: This must NOT feed into the dual-ascent controller!
        audit_force_2 = (step >= PONDER_START) and (random.random() < AUDIT_PROB)
        force_2_effective = force_2_sched or audit_force_2
        
        # Use current lambda_p_state (updated at end of each non-audit step)
        lambda_p = lambda_p_state if step >= PONDER_START else 0.0
            
        clamp_p = (step < CLAMP_END)
        
        # 3. Forward & Loss
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
            
        input_ids = batch['ids'].to(device)
        target_ids = batch['targets'].to(device)
        
        # Timing start
        sync_device(device)
        t0 = time.time()
        
        B, Seq = input_ids.shape
        attention_mask = torch.triu(torch.full((Seq, Seq), float('-inf'), device=device), diagonal=1)
        
        # Input: input_ids[:, :-1], Target: target_ids[:, 1:]
        inp_seq = input_ids[:, :-1]
        tgt_seq = target_ids[:, 1:]
        
        # Adjust mask for sequence length -1
        seq_len_adj = Seq - 1
        attention_mask = attention_mask[:seq_len_adj, :seq_len_adj]

        # Determine if we should capture pass-1 for credit (Economic Fix)
        # We do this every step after PONDER_START to keep the economic signal frequent
        do_credit = (step >= PONDER_START)
        do_diagnostics = (step % 10 == 0)  # Keep diagnostic logging every 10 steps
        
        outputs, logits_1, stats = model(
            inp_seq, 
            mask=attention_mask, 
            global_step=step, 
            lambda_p=lambda_p, 
            force_2=force_2_effective,  # CHANGED: Use effective (includes audit)
            clamp_p=clamp_p, 
            capture_h1=do_credit,  # Enable Pass-1 capture for the coupon
            halt_thr=HALT_THR      # Pass dynamic threshold (Economic Spec)
        )
        
        # Calculate loss
        lm_loss = criterion(outputs.reshape(-1, vocab_size), tgt_seq.reshape(-1))
        
        # 3.5 Recurrence Diagnostics (Delta Entropy, Delta NLL)
        d_ent_global = 0.0
        d_nll_global = 0.0
        delta_h = 0.0
        nll_1_val = None
        
        # Only compute these expensive metrics if needed (Diagnostics or Credit tracking)
        if (do_diagnostics or do_credit) and logits_1 is not None:
            with torch.no_grad():
                # 1. Entropy Delta (Model Confidence)
                # Compute H1
                prob_1 = F.softmax(logits_1, dim=-1)
                ent_1 = -(prob_1 * torch.log(prob_1 + 1e-9)).sum(dim=-1).mean()
                
                # Compute H2
                prob_2 = F.softmax(outputs, dim=-1)
                ent_2 = -(prob_2 * torch.log(prob_2 + 1e-9)).sum(dim=-1).mean()
                
                d_ent_global = (ent_2 - ent_1).item()
                delta_h = d_ent_global # For the log line below
                
                # 2. NLL Delta (Model Accuracy)
                nll_1 = criterion(logits_1.reshape(-1, vocab_size), tgt_seq.reshape(-1))
                d_nll_global = (lm_loss - nll_1).item()
                nll_1_val = nll_1 # Save for diagnostics
                
                # Cleanup intermediates immediately
                del prob_1, prob_2, ent_1, ent_2, nll_1
            
        
        # --- Economic Fix: Ponder Credit (The Coupon) ---
        # Total ponder from all layers (raw)
        total_ponder_loss = sum([s['p_loss'] for s in stats])
        
        # Calculate Ponder Credit: reward for improvement (CE1 - CE2)
        delta_ce = torch.tensor(0.0, device=device)

        if logits_1 is not None:
            # Note: nll_1_val was computed in Section 3.5 above
            # credit = relu(CE1 - CE2): positive when Pass-2 improved over Pass-1
            delta_ce = (nll_1_val - lm_loss).clamp(min=0)
        
        # Apply: ponder = relu(ponder_raw - η * credit)
        # This "rewards" the model for spending compute that actually helped
        total_ponder_loss = torch.relu(total_ponder_loss - REWARD_ETA * delta_ce)
        
        # Adaptive Objective
        total_loss = lm_loss + total_ponder_loss

        total_loss.backward()
        
        # Clip grad norm and capture it (User Feedback #E)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        
        # --- DUAL-ASCENT CONTROLLER UPDATE (Collapse Fix) ---
        # CRITICAL: Only update on non-audit batches to prevent feedback trap
        if step >= PONDER_START and not audit_force_2:
            # Compute policy-induced avg_k (not forced)
            policy_avg_k = sum([s['k'] for s in stats]) / len(stats) if stats else 1.0
            
            # --- DEADBAND TRIGGER CHECK ---
            # Normalize lambda_p_state to [0,1] for threshold comparison
            lam_normalized = (lambda_p_state - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
            
            if not deadband_active:
                # Track sustained high λ_p
                if lam_normalized >= DEADBAND_TRIGGER_LAM:
                    deadband_steps_above_threshold += 1
                else:
                    deadband_steps_above_threshold = max(0, deadband_steps_above_threshold - 1)
                
                # Trigger deadband once threshold is sustained
                if deadband_steps_above_threshold >= DEADBAND_TRIGGER_WINDOW:
                    deadband_active = True
                    deadband_trigger_step = step
                    print(f"\n🔒 [DEADBAND ENGAGED] λ_p={lambda_p_state:.2e} stable for {DEADBAND_TRIGGER_WINDOW} steps")
                    print(f"   Controller now uses asymmetric bounds: TARGET_K={TARGET_K:.2f} ± [{-DEADBAND_LOW_PCT*100:.1f}%, +{DEADBAND_HIGH_PCT*100:.1f}%]")
            
            # --- CONTROLLER UPDATE (Asymmetric Deadband or Standard) ---
            if deadband_active:
                # Asymmetric deadband: freeze λ_p inside band, slow adjust outside
                k_high = TARGET_K * (1.0 + DEADBAND_HIGH_PCT)
                k_low = TARGET_K * (1.0 - DEADBAND_LOW_PCT)
                
                if policy_avg_k > k_high:
                    # Above ceiling: RAISE λ_p (penalize excess compute)
                    lambda_p_state += DUAL_LR * DEADBAND_LR_SCALE * (policy_avg_k - k_high)
                elif policy_avg_k < k_low:
                    # Below floor: LOWER λ_p (encourage more compute)
                    lambda_p_state += DUAL_LR * DEADBAND_LR_SCALE * (policy_avg_k - k_low)
                # else: Inside deadband → λ_p frozen (no update)
            else:
                # Standard dual ascent: raise price if overspending, lower if underspending
                lambda_p_state += DUAL_LR * (policy_avg_k - TARGET_K)
            
            lambda_p_state = float(max(LAMBDA_MIN, min(LAMBDA_MAX, lambda_p_state)))
        
        # EMA credit baseline update (for advantage-style reward)
        if delta_ce.item() > 0:
            credit_ema = CREDIT_BETA * credit_ema + (1 - CREDIT_BETA) * delta_ce.item()

        # Timing end
        sync_device(device)
        dt = (time.time() - t0) * 1000.0  # ms
        
        # tok/sec (tokens processed = B * (Seq-1))
        tokens_per_sec = (inp_seq.numel()) / (dt / 1000.0)

        # 4. Telemetry Log
        # Log MAIN process line EVERY step (or close to it)
        if step % 1 == 0:
            p_loss_val = total_ponder_loss.item()
            grad_norm_val = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
            loss_val = total_loss.item()
            lm_val = lm_loss.item()
            reward_val = (REWARD_ETA * delta_ce).item() if delta_ce.item() > 0 else 0.0
            
            # Construct log line with Economic metrics (Net Ponder + Coupon)
            # Format: PONDER: [Net Cost] (RWD: [Coupon Value])
            audit_flag = "[AUDIT] " if audit_force_2 else ""
            # Compute controller error for dashboard
            policy_k = sum([s['k'] for s in stats]) / len(stats) if stats else 1.0
            ctrl_err = policy_k - TARGET_K
            # Compute absorbing-state proximity (avg across layers)
            p_near_avg = sum([s.get('p_near_clamp', 0.0) for s in stats]) / len(stats) if stats else 0.0
            
            # Compute raw ponder cost before coupon (for transparency)
            raw_ponder = sum([s['p_loss'].item() if hasattr(s['p_loss'], 'item') else s['p_loss'] for s in stats])
            
            log_line = (f"STEP: {step} {audit_flag}| TOT: {loss_val:.4f} | LM: {lm_val:.4f} | "
                        f"PONDER: {p_loss_val:.6f} (RAW: {raw_ponder:.6f}, RWD: {reward_val:.6f}) | "
                        f"LR: {lr:.4e} | GRAD: {grad_norm_val:.4f} | λ_p: {lambda_p_state:.2e}{'[DB]' if deadband_active else ''} | "
                        f"CTRL_ERR: {ctrl_err:+.3f} | P_NR: {p_near_avg:.1%}")
            
            # Append global baton stats if available
            baton_str = ""
            if stats and "baton_rat" in stats[0]:
                baton_str = f" | BATON: {stats[0]['baton_rat']:.3f} (lam={stats[0]['lam_mu']:.3f})"
                log_line += baton_str
            
            # Append aggregated ponder stats
            # avg_exec, avg_k, avg_gap across layers
            if stats:
                k_execs = [s.get('k_exec', 0) for s in stats]
                ks = [s.get('k', 0) for s in stats]
                avg_exec = sum(k_execs) / len(k_execs) if k_execs else 0
                avg_k = sum(ks) / len(ks) if ks else 0
                avg_gap = avg_exec - avg_k
                pl_ratio = (p_loss_val / lm_val) * 100 if lm_val > 0 else 0.0
                
                log_line += f" | PL/LM: {pl_ratio:.2f}% | AVGS: exec={avg_exec:.2f}, k={avg_k:.2f}, gap={avg_gap:.2f}"

            # Append timing stats (User requested dt, tok/sec)
            log_line += f" | dt: {dt:7.2f}ms | tok/sec: {tokens_per_sec:9.2f}"
            
            # Print to console (refresh line)
            print(log_line)
            
            with open(log_name, "a") as f:
                f.write(log_line + "\n")
                
                # Log Layer stats (Detailed)
                # User Fixes:
                # 1. k_exec, expected_k, s_final
                # 2. s_trace (every 50 steps)
                # 5. Ponder components (expected_k_mean, ponder_norm) - implicit in k and PONDER loss
                
                if step % 10 == 0:
                    for i, s in enumerate(stats):
                        # Fix #1 & #5: Explicit k_exec and expected_k
                        # Also printing s_final
                        base = f" L{i}: k_exec={s.get('k_exec',0)}, expected_k={s.get('k',0):.2f}, s_fin={s.get('s_final',0):.3f}"
                        
                        # NEW: Per-token execution stats (ChatGPT alignment)
                        avg_et = s.get('avg_exec_token', 0)
                        p95_et = s.get('p95_exec_token', 0)
                        waste = s.get('wasted_compute', 0)
                        # Hard exec prediction (should match tok_exec)
                        hard_avg = s.get('hard_exec_pred_avg', 0)
                        hard_p95 = s.get('hard_exec_pred_p95', 0)
                        # Execution tail fractions
                        frac_ge2 = s.get('frac_ge2', 0)
                        frac_ge3 = s.get('frac_ge3', 0)
                        frac_ge4 = s.get('frac_ge4', 0)
                        token_exec_ext = f" | tok: avg={avg_et:.2f}, p95={p95_et:.2f} | tail: ≥2:{frac_ge2:.2%}, ≥3:{frac_ge3:.2%}, ≥4:{frac_ge4:.2%}"
                        
                        # Ponder diagnostics
                        ponder_sig = s.get('ponder_signal', 0)
                        exec_fr = s.get('exec_frac', 0)
                        comb_p = s.get('combined_ponder', 0)
                        
                        # Recurrence Diagnostics (FIXED: now properly populated)
                        h_diff = s.get('h_diff', 0.0) # Mean ||gated_dh|| / ||h|| 
                        p_diff = s.get('p_diff', 0.0) # p2 - p1
                        
                        ext = f" | ent={s.get('ent',0):.4f} | p_mu={s.get('p_mean',0):.2f}, p_std={s.get('p_std',0):.3f}, d_rel={s.get('d_rel',0):.3f}"
                        rec_ext = f" | Δh={h_diff:.3f}, Δp={p_diff:.2f}"
                        ponder_ext = f" | ponder={ponder_sig:.3f}, λ_p={lambda_p:.2e}"
                        log_msg = base + token_exec_ext + ext + rec_ext + ponder_ext
                        
                        f.write(log_msg + "\n")
                        print(log_msg)
                        
                        # Fix #2: Log raw survival trajectory (debug mode) every 50 steps
                        if step % 50 == 0:
                            s_trace = s.get('s_trace', [])
                            s_trace_str = "[" + ", ".join([f"{x:.2f}" for x in s_trace]) + "]"
                            trace_msg = f"  L{i} s_trace: {s_trace_str}"
                            f.write(trace_msg + "\n")
                            print(trace_msg)
                
                # Fix #6: verify LR schedule phase every 50 steps
                if step % 50 == 0:
                    # Infer phase
                    if step < 500: phase = "warmup"
                    elif step > 10000: phase = "done" # simplistic
                    else: phase = "cosine" # Assuming notch removed/passed? Wait, notch was removed in previous session? 
                    # Actually, let's just print what we see.
                    lr_msg = f"LR CHECK: ({step}, {lr:.2e}, phase={phase})"
                    f.write(lr_msg + "\n")
                    print(lr_msg)
                
                # NEW: Economic Recurrence Quality (ΔCE with COUPON status)
                if do_credit and logits_1 is not None and nll_1_val is not None:
                    # CE1 - CE2: Positive means the second pass improved the model
                    ce_gain = (nll_1_val - lm_loss).item() 
                    # delta_h was computed in Section 3.5 above

                    rec_quality_msg = f"  RECURRENCE QUALITY: ΔCE={ce_gain:.6f} | ΔH={delta_h:.4f} | COUPON: {reward_val:.6f}"
                    
                    # Fix: Separate labeling for audit vs policy batches
                    if audit_force_2:
                        # Audit batch labeling
                        status = "[AUDIT-POS]" if ce_gain > 0 else "[AUDIT-NEG]"
                    else:
                        # Policy batch labeling  
                        status = "[ECONOMIC WIN]" if ce_gain > 0 else "[TAX ONLY]"
                    
                    # Confidence only makes sense for non-audit AND positive ΔCE
                    conf = " [CONFIDENCE+]" if (ce_gain > 0 and delta_h < 0 and not audit_force_2) else ""
                    rec_quality_msg += f" {status}{conf}"
                            
                    f.write(rec_quality_msg + "\n")
                    print(rec_quality_msg)

            
            # Dashboard JSON sync
            try:
                # Convert stats to JSON-serializable format (handle tensors)
                serializable_stats = []
                for s in stats:
                    s_clean = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in s.items() if k != 'h_1'}
                    serializable_stats.append(s_clean)
                    
                with open("stats.json", "w") as f:
                    json.dump({
                        "step": step, 
                        "lm_loss": lm_loss.detach().item(),  # Detach to break graph reference
                        "total_ponder": total_ponder_loss.detach().item(),  # CRITICAL: was holding graph!
                        "delta_ce": delta_ce.detach().item() if 'delta_ce' in locals() and isinstance(delta_ce, torch.Tensor) else 0.0,
                        "delta_h": delta_h if 'delta_h' in locals() else 0.0,
                        "deadband_active": deadband_active,
                        "deadband_steps_since_trigger": (step - deadband_trigger_step) if deadband_trigger_step else 0,
                        "layers": serializable_stats
                    }, f)
                del serializable_stats  # Explicit cleanup
            except Exception as e:
                print(f"JSON Dump failed: {e}")

        # 5. Mandatory Memory Cleanup
        # Explicitly delete largest tensors to prevent carry-over to next iteration.
        # This is critical on Mac (MPS) where unified memory allocation can be "sticky".
        # We also clear intermediate distributions and slices.
        del outputs, logits_1, total_loss, lm_loss, total_ponder_loss, delta_ce, batch, \
            input_ids, target_ids, stats, inp_seq, tgt_seq, attention_mask
        
        if 'nll_1_val' in locals() and nll_1_val is not None:
             del nll_1_val

        if device.type == "mps":
            try:
                torch.mps.empty_cache()
            except Exception: pass
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        # YES: gc.collect() EVERY step (matching your aggressive cleanup)
        gc.collect()

        # Generation after warmup
        if gen_every > 0 and step >= gen_warmup_steps and (step % gen_every == 0):
            model.eval()
            try:
                # Pass step number for deterministic prompt sampling
                prompts_t = prompt_sampler.sample_token_ids(n=3, step=step)
                prompts_t = [p.to(device) for p in prompts_t]
                
                if prompts_t:
                    print(f"\n{'='*20} GENERATION STEP {step} {'='*20}")
                    with open(log_name, "a") as f:
                        f.write(f"\n{'='*20} GENERATION STEP {step} {'='*20}\n")
                    
                    # Iterate through each prompt and generate sequentially
                    for gi, prompt_t in enumerate(prompts_t):
                        gen_t = sample_generate_single_fast(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_ids=prompt_t,
                            max_new_tokens=128,
                            temperature=0.8,
                            top_p=0.95,
                            max_seq_len=512,
                            force_2=force_2_effective,  # Align generation compute with training
                            step=step,        # Pass step for potential scheduling
                            lambda_p=lambda_p # Pass current lambda_p
                        )

                        prompt_text = tokenizer.decode(prompt_t, skip_special_tokens=False)
                        new_tokens = gen_t[prompt_t.size(0):]
                        new_text = tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens.numel() > 0 else ""

                        # Log to console and file
                        log_msg = f"[GEN {gi+1}]\n  Prompt ({prompt_t.size(0)} tokens):\n"
                        prompt_display = prompt_text[:500] + "..." if len(prompt_text) > 500 else prompt_text
                        for line in prompt_display.split('\n'):
                            log_msg += f"    {line}\n"
                        
                        log_msg += f"  Output ({new_tokens.numel()} tokens):\n"
                        output_display = new_text[:1000] + "..." if len(new_text) > 1000 else new_text
                        for line in output_display.split('\n'):
                            log_msg += f"    {line}\n"
                        log_msg += "\n"
                        
                        print(log_msg)
                        with open(log_name, "a") as f:
                            f.write(log_msg)

                    print(f"{'='*60}\n")
                    with open(log_name, "a") as f:
                        f.write(f"{'='*60}\n\n")
                    
                    # Clean up
                    del prompts_t
                
                gc.collect()
                if device == "mps": # Device check string or object? 'device' is string "mps" or object?
                    # In main(), device = "mps" if ... else "cpu" (line 556)
                    # wait, later line 598 model.to(device).
                    # 'device' variable at line 556 is a STRING.
                    # later line 701 ids = batch[...].to(device) -> works with string.
                    # So device.type is invalid.
                    try:
                        torch.mps.empty_cache()
                    except:
                        pass
                elif device == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Generation failed: {e}")
                import traceback
                traceback.print_exc()
            finally:
                model.eval() # wait, should be train?
                # Ah, originally it was model.train()
                # But wait, Ouroboros training might need explicit reset or something?
                # Standard is model.train()
                model.train()

if __name__ == "__main__": 
    main()


# python new-fourier-test.py --resume checkpoints/checkpoint_step_0001201.pt