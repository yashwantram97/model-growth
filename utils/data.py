"""Data loading utilities."""

import torch
import random
from typing import Tuple


class TinyStoriesDataset:
    """
    Loads a slice of TinyStories, tokenizes every story with GPT-2,
    concatenates them into one long token stream (separated by
    <|endoftext|>), and serves random windows of seq_len for training.

    WHY a single long stream?
      Individual stories are short (often < 256 tokens). Padding every
      story to seq_len wastes compute on <pad> tokens. Concatenating
      into one stream and slicing random windows is the standard trick
      used by GPT-2/3/LLaMA training. The <|endoftext|> token acts as
      a boundary marker so attention doesn't bleed across stories.

    LAZY LOADING:
      tokenize() must be called once before get_batch(). It downloads
      the dataset (cached by HuggingFace after the first run) and
      tokenizes the stories. On subsequent runs the HF cache is reused
      so the download is skipped.
    """
    def __init__(self, num_stories: int = 50_000):
        self.num_stories = num_stories
        self.tokens = None  # filled by tokenize()
        self.tokenizer = None

    def tokenize(self):
        """Download dataset + tokenizer, tokenize, concatenate into one stream."""
        from transformers import AutoTokenizer
        from datasets import load_dataset

        print("[dataset] Loading tokenizer (gpt2) …")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT-2 has no pad token by default; set eos as pad for safety
        self.tokenizer.pad_token = self.tokenizer.eos_token
        eot = self.tokenizer.eos_token_id  # <|endoftext|>

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
        print(f"[dataset] Tokenizing {self.num_stories:,} stories …")
        all_tokens = []
        count = 0
        for example in ds:
            text = example["text"].strip()
            if not text:
                continue
            # Tokenize one story, append <|endoftext|> as separator
            ids = self.tokenizer.encode(text)
            all_tokens.extend(ids)
            all_tokens.append(eot)
            count += 1
            if count >= self.num_stories:
                break

        self.tokens = all_tokens
        print(f"[dataset] Done. {len(self.tokens):,} total tokens "
              f"from {count:,} stories.")

    def get_batch(self, batch_size: int, seq_len: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `batch_size` random windows of length `seq_len` from the
        token stream. Returns (input_ids, labels) where labels is the
        stream shifted left by 1 (next-token prediction).

        input_ids : (B, seq_len)
        labels    : (B, seq_len)   — labels[i] = input_ids[i] shifted left by 1
        """
        assert self.tokens is not None, \
            "Call tokenize() before get_batch()"

        L = len(self.tokens)
        input_ids = []
        labels = []

        for _ in range(batch_size):
            # Random start position (need seq_len + 1 tokens: input + label)
            start = random.randint(0, L - seq_len - 1)
            chunk = self.tokens[start: start + seq_len + 1]
            input_ids.append(chunk[:-1])  # positions 0 … seq_len-1
            labels.append(chunk[1:])      # positions 1 … seq_len (shifted)

        return (torch.tensor(input_ids, dtype=torch.long, device=device),
                torch.tensor(labels, dtype=torch.long, device=device))
