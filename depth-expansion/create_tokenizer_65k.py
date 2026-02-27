#!/usr/bin/env python3
"""
create_tokenizer_65k.py
Truncates the TSAI 131K BPE tokenizer to exactly 65536 tokens (2^16).

Keeps:
  - All vocab entries with id < 65536
  - Only merges whose result token is in the kept vocab
  - Added special tokens with id < 65536

Run once:
    python create_tokenizer_65k.py

Output: tokenizer_65k/  (drop-in replacement, load with AutoTokenizer)
"""

import json
import os
import shutil

TARGET    = 65536
SRC_DIR   = "tokenizer"
DST_DIR   = "tokenizer_65k"

print(f"Loading tokenizer from {SRC_DIR}/tokenizer.json ...")
with open(os.path.join(SRC_DIR, "tokenizer.json"), encoding="utf-8") as f:
    data = json.load(f)

vocab   = data["model"]["vocab"]   # token_str -> id
merges  = data["model"]["merges"]  # list of "a b" strings

# ── 1. Truncate vocab ────────────────────────────────────────────────────────
new_vocab   = {k: v for k, v in vocab.items() if v < TARGET}
kept_tokens = set(new_vocab.keys())
removed     = {k for k in vocab if k not in kept_tokens}
print(f"  Vocab: {len(vocab):,} → {len(new_vocab):,}  (removed {len(removed):,})")

# ── 2. Truncate merges ───────────────────────────────────────────────────────
# For BPE, each merge "a b" creates token "ab" (byte-level BPE concatenates).
# Keep a merge only if its result is still in the kept vocab.
new_merges = []
skipped    = 0
for merge in merges:
    if isinstance(merge, list):
        a, b = merge
    else:
        a, b = merge.split(" ", 1)
    result = a + b
    if result in kept_tokens:
        new_merges.append(merge)
    else:
        skipped += 1
print(f"  Merges: {len(merges):,} → {len(new_merges):,}  (removed {skipped:,})")

# ── 3. Truncate added_tokens ─────────────────────────────────────────────────
added = data.get("added_tokens", [])
new_added = [t for t in added if t["id"] < TARGET]
print(f"  Added tokens: {len(added)} → {len(new_added)}")

# ── 4. Write output ──────────────────────────────────────────────────────────
data["model"]["vocab"]  = new_vocab
data["model"]["merges"] = new_merges
data["added_tokens"]    = new_added

os.makedirs(DST_DIR, exist_ok=True)
out_path = os.path.join(DST_DIR, "tokenizer.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"  Saved: {out_path}")

# Copy tokenizer_config.json and special_tokens_map.json unchanged
for fname in ("tokenizer_config.json", "special_tokens_map.json"):
    src = os.path.join(SRC_DIR, fname)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(DST_DIR, fname))
        print(f"  Copied: {fname}")

# ── 5. Verify with HuggingFace ───────────────────────────────────────────────
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DST_DIR)
    print(f"\nVerification:")
    print(f"  tokenizer.vocab_size = {tok.vocab_size:,}")
    print(f"  len(tokenizer)       = {len(tok):,}")
    test = tok.encode("Hello world")
    assert all(i < TARGET for i in test), f"ID out of range: {[i for i in test if i >= TARGET]}"
    print(f"  Encode test:         {test}  ✓")
    print(f"\nDone — tokenizer_65k/ ready.")
except Exception as e:
    print(f"\nWarning during verification: {e}")
    print("tokenizer_65k/ was written; check manually before use.")
