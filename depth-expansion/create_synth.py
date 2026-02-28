from datasets import load_dataset, Dataset

# Stream the dataset directly from Hugging Face
stream = load_dataset("PleIAs/SYNTH", streaming=True, split="train")

examples = []
# Increased target to ensure ~20.5M+ tokens for your 10K steps
target_examples = 30000 

for ex in stream:
    lang = ex.get("language")
    if lang and lang.lower() == "en":
        examples.append(ex)
    
    if len(examples) >= target_examples:
        break

print(f"Collected {len(examples)} English examples")

# Convert back to an Arrow-backed dataset for fast training
ds = Dataset.from_list(examples)
ds.save_to_disk("synth_local")
print("Saved to synth_local")