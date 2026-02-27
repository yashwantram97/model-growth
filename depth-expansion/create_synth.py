from datasets import load_dataset, Dataset

stream = load_dataset("PleIAs/SYNTH", streaming=True, split="train")

examples = []
for ex in stream:
    lang = ex.get("language")
    if lang and lang.lower() == "en":
        examples.append(ex)
    if len(examples) >= 5000:
        break

print(f"Collected {len(examples)} English examples")

ds = Dataset.from_list(examples)
ds.save_to_disk("synth_local")
print("Saved to synth_local/")
