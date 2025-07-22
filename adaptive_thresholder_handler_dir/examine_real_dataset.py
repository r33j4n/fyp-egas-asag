# examine_real_dataset.py
import json
import numpy as np
import pandas as pd  # optional – useful for quick inspection

# 1️⃣ Load the dataset
DATA_PATH = "threshold_training_data.json"  # ← adjust if needed
with open(DATA_PATH, "r") as f:
    real_data = json.load(f)

print(f"Total examples in real dataset: {len(real_data):,}")

# 2️⃣ Show one example
print("\nSample example:")
print(json.dumps(real_data[0], indent=2))

# 3️⃣ Collect **all** candidate scores
scores = []
for example in real_data:
    for cand in example["candidates"]:
        scores.append(cand["score"])

# 4️⃣ Stats
print("\nScore statistics:")
print(f"Min score : {np.min(scores):.3f}")
print(f"Max score : {np.max(scores):.3f}")
print(f"Mean score: {np.mean(scores):.3f}")
print(f"Std  dev  : {np.std(scores):.3f}")

# 5️⃣ (Optional) load into DataFrame for quick queries
df = pd.DataFrame(real_data)
print("\nDataFrame preview:")
print(df.head())