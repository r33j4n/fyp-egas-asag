# examine_real_dataset.py
import json
import pandas as pd
import numpy as np

# Load the dataset you created
with open('threshold_training_data.json', 'r') as f:
    real_data = json.load(f)

print(f"Total examples in real dataset: {len(real_data)}")
print("\nSample example:")
print(json.dumps(real_data[0], indent=2))

# Analyze the dataset
total_candidates = []
for example in real_data:
    candidates = example['candidates']
    total_candidates.extend([score for _, score in candidates])

print(f"\nScore statistics:")
print(f"Min score: {min(total_candidates):.3f}")
print(f"Max score: {max(total_candidates):.3f}")
print(f"Mean score: {np.mean(total_candidates):.3f}")
print(f"Std score: {np.std(total_candidates):.3f}")