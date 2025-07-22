# train_adaptive_threshold_with_progress.py
import json
import pickle

import pandas as pd
import numpy as np
from adaptive_threshold_trainer import AdaptiveThresholdTrainer
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

print("=" * 60)
print("ADAPTIVE THRESHOLD TRAINING")
print("=" * 60)

# Load your real dataset
print("\n📁 Loading real dataset...")
start_time = time.time()
with open('threshold_training_data.json', 'r') as f:
    real_data = json.load(f)
print(f"✅ Loaded {len(real_data):,} examples in {time.time() - start_time:.1f}s")

# Initialize trainer
trainer = AdaptiveThresholdTrainer(checkpoint_dir="./checkpoints")


# Convert real data to training format
def convert_real_to_training_format(real_data):
    """Convert your real dataset to the format needed for training"""
    training_rows = []

    print("\n🔄 Converting real dataset to training format...")
    # Add progress bar for conversion
    for example in tqdm(real_data, desc="Processing examples"):
        query = example['query']
        correct_concept = example['correct_concept']
        candidates = [(c['concept'], c['score']) for c in example['candidates']]

        # Extract features from this example
        features = trainer.extract_features(query, candidates)

        # Create training examples for different thresholds
        for threshold in np.arange(0.3, 0.95, 0.05):
            row = features.copy()
            row['threshold'] = threshold

            # Check if correct concept would be included at this threshold
            filtered = [c for c, s in candidates if s >= threshold]
            row['label'] = 1 if correct_concept in filtered else 0

            # Metadata
            row['query'] = query
            row['correct_concept'] = correct_concept
            row['n_candidates'] = len(candidates)

            training_rows.append(row)

    return pd.DataFrame(training_rows)


# Convert real data
print("\n📊 Converting dataset...")
convert_start = time.time()
real_df = convert_real_to_training_format(real_data)
print(f"✅ Conversion complete in {time.time() - convert_start:.1f}s")
print(f"📈 Real dataset shape: {real_df.shape}")

# Generate synthetic data to augment training
print("\n🎲 Generating synthetic data to augment training...")
synthetic_start = time.time()
synthetic_df = trainer.generate_synthetic_dataset(n_samples=10000)
metrics = trainer.train_with_visualization(synthetic_df)
print(f"✅ Synthetic data generated in {time.time() - synthetic_start:.1f}s")

# Combine datasets
combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
print(f"\n📊 Combined dataset shape: {combined_df.shape}")
print(f"   - Real data: {len(real_df):,} rows")
print(f"   - Synthetic data: {len(synthetic_df):,} rows")
print(f"   - Total: {len(combined_df):,} rows")

# Save combined dataset
print("\n💾 Saving combined dataset...")
combined_df.to_csv('combined_training_data.csv', index=False)
print("✅ Dataset saved to combined_training_data.csv")

# Estimate training time
estimated_time = len(combined_df) * 0.001  # Rough estimate: 1ms per row
print(f"\n⏱️  Estimated training time: {estimated_time:.0f}-{estimated_time * 2:.0f} seconds")

# Train the model with progress tracking
print("\n" + "=" * 60)
print("🚀 STARTING MODEL TRAINING")
print("=" * 60)
training_start = time.time()

try:
    metrics = trainer.train_with_visualization(combined_df, show_plots=True)

    # Save the final model
    print("\n💾 Saving trained model...")
    with open('adaptive_threshold_model.pkl', 'wb') as f:
        pickle.dump(trainer, f)

    total_time = time.time() - training_start
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"⏱️  Total training time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    print(f"📊 Test Accuracy: {metrics['test_acc']:.4f}")
    print(f"💾 Model saved to: adaptive_threshold_model.pkl")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ Error during training: {e}")
    raise

