# progress_monitor.py
import time
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd


class ProgressMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.history = {
            'epoch': [],
            'train_acc': [],
            'val_acc': [],
            'time': []
        }

    def update(self, epoch, train_acc, val_acc):
        """Update training progress"""
        elapsed = time.time() - self.start_time
        self.history['epoch'].append(epoch)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['time'].append(elapsed)

        # Clear output and show progress
        clear_output(wait=True)

        # Text progress
        print(f"Epoch {epoch} | Time: {elapsed:.1f}s")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print("=" * 50)

        # Plot progress
        if len(self.history['epoch']) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Accuracy plot
            ax1.plot(self.history['epoch'], self.history['train_acc'], 'b-', label='Train')
            ax1.plot(self.history['epoch'], self.history['val_acc'], 'r-', label='Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Training Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Time plot
            ax2.plot(self.history['epoch'], self.history['time'], 'g-')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Time (seconds)')
            ax2.set_title('Training Time')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()


# Enhanced trainer with progress monitoring
class MonitoredAdaptiveThresholdTrainer(AdaptiveThresholdTrainer):
    def train_with_progress(self, df, epochs=10, batch_size=1000):
        """Train with real-time progress monitoring"""
        monitor = ProgressMonitor()

        # Prepare data
        feature_cols = [col for col in df.columns if col not in
                        ['label', 'query', 'correct_concept', 'n_candidates']]
        X = df[feature_cols].values
        y = df['label'].values

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Simulated epochs for progress (RandomForest doesn't have epochs)
        # In practice, you might use incremental learning or partial_fit
        for epoch in range(epochs):
            # Simulate training on batches
            indices = np.random.choice(len(X_train), size=batch_size, replace=False)
            X_batch = X_train[indices]
            y_batch = y_train[indices]

            # Train on batch (for real epochs, use models that support it)
            if epoch == 0:
                self.classifier.fit(X_batch, y_batch)
            else:
                # For RandomForest, we'll just evaluate
                pass

            # Calculate accuracies
            train_acc = self.classifier.score(X_train, y_train)
            val_acc = self.classifier.score(X_val, y_val)

            # Update progress
            monitor.update(epoch + 1, train_acc, val_acc)

            # Small delay to see progress
            time.sleep(0.5)

        return monitor.history