# threshold_optimizer.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
import pickle
from typing import List, Tuple, Dict
import json


class ThresholdOptimizer:
    """
    Train a classifier to determine optimal threshold based on features
    """

    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.optimal_threshold = 0.7  # Default

    def extract_features(self, query: str, candidates: List[Tuple[str, float]]) -> Dict:
        """Extract features for threshold prediction"""
        if not candidates:
            return {}

        scores = [score for _, score in candidates]

        features = {
            # Query features
            'query_length': len(query),
            'query_word_count': len(query.split()),
            'query_has_typo': self._detect_potential_typo(query),

            # Score distribution features
            'max_score': max(scores) if scores else 0,
            'mean_score': np.mean(scores) if scores else 0,
            'std_score': np.std(scores) if scores else 0,
            'score_range': max(scores) - min(scores) if scores else 0,

            # Gap analysis
            'top_score_gap': scores[0] - scores[1] if len(scores) > 1 else 1.0,
            'candidate_count': len(candidates),

            # Score percentiles
            'percentile_90': np.percentile(scores, 90) if scores else 0,
            'percentile_75': np.percentile(scores, 75) if scores else 0,
            'percentile_50': np.percentile(scores, 50) if scores else 0,
        }

        return features

    def _detect_potential_typo(self, text: str) -> int:
        """Simple heuristic to detect potential typos"""
        common_typos = ['teh', 'hte', 'adn', 'tow', 'ot']
        return 1 if any(typo in text.lower() for typo in common_typos) else 0

    def prepare_training_data(self, labeled_data_file: str):
        """
        Prepare training data from labeled examples
        Format: [{"query": "...", "correct_concept": "...", "candidates": [...]}]
        """
        with open(labeled_data_file, 'r') as f:
            labeled_data = json.load(f)

        X = []
        y = []

        for example in labeled_data:
            query = example['query']
            correct = example['correct_concept']
            candidates = example['candidates']  # List of (concept, score) tuples

            # Extract features
            features = self.extract_features(query, candidates)

            # For each possible threshold, check if it would include the correct answer
            for threshold in np.arange(0.5, 0.95, 0.05):
                filtered = [(c, s) for c, s in candidates if s >= threshold]

                # Label: 1 if correct concept is in filtered results, 0 otherwise
                label = 1 if any(c == correct for c, _ in filtered) else 0

                # Add threshold as a feature
                features['threshold'] = threshold

                X.append(list(features.values()))
                y.append(label)

        return np.array(X), np.array(y)

    def train(self, X, y):
        """Train the classifier"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.classifier.fit(X_train, y_train)

        # Find optimal threshold on test set
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx]

        print(f"Model accuracy: {self.classifier.score(X_test, y_test):.3f}")
        print(f"Optimal threshold: {self.optimal_threshold:.3f}")

        return self

    def predict_threshold(self, query: str, candidates: List[Tuple[str, float]]) -> float:
        """Predict optimal threshold for a given query and candidates"""
        features = self.extract_features(query, candidates)

        # Test different thresholds
        best_threshold = 0.7
        best_score = 0

        for threshold in np.arange(0.5, 0.95, 0.05):
            features['threshold'] = threshold
            feature_vector = np.array([list(features.values())])

            # Get probability of this threshold being good
            prob = self.classifier.predict_proba(feature_vector)[0, 1]

            if prob > best_score:
                best_score = prob
                best_threshold = threshold

        return best_threshold

    def save_model(self, filepath: str):
        """Save trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'optimal_threshold': self.optimal_threshold
            }, f)

    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.classifier = data['classifier']
            self.optimal_threshold = data['optimal_threshold']
        return self


# Create training data collection script
def collect_training_data():
    """Helper to collect training data"""
    training_examples = []

    # Example format
    example = {
        "query": "data stucture",
        "correct_concept": "Data Structure",
        "candidates": [
            ("Data Structure", 0.92),
            ("Data Science", 0.65),
            ("Structure", 0.58),
            ("Database Structure", 0.71)
        ]
    }
    training_examples.append(example)

    # Save training data
    with open('threshold_training_data.json', 'w') as f:
        json.dump(training_examples, f, indent=2)


# Integration with semantic enhanced kg utils
def get_adaptive_threshold(query: str, candidates: List[Tuple[str, float]],
                           optimizer: ThresholdOptimizer = None) -> float:
    """Get adaptive threshold based on query and candidates"""
    if optimizer is None:
        return 0.7  # Default

    try:
        return optimizer.predict_threshold(query, candidates)
    except:
        return 0.7  # Fallback to default