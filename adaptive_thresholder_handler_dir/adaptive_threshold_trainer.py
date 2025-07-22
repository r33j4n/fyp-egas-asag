# adaptive_threshold_trainer.py
# ────────────────────────────────────────────────────────────────────
"""
Train a classifier that decides whether the correct concept will be
retrieved at a given similarity-score threshold.

▪ Builds synthetic data from EVERY concept found in triplets_data.json
▪ Generates many realistic typo / abbreviation variations
▪ Shows tqdm progress bars while the dataset is being built
▪ Produces all original visualisations & model checkpoints
"""
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import json, random, unicodedata, itertools, time, os, pickle
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm                      # progress-bar
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve, auc, f1_score
)

# ────────────────────────────────────────────────────────────────────
class AdaptiveThresholdTrainer:
    """
    End-to-end training helper:
      • builds query-patterns automatically from KG triples
      • generates a large synthetic dataset with typos & synonyms
      • trains a RandomForest and visualises performance
      • saves checkpoints
    """

    # ───────────── INIT ─────────────
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.training_history = {
            "epochs": [], "train_scores": [],
            "val_scores": [], "best_threshold": 0.7
        }

        # Build query-patterns **once**
        self.query_patterns = self._load_query_patterns("triplets_data.json")

    # ───────────── QUERY-PATTERN BUILDER ─────────────
    def _load_query_patterns(self,
                             triples_path: str,
                             max_variations_per_concept: int = 6
    ) -> list[tuple[str, list[str], str]]:
        """
        Scan the triples file and create:
        (base_query, [variation1, variation2, …], correct_concept)
        """
        print(f"🔍 Loading concepts from {triples_path} …")

        with open(triples_path, "r") as fh:
            triples_json = json.load(fh)

        concepts: set[str] = set()
        for rec in triples_json:
            for h, _, t in rec["triples"]:
                concepts.add(h.strip())
                concepts.add(t.strip())

        def make_variations(term: str) -> list[str]:
            """Very simple typo / abbrev maker."""
            base = term.strip()
            var = {
                base.lower(), base.upper(), base.title(),
                base.replace(" ", ""),              # no spaces
                base.replace("-", " "),             # dash→space
            }
            if len(base) > 3:
                var.add(base[:-1])                  # missing last char
                var.add(base[1:])                   # missing first char
                if len(base) > 4:                   # swap two middle chars
                    mid = len(base)//2
                    trans = list(base)
                    trans[mid], trans[mid+1] = trans[mid+1], trans[mid]
                    var.add("".join(trans))
            if " " in base:                         # abbreviation
                abbr = "".join(w[0] for w in base.split())
                var.update({abbr.lower(), abbr.upper()})

            # strip accents, drop duplicates & base itself
            var = {unicodedata.normalize("NFKD", v).encode("ascii", "ignore")
                     .decode() for v in var}
            var.discard(base)
            return sorted(var)[:max_variations_per_concept]

        patterns: list[tuple[str, list[str], str]] = []
        for concept in concepts:
            variations = make_variations(concept)
            if variations:
                patterns.append((concept, variations, concept))

        print(f"⚙️  Built {len(patterns):,} query-patterns "
              f"from {len(concepts):,} unique concepts.")
        return patterns

    # ───────────── SYNTHETIC-DATA GENERATOR ─────────────
    def generate_synthetic_dataset(self,n_samples: int = 10000
    ) -> pd.DataFrame:
        print(f"Generating {n_samples:,} synthetic training samples …")
        data_rows = []

        all_concept_names = [p[2] for p in self.query_patterns]

        for _ in tqdm(range(n_samples), unit="sample"):
            base_query, variations, correct = random.choice(self.query_patterns)
            query = random.choice(variations) if random.random() < .7 else base_query

            # Candidate list with similarity scores
            candidates = [(correct, random.uniform(.75, .95))]
            distractors = [c for c in all_concept_names if c != correct]
            for d in random.sample(distractors,
                                   k=min(len(distractors),
                                         random.randint(3, 8))):
                candidates.append((d, random.uniform(.4, .8)))
            candidates.sort(key=lambda x: x[1], reverse=True)

            features = self.extract_features(query, candidates)

            for thr in np.arange(.5, .95, .05):
                row = features | {
                    "threshold": thr,
                    "label": int(any(c == correct and s >= thr
                                     for c, s in candidates)),
                    "query": query,
                    "correct_concept": correct,
                    "n_candidates": len(candidates)
                }
                data_rows.append(row)

        df = pd.DataFrame(data_rows)
        print(f"✅  Synthetic dataset shape: {df.shape}")
        return df

    # ───────────── FEATURE EXTRACTION ─────────────
    def extract_features(self, query: str,
                         candidates: List[Tuple[str, float]]) -> Dict:
        scores = [s for _, s in candidates] or [0]
        probs  = np.array(scores) / np.sum(scores)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return {
            "query_length": len(query),
            "query_word_count": len(query.split()),
            "max_score": max(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "score_range": max(scores) - min(scores),
            "top_score_gap": scores[0]-scores[1] if len(scores) > 1 else 1,
            "candidate_count": len(scores),
            "percentile_90": np.percentile(scores, 90),
            "percentile_75": np.percentile(scores, 75),
            "percentile_50": np.percentile(scores, 50),
            "score_entropy":  entropy,
        }

    # ───────────── TRAIN / VISUALISE / CHECKPOINT  ─────────────
    def train_with_visualization(self, df: pd.DataFrame,
                                 test_size: float = .2,
                                 show_plots: bool = True) -> Dict:
        start = time.time()
        feat_cols = [c for c in df.columns
                     if c not in ("label", "query",
                                  "correct_concept", "n_candidates")]
        X = df[feat_cols].values
        y = df["label"].values

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size,
            stratify=y, random_state=42
        )
        self.classifier.fit(Xtr, ytr)

        train_acc = self.classifier.score(Xtr, ytr)
        test_acc  = self.classifier.score(Xte, yte)
        cv_scores = cross_val_score(self.classifier, Xtr, ytr, cv=5)

        print(f"\nTrain acc: {train_acc:.4f}  |  "
              f"Test acc: {test_acc:.4f}  |  "
              f"CV mean: {cv_scores.mean():.4f}")

        ypred = self.classifier.predict(Xte)
        yprob = self.classifier.predict_proba(Xte)[:, 1]

        metrics = {
            "train_acc": train_acc, "test_acc": test_acc,
            "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
            "cls_report": classification_report(yte, ypred, digits=3)
        }

        if show_plots:
            self._create_visualizations(yte, ypred, yprob,
                                        feat_cols, metrics)

        self._save_checkpoint(metrics)
        print(f"⏱️  Training + plotting took {time.time()-start:.1f}s")
        return metrics

    # ───────────── VISUALISATIONS ─────────────
    def _create_visualizations(self, y_test, y_pred, y_pred_proba,
                               feature_cols, metrics):
        """Comprehensive set of plots (unchanged from original)."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Adaptive Threshold Model Performance', fontsize=16)

        # Confusion-matrix
        ax = axes[0, 0]
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        # ROC
        ax = axes[0, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")

        # Precision-Recall
        ax = axes[0, 2]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ax.plot(recall, precision, color='blue', lw=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.grid(alpha=.3)

        # Feature-importance
        ax = axes[1, 0]
        feat_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        ax.barh(feat_imp['feature'], feat_imp['importance'])
        ax.set_xlabel('Importance')
        ax.set_title('Top-10 Feature Importances')
        ax.invert_yaxis()

        # Optimal-threshold curve
        ax = axes[1, 1]
        f1_scores = []
        thr_rng = np.linspace(0.1, 0.9, 50)
        for thr in thr_rng:
            f1_scores.append(f1_score(y_test, (y_pred_proba >= thr).astype(int)))
        opt_thr = thr_rng[int(np.argmax(f1_scores))]
        ax.plot(thr_rng, f1_scores, 'b-', lw=2)
        ax.axvline(opt_thr, color='r', ls='--', label=f'Optimal={opt_thr:.3f}')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1')
        ax.set_title('F1 vs Threshold')
        ax.legend()
        ax.grid(alpha=.3)
        print(f"⭐ Optimal threshold ≈ {opt_thr:.3f}")
        self.training_history["best_threshold"] = float(opt_thr)

        # Score distribution
        ax = axes[1, 2]
        ax.hist(y_pred_proba[y_test == 1], bins=30, alpha=.5,
                label='Positive', density=True)
        ax.hist(y_pred_proba[y_test == 0], bins=30, alpha=.5,
                label='Negative', density=True)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Density')
        ax.set_title('Score Distribution')
        ax.legend()

        plt.tight_layout()
        ts = time.strftime("%Y%m%d_%H%M%S")
        plt.savefig(self.checkpoint_dir / f"training_plots_{ts}.png",
                    dpi=300, bbox_inches='tight')
        plt.show()

    # ───────────── CHECKPOINT ─────────────
    def _save_checkpoint(self, metrics: Dict):
        ts = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = self.checkpoint_dir / f"model_checkpoint_{ts}.pkl"
        with open(ckpt_path, "wb") as fh:
            pickle.dump({
                "classifier": self.classifier,
                "metrics": metrics,
                "training_history": self.training_history,
                "timestamp": ts
            }, fh)
        print(f"💾  Saved checkpoint → {ckpt_path}")