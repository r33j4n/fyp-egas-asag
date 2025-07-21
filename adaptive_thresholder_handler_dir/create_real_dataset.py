# create_real_dataset.py  (instrumented)
import json, os, logging
from neo4j import GraphDatabase
from tqdm import tqdm                      # progress-bar

# ─── Neo4j connection ───────────────────────────────────────────────────────────
NEO_URI = os.getenv("NEO4J_URI",     "bolt://localhost:7687")
NEO_USER = os.getenv("NEO4J_USER",   "neo4j")
NEO_PW   = os.getenv("NEO4J_PASSWORD", "password")
driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PW))

# ─── Verbose Neo4j driver logs? (optional) ──────────────────────────────────────
# logging.basicConfig(level=logging.INFO)
# logging.getLogger("neo4j").setLevel(logging.DEBUG)

# ─── Your helper import (unchanged) ─────────────────────────────────────────────
from semantic_enhanced_kg_utils import SemanticGraphQueryBuilder


def create_dataset_from_kg(
        neo4j_driver,
        output_file: str,
        typo_variations: bool = True,
        threshold: float = 0.30
    ):
    """
    Build a training JSON of   {query, correct_concept, candidates}
    and show a live progress bar. Returns the list.
    """
    all_concepts = SemanticGraphQueryBuilder.get_all_concepts()
    total_variants = len(all_concepts) * (5 if typo_variations else 1)

    training_data = []
    bar = tqdm(total=total_variants, unit="var", desc="Generating pairs")

    for concept in all_concepts:
        # ── 1. Build query variants ────────────────────────────────────────────
        variations = [concept] if not typo_variations else [
            concept.lower(),                         # lowercase
            concept.upper(),                         # UPPERCASE
            concept.replace(" ", ""),                # no spaces
            concept[:-1] if len(concept) > 3 else concept,  # drop last char
            concept[1:]  if len(concept) > 3 else concept,  # drop first char
        ]

        # ── 2. For every variation retrieve semantic neighbours ───────────────
        for query in variations:
            candidates = SemanticGraphQueryBuilder.find_similar_concepts(
                query, threshold=threshold
            )

            training_data.append({
                "query": query,
                "correct_concept": concept,
                "candidates": [
                    {"concept": c, "score": round(s, 3)} for c, s in candidates
                ]
            })
            bar.update(1)               # advance progress-bar

    bar.close()

    # ── 3. Persist JSON ────────────────────────────────────────────────────────
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"✅  Saved {len(training_data):,} rows ➜  {output_file}")
    return training_data


# ─── Run the builder ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_dataset_from_kg(driver, "threshold_training_data.json")