# build_dataset.py
import csv
import random
import json
from typing import List, Dict, Set
from sentence_transformers import InputExample
from neo4j import GraphDatabase
import os
from fuzzywuzzy import fuzz
import numpy as np
from itertools import combinations
import re

# Neo4j connection
NEO_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO_USER = os.getenv("NEO4J_USER", "neo4j")
NEO_PW = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PW))

def get_concepts_from_kg() -> List[str]:
    """Extract all concept names from the knowledge graph"""
    query = """
    MATCH (c:Concept)
    WHERE c.name IS NOT NULL
    RETURN DISTINCT c.name as name
    """
    with driver.session() as session:
        result = session.run(query)
        return [record["name"] for record in result if record["name"]]

def get_synonyms_from_kg() -> Dict[str, List[str]]:
    """Extract synonyms from the knowledge graph, or generate basic ones if none exist"""
    query = """
    OPTIONAL MATCH (c1:Concept)-[:IS_SYNONYM_OF]-(c2:Concept)
    WHERE c1.name IS NOT NULL AND c2.name IS NOT NULL
    RETURN c1.name as name1, c2.name as name2
    """
    synonyms = {}
    with driver.session() as session:
        result = session.run(query)
        for record in result:
            if record["name1"] and record["name2"]:
                name1, name2 = record["name1"], record["name2"]
                if name1 not in synonyms:
                    synonyms[name1] = []
                if name2 not in synonyms:
                    synonyms[name2] = []
                synonyms[name1].append(name2)
                synonyms[name2].append(name1)

    # If no synonyms found in DB, generate some basic ones
    if not synonyms:
        concepts = get_concepts_from_kg()
        for concept in concepts:
            # Generate basic variations
            variations = set()
            # Convert camelCase/PascalCase to space-separated
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', concept)
            variations.add(spaced.lower())
            # Remove spaces
            variations.add(concept.replace(' ', ''))
            # Convert spaces to underscores and vice versa
            if ' ' in concept:
                variations.add(concept.replace(' ', '_'))
            if '_' in concept:
                variations.add(concept.replace('_', ' '))

            if variations:
                synonyms[concept] = list(variations)

    return synonyms

def generate_typos(word: str, num_variations: int = 3) -> Set[str]:
    """Generate realistic typos using various strategies"""
    variations = set()

    # Common typo patterns
    patterns = [
        # Swap adjacent letters
        lambda w: ''.join(w[i+1] + w[i] + w[i+2:] if i+1 < len(w) else w for i in range(0, len(w)-1, 2)),
        # Drop vowels randomly
        lambda w: re.sub(r'[aeiou]', '', w, count=random.randint(1, 2)),
        # Double letters
        lambda w: w[:random.randint(1,len(w))] + w[random.randint(0,len(w)-1)] + w[random.randint(1,len(w)):],
        # Replace similar looking characters
        lambda w: w.replace('a','@').replace('i','1').replace('o','0').replace('s','5'),
        # Missing space
        lambda w: w.replace(' ', '') if ' ' in w else w,
        # Extra space
        lambda w: ' '.join(w) if len(w) > 3 else w,
        # Common misspellings
        lambda w: w.replace('ie', 'ei'),
        lambda w: w.replace('ei', 'ie'),
        # Missing letter
        lambda w: w[:random.randint(0,len(w))] + w[random.randint(0,len(w)):],
        # Extra letter
        lambda w: w[:random.randint(0,len(w))] + random.choice('abcdefghijklmnopqrstuvwxyz') + w[random.randint(0,len(w)):],
        # Keyboard proximity errors (common adjacent key substitutions)
        lambda w: w.replace('a', 's').replace('s', 'a'),
        lambda w: w.replace('e', 'r').replace('r', 'e'),
    ]

    # Apply patterns randomly
    attempts = 0
    while len(variations) < num_variations and attempts < 20:
        pattern = random.choice(patterns)
        variation = pattern(word)
        if variation != word and len(variation) > 2 and fuzz.ratio(word, variation) > 60:
            variations.add(variation)
        attempts += 1

    return variations

def create_training_pairs(min_pairs: int = 5000):
    """Create training pairs with concepts, synonyms and typos"""
    # Get concepts and synonyms from KG
    concepts = get_concepts_from_kg()
    print(f"Found {len(concepts)} concepts in the knowledge graph")

    synonyms = get_synonyms_from_kg()
    print(f"Found/generated synonyms for {len(synonyms)} concepts")

    # Generate typos for each concept
    typos = {}
    for concept in concepts:
        typos[concept] = generate_typos(concept, num_variations=5)  # Increased variations
    print(f"Generated typos for {len(typos)} concepts")

    # Create positive pairs
    pos_pairs = []
    for concept in concepts:
        # Concept with its synonyms
        alts = set([concept] + synonyms.get(concept, []))
        # Add typos
        alts.update(typos.get(concept, set()))

        # Create pairs
        for pair in combinations(alts, 2):
            pos_pairs.append(InputExample(texts=list(pair), label=1.0))

    print(f"Generated {len(pos_pairs)} positive pairs")

    # Create negative pairs (sample from concepts)
    neg_pairs = []
    num_neg_pairs = max(min_pairs - len(pos_pairs), len(pos_pairs))  # At least equal number of negatives

    while len(neg_pairs) < num_neg_pairs:
        a, b = random.sample(concepts, 2)
        # Ensure they're not synonyms
        if b not in synonyms.get(a, []):
            # Add some hard negatives (similar but different concepts)
            similarity = fuzz.ratio(a.lower(), b.lower()) / 100.0
            if similarity > 0.5:  # Add more weight to hard negatives
                neg_pairs.extend([InputExample(texts=[a, b], label=0.0)] * 2)
            else:
                neg_pairs.append(InputExample(texts=[a, b], label=0.0))

    print(f"Generated {len(neg_pairs)} negative pairs")

    # Combine and shuffle
    all_examples = pos_pairs + neg_pairs
    random.shuffle(all_examples)

    return all_examples

def save_dataset(examples: List[InputExample], filename: str):
    """Save dataset to JSONL format"""
    with open(filename, "w") as f:
        for ex in examples:
            f.write(json.dumps({
                "text1": ex.texts[0],
                "text2": ex.texts[1],
                "label": ex.label,
                "similarity": fuzz.ratio(ex.texts[0].lower(), ex.texts[1].lower()) / 100.0
            }) + "\n")

if __name__ == "__main__":
    # Create training pairs
    print("Generating training pairs...")
    training_examples = create_training_pairs(min_pairs=5000)

    # Save dataset
    print(f"Generated {len(training_examples)} pairs")
    print(f"Positive pairs: {sum(1 for ex in training_examples if ex.label > 0.5)}")
    print(f"Negative pairs: {sum(1 for ex in training_examples if ex.label <= 0.5)}")

    save_dataset(training_examples, "train_pairs.jsonl")
    print("Dataset saved to train_pairs.jsonl")

