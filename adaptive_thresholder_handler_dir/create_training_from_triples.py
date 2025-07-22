# create_training_from_triples.py
def create_training_from_triples(triples_data):
    """Create training examples from your triples"""
    from semantic_enhanced_kg_utils import SemanticGraphQueryBuilder

    training_examples = []

    # Extract all unique concepts from triples
    all_concepts = set()
    for item in triples_data:
        for triple in item['triples']:
            all_concepts.add(triple[0])  # Subject
            all_concepts.add(triple[2])  # Object

    # Create query variations
    for concept in all_concepts:
        # Skip very short concepts
        if len(concept) < 3:
            continue

        variations = [
            concept.lower(),
            concept.upper(),
            concept.replace(" ", ""),
            concept.replace("_", " "),
            concept[:-1] if len(concept) > 4 else concept,  # Typo
        ]

        for query in variations:
            # Get candidates
            candidates = SemanticGraphQueryBuilder.find_similar_concepts(
                query, threshold=0.3
            )

            if candidates and len(candidates) > 2:
                training_examples.append({
                    "query": query,
                    "correct_concept": concept,
                    "candidates": candidates
                })

    return training_examples