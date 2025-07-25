# semantic_enhanced_kg_utils.py
import pickle

from neo4j import GraphDatabase
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import jellyfish  # For fuzzy string matching
from functools import lru_cache

NEO_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO_USER = os.getenv("NEO4J_USER", "neo4j")
NEO_PW = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PW))

# Initialize sentence transformer for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load adaptive threshold model if available
ADAPTIVE_MODEL = None
try:
    with open('adaptive_thresholder_handler_dir/adaptive_threshold_model.pkl', 'rb') as f:
        ADAPTIVE_MODEL = pickle.load(f)
    print("✅ Adaptive threshold model loaded successfully")
except:
    print("⚠️ Adaptive threshold model not found, using fixed threshold")


class SemanticGraphQueryBuilder:
    """Enhanced query builder with semantic understanding"""

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_all_concepts() -> List[str]:
        """Get all concept names from the graph"""
        query = """
        MATCH (c:Concept)
        RETURN DISTINCT c.name as name
        """
        with driver.session() as session:
            result = session.run(query)
            return [record["name"] for record in result if record["name"]]

    @staticmethod
    @lru_cache(maxsize=1000)
    def get_all_relationship_types() -> List[str]:
        """Get all relationship types from the graph"""
        query = """
        MATCH ()-[r]->()
        RETURN DISTINCT type(r) as rel_type
        """
        with driver.session() as session:
            result = session.run(query)
            return [record["rel_type"] for record in result]

    @staticmethod
    def find_similar_concepts(query_concept: str, threshold: float = 0.7,
                              use_adaptive: bool = False) -> List[Tuple[str, float]]:
        """
        Find semantically similar concepts using embeddings and fuzzy matching
        NEW: use_adaptive parameter to enable adaptive thresholding
        """
        all_concepts = SemanticGraphQueryBuilder.get_all_concepts()

        if not all_concepts:
            return []

        # Get embeddings
        query_embedding = semantic_model.encode(query_concept.lower())
        concept_embeddings = semantic_model.encode([c.lower() for c in all_concepts])

        # Calculate cosine similarities
        similarities = np.dot(concept_embeddings, query_embedding) / (
                np.linalg.norm(concept_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        similarities = similarities.astype(float)

        # Calculate fuzzy string matching scores
        fuzzy_scores = [
            jellyfish.jaro_winkler_similarity(query_concept.lower(), concept.lower())
            for concept in all_concepts
        ]

        # Combine semantic and fuzzy scores
        combined_scores = [
            float(0.7 * sem_score + 0.3 * fuzzy_score)
            for sem_score, fuzzy_score in zip(similarities, fuzzy_scores)
        ]

        # Create initial candidates list
        all_candidates = [
            (concept, float(score))
            for concept, score in zip(all_concepts, combined_scores)
        ]

        # Sort by score
        all_candidates.sort(key=lambda x: x[1], reverse=True)

        # ADAPTIVE THRESHOLD LOGIC
        if use_adaptive and ADAPTIVE_MODEL is not None:
            # Use adaptive threshold
            final_threshold = _get_adaptive_threshold(query_concept, all_candidates)
            print(f"🎯 Adaptive threshold for '{query_concept}': {final_threshold:.3f}")
        else:
            # Use fixed threshold
            final_threshold = threshold
            if use_adaptive:
                print(f"⚠️ Adaptive model not available, using fixed threshold: {final_threshold:.3f}")

        # Filter results based on final threshold
        results = [
            (concept, score)
            for concept, score in all_candidates
            if score >= final_threshold
        ]

        return results

    @staticmethod
    def find_similar_relationships(query_rels: List[str]) -> List[str]:
        """Find semantically similar relationship types"""
        all_rels = SemanticGraphQueryBuilder.get_all_relationship_types()

        if not all_rels or not query_rels:
            return query_rels

        # Create a mapping of similar relationships
        relationship_groups = {
            "hierarchical": ["IS_A", "SUBCLASS_OF", "INSTANCE_OF", "TYPE_OF", "KIND_OF", "EXTENDS", "INHERITS"],
            "compositional": ["HAS", "CONTAINS", "PART_OF", "HAS_COMPONENT", "COMPRISES", "CONSISTS_OF", "INCLUDES"],
            "usage": ["USES", "USED_IN", "APPLIED_TO", "IMPLEMENTS", "UTILIZED_BY", "EMPLOYED_IN"],
            "example": ["EXAMPLE_OF", "INSTANCE", "CASE_OF", "DEMONSTRATION_OF"],
            "dependency": ["DEPENDS_ON", "REQUIRES", "NEEDS", "PREREQUISITE_OF"],
        }

        # Find which group each query relationship belongs to
        similar_rels = set(query_rels)

        for query_rel in query_rels:
            query_rel_upper = query_rel.upper()
            for group, members in relationship_groups.items():
                if query_rel_upper in members:
                    # Add all members of the same group
                    similar_rels.update(members)
                    break

            # Also check semantic similarity with existing relationships
            if query_rel_upper not in all_rels:
                query_embedding = semantic_model.encode(query_rel.lower())
                rel_embeddings = semantic_model.encode([r.lower() for r in all_rels])

                similarities = np.dot(rel_embeddings, query_embedding) / (
                        np.linalg.norm(rel_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )

                # Add relationships with high similarity
                for rel, sim in zip(all_rels, similarities):
                    if sim > 0.8:
                        similar_rels.add(rel)

        return list(similar_rels)

    @staticmethod
    def get_semantic_subgraph(concept: str, hops: int, limit: int,
                              relationship_filter: List[str] = None) -> str:
        """Get subgraph with semantic understanding of relationships"""

        # If relationship filter provided, expand it semantically
        if relationship_filter:
            expanded_rels = SemanticGraphQueryBuilder.find_similar_relationships(relationship_filter)
            rel_filter = "|".join(expanded_rels)
        else:
            rel_filter = ">"  # All relationships

        return f"""
        MATCH (c:Concept {{name: $name}})
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {hops},
            limit: {limit},
            relationshipFilter: '{rel_filter}'
        }}) YIELD nodes, relationships

        RETURN {{
            root_node: c,
            nodes: [node IN nodes | {{
                id: id(node),
                labels: labels(node),
                properties: properties(node)
            }}],
            relationships: [rel IN relationships | {{
                id: id(rel),
                type: type(rel),
                start_node: id(startNode(rel)),
                end_node: id(endNode(rel)),
                properties: properties(rel)
            }}],
            stats: {{
                node_count: size(nodes),
                relationship_count: size(relationships),
                max_hops: {hops},
                relationship_types: [rel IN relationships | type(rel)] 
            }}
        }} AS subgraph
        """


def get_subgraph_semantic(concept: str, hops: int, limit: int,
                          query_type: str = "neighborhood",
                          use_semantic_matching: bool = True,
                          use_adaptive_threshold: bool = False) -> Optional[Dict]:
    """
    Enhanced subgraph retrieval with semantic concept matching
    NEW: use_adaptive_threshold parameter
    """
    # Validate parameters
    hops = max(1, min(hops, 10))
    limit = max(1, min(limit, 100))

    # First, try exact match
    exact_result = _try_exact_match(concept, hops, limit, query_type)
    if exact_result:
        return exact_result

    # If no exact match and semantic matching enabled, find similar concepts
    if use_semantic_matching:
        # Use adaptive threshold if enabled
        similar_concepts = SemanticGraphQueryBuilder.find_similar_concepts(
            concept,
            threshold=0.7,  # This will be overridden if adaptive is enabled
            use_adaptive=use_adaptive_threshold
        )

        if similar_concepts:
            # Try the most similar concept
            best_match, score = similar_concepts[0]
            print(f"No exact match for '{concept}', using '{best_match}' (similarity: {score:.2f})")

            result = _try_exact_match(best_match, hops, limit, query_type)
            if result:
                # Add information about the semantic match
                result["semantic_match"] = {
                    "original": concept,
                    "matched": best_match,
                    "score": score,
                    "alternatives": similar_concepts[1:5],
                    "adaptive_threshold_used": use_adaptive_threshold
                }
                return result

    return None


def _try_exact_match(concept: str, hops: int, limit: int, query_type: str) -> Optional[Dict]:
    """Try to get subgraph with exact concept match"""

    # Define relationship filters for each query type
    query_configs = {
        "neighborhood": None,  # All relationships
        "hierarchy": ["IS_A", "PART_OF", "HAS_COMPONENT", "EXTENDS"],
        "usage": ["USED_IN", "APPLIED_TO", "IMPLEMENTS", "EXAMPLE_OF"],
        "weighted": None,
        "semantic": None  # New: uses semantic expansion
    }

    relationship_filter = query_configs.get(query_type)

    query = SemanticGraphQueryBuilder.get_semantic_subgraph(
        concept, hops, limit, relationship_filter
    )

    with driver.session() as session:
        try:
            result = session.run(query, name=concept).single()
            if result and result["subgraph"]:
                subgraph = result["subgraph"]
                if subgraph.get("stats", {}).get("node_count", 0) > 1:
                    return subgraph
            return None
        except Exception as e:
            print(f"Query error for concept '{concept}': {e}")
            return None


# Wrapper function for easy migration

def get_subgraph_enhanced(concept: str, hops: int = 2, limit: int = 10,
                          query_type: str = "neighborhood",
                          use_semantic: bool = True,
                          use_adaptive: bool = False) -> Optional[Dict]:
    """
    Enhanced version with semantic understanding
    NEW: use_adaptive parameter for adaptive thresholding

    Parameters:
        use_semantic: Enable semantic matching (default: True)
        use_adaptive: Enable adaptive threshold (default: False)
    """
    result = get_subgraph_semantic(
        concept, hops, limit, query_type,
        use_semantic,
        use_adaptive  # NEW parameter
    )

    if not result and not use_semantic:
        # Fall back to original implementation
        from kg_utils import get_subgraph_dynamic
        result = get_subgraph_dynamic(concept, hops, limit, query_type)

    return result


def _get_adaptive_threshold(query: str, candidates: List[Tuple[str, float]]) -> float:
    """
    Calculate adaptive threshold using the trained model
    """
    if not ADAPTIVE_MODEL or not candidates:
        return 0.7  # Default fallback

    try:
        # Extract features using the model's method
        features = ADAPTIVE_MODEL.extract_features(query, candidates)

        # Test different thresholds to find the best one
        best_threshold = 0.7
        best_prob = 0

        feature_cols = sorted(features.keys())

        for test_threshold in np.arange(0.4, 0.9, 0.05):
            features['threshold'] = test_threshold
            feature_vector = [features[k] for k in sorted(features.keys())]

            # Get probability that this threshold is good
            prob = ADAPTIVE_MODEL.classifier.predict_proba([feature_vector])[0, 1]

            if prob > best_prob:
                best_prob = prob
                best_threshold = test_threshold

        return float(best_threshold)

    except Exception as e:
        print(f"❌ Error in adaptive threshold: {e}")
        return 0.7  # Fallback to default

