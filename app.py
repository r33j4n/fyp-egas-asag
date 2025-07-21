# app.py (Enhanced version)
from flask import Flask, request, jsonify
from kg_utils import get_subgraph_dynamic
from qa_generator import triples_to_sentences,generate_multiple_qa_from_paragraph
from semantic_enhanced_kg_utils import get_subgraph_enhanced, SemanticGraphQueryBuilder


app = Flask(__name__)


@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json

    # Enhanced parameter handling
    concepts = data.get("concepts", [])
    hops = data.get("hops", 2)
    limit = data.get("limit", 10)
    question_count = data.get("question_count", 1)  # NEW: Number of questions per concept
    query_type = data.get("query_type", "neighborhood")  # NEW: Type of graph query
    question_types = data.get("question_types", ["factual"])  # NEW: Types of questions

    # Validate inputs
    if not concepts:
        return jsonify({"error": "No concepts provided"}), 400

    results = []

    for concept in concepts:
        concept_result = {"concept": concept}

        # Get subgraph with specified query type
        tree = get_subgraph_dynamic(concept, hops, limit, query_type)

        if not tree:
            concept_result["error"] = f"No subgraph found for concept '{concept}'"
            concept_result["qa"] = []
            results.append(concept_result)
            continue

        # Convert to sentences
        paragraph = triples_to_sentences(tree)

        if not paragraph.strip():
            concept_result["error"] = f"No meaningful content extracted for concept '{concept}'"
            concept_result["qa"] = []
            results.append(concept_result)
            continue

        # Generate multiple questions
        qa_pairs = generate_multiple_qa_from_paragraph(
            paragraph, question_count, question_types
        )

        concept_result["qa"] = qa_pairs
        concept_result["subgraph_stats"] = tree.get("stats", {})
        concept_result["paragraph"] = paragraph  # For debugging

        results.append(concept_result)

    return jsonify(results)


@app.route('/available_query_types', methods=['GET'])
def available_query_types():
    """Return available query types for the frontend"""
    return jsonify({
        "query_types": ["neighborhood", "hierarchy", "usage", "weighted"],
        "question_types": ["factual", "conceptual", "application"]
    })


@app.route('/')
def home():
    return '''
    <h1>Enhanced G-RAG Question Generator API</h1>
    <h2>Available Versions:</h2>
    <ul>
        <li><b>V1 (Original):</b> /generate_questions_v1 - Rule-based matching</li>
        <li><b>V2 (Enhanced):</b> /generate_questions_v2 - Semantic understanding</li>
    </ul>
    <h3>New Features in V2:</h3>
    <ul>
        <li>Handles spelling mistakes automatically</li>
        <li>Finds semantically similar concepts</li>
        <li>Expands relationship types intelligently</li>
        <li>Provides concept suggestions when no match found</li>
    </ul>
    '''


@app.route('/generate_questions_v2', methods=['POST'])
def generate_questions_v2():
    """Enhanced version with semantic understanding"""
    data = request.json

    # Enhanced parameter handling
    concepts = data.get("concepts", [])
    hops = data.get("hops", 2)
    limit = data.get("limit", 10)
    question_count = data.get("question_count", 1)
    query_type = data.get("query_type", "neighborhood")
    question_types = data.get("question_types", ["factual"])
    use_semantic = data.get("use_semantic", True)  # NEW: Enable semantic matching

    # Validate inputs
    if not concepts:
        return jsonify({"error": "No concepts provided"}), 400

    results = []

    for concept in concepts:
        concept_result = {"concept": concept}

        # Get subgraph with semantic matching
        tree = get_subgraph_enhanced(concept, hops, limit, query_type, use_semantic)

        if not tree:
            # If no match found, provide suggestions
            if use_semantic:
                suggestions = SemanticGraphQueryBuilder.find_similar_concepts(concept, threshold=0.5)
                concept_result["error"] = f"No subgraph found for concept '{concept}'"
                concept_result["suggestions"] = [
                    {"concept": c, "similarity": float(s)}  # Explicit float conversion
                    for c, s in suggestions[:5]
                ]
            else:
                concept_result["error"] = f"No subgraph found for concept '{concept}'"

            concept_result["qa"] = []
            results.append(concept_result)
            continue

        # Add semantic match info if available
        if "semantic_match" in tree:
            semantic_match = tree["semantic_match"]
            # Ensure all numerical values are native Python floats
            if "score" in semantic_match:
                semantic_match["score"] = float(semantic_match["score"])
            if "alternatives" in semantic_match:
                semantic_match["alternatives"] = [
                    (c, float(s)) for c, s in semantic_match["alternatives"]
                ]
            concept_result["semantic_match"] = semantic_match

        # Convert to sentences
        paragraph = triples_to_sentences(tree)

        if not paragraph.strip():
            concept_result["error"] = f"No meaningful content extracted for concept '{concept}'"
            concept_result["qa"] = []
            results.append(concept_result)
            continue

        # Generate multiple questions
        qa_pairs = generate_multiple_qa_from_paragraph(
            paragraph, question_count, question_types
        )

        concept_result["qa"] = qa_pairs
        concept_result["subgraph_stats"] = tree.get("stats", {})
        concept_result["paragraph"] = paragraph

        results.append(concept_result)

    return jsonify(results)


@app.route('/search_concepts', methods=['POST'])
def search_concepts():
    """Search for concepts with semantic similarity"""
    data = request.json
    query = data.get("query", "")
    threshold = data.get("threshold", 0.6)

    if not query:
        return jsonify({"error": "No query provided"}), 400

    similar_concepts = SemanticGraphQueryBuilder.find_similar_concepts(query, threshold)

    return jsonify({
        "query": query,
        "matches": [
            {"concept": c, "similarity": f"{s:.2f}"}
            for c, s in similar_concepts
        ]
    })


@app.route('/available_features_v2', methods=['GET'])
def available_features_v2():
    """Return available features for the enhanced version"""
    return jsonify({
        "query_types": ["neighborhood", "hierarchy", "usage", "weighted", "semantic"],
        "question_types": ["factual", "conceptual", "application"],
        "features": {
            "semantic_matching": "Handles spelling mistakes and finds similar concepts",
            "fuzzy_matching": "Combines with semantic similarity for better results",
            "relationship_expansion": "Automatically includes similar relationship types",
            "concept_suggestions": "Provides alternatives when no exact match found"
        }
    })


if __name__ == '__main__':
    app.run(debug=True)

