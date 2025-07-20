# app.py (Enhanced version)
from flask import Flask, request, jsonify
from kg_utils import get_subgraph_dynamic
from qa_generator import triples_to_sentences,generate_multiple_qa_from_paragraph

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
def hello_world():
    return 'Enhanced G-RAG Question Generator API'


if __name__ == '__main__':
    app.run(debug=True)