# app.py
from flask import Flask, request, jsonify
from kg_utils import get_subgraph
from qa_generator import triples_to_sentences, generate_qa_from_paragraph

app = Flask(__name__)

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    data = request.json
    concepts = data.get("concepts", [])
    hops = data.get("hops", 2)
    limit = data.get("limit", 10)
    results = []
    for concept in concepts:
        tree = get_subgraph(concept, hops=hops, limit=limit)
        if not tree:
            results.append({"concept": concept, "error": "No subgraph found"})
            continue
        paragraph = triples_to_sentences(tree)
        qa = generate_qa_from_paragraph(paragraph)
        results.append({"concept": concept, "qa": qa})
    return jsonify(results)

@app.route('/')
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run(debug=True)