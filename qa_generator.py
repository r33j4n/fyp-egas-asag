# qa_generator.py (Enhanced version)
import os
import requests
import re
import json
from typing import Dict, Set, List, Optional

IGNORE_KEYS = re.compile(r"^(?:name$|_|\.)")


def triples_to_sentences(tree: Dict) -> str:
    """Enhanced triple extraction with better sentence formation"""
    if not tree or not isinstance(tree, dict):
        return ""

    visited_nodes: Set[int] = set()
    sentences: Set[str] = set()

    def walk(node: Dict, depth: int = 0):
        if not isinstance(node, dict) or id(node) in visited_nodes or depth > 5:
            return
        visited_nodes.add(id(node))

        # Handle both 'name' and 'properties.name'
        subj = node.get("name") or node.get("properties", {}).get("name")
        if not subj:
            return

        for key, value in node.items():
            if IGNORE_KEYS.match(key):
                continue

            # Better relation formatting
            relation = key.replace("_", " ").replace("-", " ").lower()

            if isinstance(value, dict):
                obj = value.get("name") or value.get("properties", {}).get("name")
                if obj and obj != subj:  # Avoid self-references
                    sentences.add(f"{subj} {relation} {obj}")
                walk(value, depth + 1)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        obj = child.get("name") or child.get("properties", {}).get("name")
                        if obj and obj != subj:
                            sentences.add(f"{subj} {relation} {obj}")
                        walk(child, depth + 1)
            elif isinstance(value, str) and value != subj and len(value.strip()) > 0:
                # Include direct string properties
                sentences.add(f"{subj} {relation} {value}")

    # Also handle nodes and relationships structure
    if "nodes" in tree and "relationships" in tree:
        node_map = {node["id"]: node for node in tree.get("nodes", [])}

        for rel in tree.get("relationships", []):
            start_node = node_map.get(rel["start_node"])
            end_node = node_map.get(rel["end_node"])

            if start_node and end_node:
                start_name = start_node.get("properties", {}).get("name", str(start_node["id"]))
                end_name = end_node.get("properties", {}).get("name", str(end_node["id"]))
                rel_type = rel.get("type", "relates to").replace("_", " ").lower()

                if start_name != end_name:
                    sentences.add(f"{start_name} {rel_type} {end_name}")
    else:
        walk(tree)

    return ". ".join(sorted(sentences)) + "." if sentences else ""


def generate_enhanced_prompts(paragraph: str, question_count: int = 1, question_types: List[str] = None) -> List[str]:
    """Generate multiple varied prompts for different question types"""

    if question_types is None:
        question_types = ["factual", "conceptual", "application"]

    prompts = []

    prompt_templates = {
        "factual": f"""
You are creating exam questions for Computer Science students.

Given this knowledge: "{paragraph}"

Create {question_count} factual question(s) that test specific details from the text.
Each answer must be found EXACTLY in the given text.

Return as JSON array: [{{"question": "...", "answer": "...", "type": "factual"}}]
Make questions natural and engaging, not robotic.
""",
        "conceptual": f"""
You are creating exam questions for Computer Science students.

Given this knowledge: "{paragraph}"

Create {question_count} conceptual question(s) that test understanding of relationships and concepts.
Each answer must be derivable from the given text.

Return as JSON array: [{{"question": "...", "answer": "...", "type": "conceptual"}}]
Make questions thoughtful and educational, not mechanical.
""",
        "application": f"""
You are creating exam questions for Computer Science students.

Given this knowledge: "{paragraph}"

Create {question_count} application question(s) that test practical usage or examples.
Each answer must be supported by the given text.

Return as JSON array: [{{"question": "...", "answer": "...", "type": "application"}}]
Make questions practical and relevant to real-world scenarios.
"""
    }

    for q_type in question_types:
        if q_type in prompt_templates:
            prompts.append(prompt_templates[q_type])

    return prompts


def make_api_request(prompt: str, model: str = "llama3-8b-8192") -> Dict:
    """Enhanced API request with better error handling"""
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        return {"error": "GROQ_API_KEY not set"}

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,  # Slightly higher for more natural questions
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}


def parse_enhanced_response(response: Dict) -> List[Dict]:
    """Parse response expecting JSON array of questions"""
    if "error" in response:
        return [{"question": f"API Error: {response['error']}", "answer": "", "type": "error"}]

    if "choices" not in response or len(response["choices"]) == 0:
        return [{"question": "No response from LLM", "answer": "", "type": "error"}]

    content = response["choices"][0]["message"]["content"].strip()

    # Try to parse as JSON array first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]  # Single question as dict
    except json.JSONDecodeError:
        pass

    # Fallback: Extract from text using the previous robust method
    json_pattern = r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}'
    matches = re.findall(json_pattern, content, re.DOTALL)

    questions = []
    for match in matches:
        try:
            questions.append(json.loads(match))
        except json.JSONDecodeError:
            continue

    if questions:
        return questions

    # Last resort: single question extraction
    question_match = re.search(r'"question":\s*"([^"]*)"', content)
    answer_match = re.search(r'"answer":\s*"([^"]*)"', content)

    if question_match and answer_match:
        return [{
            "question": question_match.group(1),
            "answer": answer_match.group(1),
            "type": "extracted"
        }]

    return [{"question": "Failed to parse response", "answer": content[:100], "type": "error"}]


def generate_multiple_qa_from_paragraph(paragraph: str, question_count: int = 1,
                                        question_types: List[str] = None,
                                        model: str = "llama3-8b-8192") -> List[Dict]:
    """Generate multiple Q&A pairs with different types"""
    if not paragraph.strip():
        return [{"question": "No content available", "answer": "", "type": "error"}]

    if question_types is None:
        question_types = ["factual"]

    all_questions = []
    prompts = generate_enhanced_prompts(paragraph, question_count, question_types)

    for prompt in prompts:
        response = make_api_request(prompt, model)
        questions = parse_enhanced_response(response)
        all_questions.extend(questions)

    # If we need more questions, repeat with different prompts
    while len(all_questions) < question_count and len(all_questions) > 0:
        # Use the best performing prompt template
        additional_prompt = generate_enhanced_prompts(paragraph, 1, ["factual"])[0]
        response = make_api_request(additional_prompt, model)
        additional_questions = parse_enhanced_response(response)
        all_questions.extend(additional_questions)

    return all_questions[:question_count]  # Limit to requested count