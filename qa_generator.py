# qa_generator.py
import os
import requests
import re
import json
from typing import Dict, Set, List

IGNORE_KEYS = re.compile(r"^(?:name$|_|\.)")  # skip meta & _id / _type / in_topic._id …


def triples_to_sentences(tree: Dict) -> str:
    """
    Walk an APOC JSON tree and produce a paragraph of unique fact sentences.
    Handles:
      • nested dicts / lists
      • duplicate relations
      • cycles (visited set)
    """
    visited_nodes: Set[int] = set()
    sentences: Set[str] = set()

    def walk(node: Dict):
        if not isinstance(node, dict) or id(node) in visited_nodes:
            return
        visited_nodes.add(id(node))

        subj = node.get("name")
        if not subj:
            return

        for key, value in node.items():
            # ignore meta keys (_type, _id, in_topic._id …)
            if IGNORE_KEYS.match(key):
                continue

            relation = key.replace("_", " ").lower()

            if isinstance(value, dict):
                obj = value.get("name")
                if obj:
                    sentences.add(f"{subj} {relation} {obj}.")
                walk(value)

            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        obj = child.get("name")
                        if obj:
                            sentences.add(f"{subj} {relation} {obj}.")
                        walk(child)

    walk(tree)
    # deterministic order helps with caching / testing
    return " ".join(sorted(sentences))


def generate_prompt(paragraph):
    return f"""
You create short-answer Data Structures questions for exams.

Given the following factual paragraph:

\"\"\"{paragraph}\"\"\"

Write ONE question whose answer is **present verbatim** in the paragraph.
Return ONLY valid JSON in this exact format: {{"question": "...", "answer": "..."}}
Do not include any other text, explanations, or formatting.
"""


def make_api_request(prompt, model="llama3-8b-8192"):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=body, timeout=40)
    return response.json()


def parse_response(response):
    """
    Robustly parse LLM response to extract JSON, handling extra text and formatting.
    """
    if "choices" not in response or len(response["choices"]) == 0:
        return {"question": "No response from LLM", "answer": ""}

    content = response["choices"][0]["message"]["content"]

    # Method 1: Try to find JSON using regex
    json_pattern = r'\{[^{}]*"question"[^{}]*"answer"[^{}]*\}'
    json_match = re.search(json_pattern, content, re.DOTALL)

    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Method 2: Try to extract JSON between curly braces
    start_idx = content.find('{')
    end_idx = content.rfind('}')

    if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
        json_str = content[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Method 3: Try to parse the entire content as JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Method 4: Extract question and answer using regex patterns
    question_match = re.search(r'"question":\s*"([^"]*)"', content)
    answer_match = re.search(r'"answer":\s*"([^"]*)"', content)

    if question_match and answer_match:
        return {
            "question": question_match.group(1),
            "answer": answer_match.group(1)
        }

    # Method 5: Last resort - try to extract any text that looks like Q&A
    question_patterns = [
        r'question["\s:]*([^"\n}]+)',
        r'Q[:\s]*([^"\n}]+)',
        r'Question[:\s]*([^"\n}]+)'
    ]

    answer_patterns = [
        r'answer["\s:]*([^"\n}]+)',
        r'A[:\s]*([^"\n}]+)',
        r'Answer[:\s]*([^"\n}]+)'
    ]

    question = None
    answer = None

    for pattern in question_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            question = match.group(1).strip().strip('"').strip()
            break

    for pattern in answer_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            answer = match.group(1).strip().strip('"').strip()
            break

    return {
        "question": question or "Error parsing model response",
        "answer": answer or "Error parsing model response"
    }


def generate_qa_from_paragraph(paragraph, model="llama3-8b-8192"):
    prompt = generate_prompt(paragraph)
    response = make_api_request(prompt, model)
    return parse_response(response)