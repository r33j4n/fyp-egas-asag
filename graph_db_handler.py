import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
from langchain_groq import ChatGroq
from langchain_neo4j import Neo4jGraph

# 1️⃣ Load environment variables
load_dotenv()
assert os.getenv("GROQ_API_KEY"), "Set GROQ_API_KEY in your .env"

# 2️⃣ Initialize Groq model
llm = ChatGroq(
    model= "llama3-8b-8192", #"mixtral-8x7b-versatile",
    temperature=0,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY")
)

# 3️⃣ Prepare few-shot examples
examples = [
    {
        "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
        "head": "Adam",
        "head_type": "Person",
        "relation": "WORKS_FOR",
        "tail": "Microsoft",
        "tail_type": "Company",
    },
    {
        "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
        "head": "Adam",
        "head_type": "Person",
        "relation": "HAS_AWARD",
        "tail": "Best Talent",
        "tail_type": "Award",
    },
]

# 4️⃣ Define the prompt template with variables
prompt_template = ChatPromptTemplate(
    messages=[
        ("system", "You are an AI assistant building a knowledge graph."),
        ("user", """
You are an AI assistant helping construct a knowledge graph from educational content.
Follow these steps:
1. Identify key concepts (entities).
2. Identify how they relate.
3. For each relation, output a JSON object with keys:
   - \"head\", \"head_type\", \"relation\", \"tail\", \"tail_type\".
Return a JSON array only—no extra text.

Examples:
{{examples}}

Now process this text:
\"\"\"{{text}}\"\"\"
""")
    ],
    input_variables=["examples", "text"],
)

# 5️⃣ Load data and prepare Document objects
df = pd.read_csv("Data/test_data.csv")
docs = [
    Document(page_content=row["expanded_answer"],
             metadata={"id": row["id"], "question": row["question"], "topics": row["topics"]})
    for _, row in df.iterrows()
]

# 6️⃣ Configure LLMGraphTransformer
transformer = LLMGraphTransformer(
    llm=llm,
    prompt=prompt_template,
    strict_mode=False,
    ignore_tool_usage=True
)

# 7️⃣ Extract GraphDocuments
graph_docs = transformer.convert_to_graph_documents([
    Document(
        page_content=doc["text"],
        metadata={
            "head": doc["examples"][0]["head"],
            "head_type": doc["examples"][0]["head_type"],
            "relation": doc["examples"][0]["relation"],
            "tail": doc["examples"][0]["tail"],
            "tail_type": doc["examples"][0]["tail_type"]
        }
    ) for doc in [
        {"text": doc.page_content, "examples": examples} for doc in docs
    ]
])

# 8️⃣ Load into Neo4j with provenance
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password=os.getenv("NEO4J_PASSWORD", "password")
)
graph.add_graph_documents(graph_docs, include_source=True, baseEntityLabel=True)

# 9️⃣ Verify the graph data
print(graph.query("""
MATCH (h:__Entity__)-[r]->(t:__Entity__)
RETURN h.id AS head_id, type(r) AS rel, t.id AS tail_id LIMIT 10
"""))

