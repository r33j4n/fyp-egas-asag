"""
kg_generator.py
---------------
Hardcoded version: Build a Neo4j knowledge graph from a JSON file of triples.
"""

import json
from pathlib import Path
from typing import List
from neo4j import GraphDatabase, BoltDriver


# ── Configuration (hardcoded) ───────────────────────────────────────────────
TRIPLES_JSON_PATH = "triplets_data.json"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
WIPE_FIRST = True


# ── Data Loader ─────────────────────────────────────────────────────────────
def load_data(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ── Builder ─────────────────────────────────────────────────────────────────
class KGBuilder:
    """
    Creates:

        (:Topic {name})
        (:Concept {name})
        (:Concept)-[:RELATION]->(:Concept)
        (:Concept)-[:IN_TOPIC]->(:Topic)
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver: BoltDriver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def wipe_db(self):
        with self.driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")

    @staticmethod
    def _merge_triple_tx(
        tx,
        subj: str,
        rel: str,
        obj: str,
        topics: List[str],
        src_id: int,
    ):
        cypher = f"""
UNWIND $topics AS topicName
MERGE (t:Topic {{name: topicName}})

MERGE (s:Concept {{name: $subj}})
MERGE (o:Concept {{name: $obj}})

MERGE (s)-[r:`{rel}`]->(o)
ON CREATE SET r.source_ids = [$src]
ON MATCH  SET r.source_ids = apoc.coll.toSet(
                   coalesce(r.source_ids, []) + [$src])

MERGE (s)-[:IN_TOPIC]->(t)
MERGE (o)-[:IN_TOPIC]->(t)
"""
        tx.run(cypher, subj=subj, obj=obj, topics=topics, src=src_id)

    def build(self, data):
        with self.driver.session() as session:
            for record in data:
                src_id: int = record["id"]
                topics: List[str] = record.get("topics", [])
                for subj, rel, obj in record["triples"]:
                    session.execute_write(
                        self._merge_triple_tx, subj, rel, obj, topics, src_id
                    )


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    data = load_data(Path(TRIPLES_JSON_PATH))
    builder = KGBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        if WIPE_FIRST:
            print("🧹  Clearing existing graph …")
            builder.wipe_db()

        print(f"🚀  Loading {len(data)} question groups into Neo4j …")
        builder.build(data)
        print("✅  Knowledge-graph construction complete.")
    finally:
        builder.close()


if __name__ == "__main__":
    main()