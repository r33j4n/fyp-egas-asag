# kg_utils.py
from neo4j import GraphDatabase
import os

NEO_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO_USER = os.getenv("NEO4J_USER", "neo4j")
NEO_PW = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PW))


def get_subgraph(concept, hops=2, limit=10):
    # Option 1: Get paths instead of subgraph for tree conversion
    query = f"""
    MATCH (c:Concept {{name: $name}})
    CALL apoc.path.expandConfig(c, {{
        maxLevel: {hops},
        limit: $limit,
        relationshipFilter: '>'
    }}) YIELD path
    WITH collect(path) AS paths
    CALL apoc.paths.toJsonTree(paths) YIELD value AS tree
    RETURN tree
    """

    with driver.session() as session:
        result = session.run(query, name=concept, limit=limit).single()
    return result["tree"] if result else None


def get_subgraph_alternative(concept, hops=2, limit=10):
    # Option 2: Return structured subgraph data without tree conversion
    query = f"""
    MATCH (c:Concept {{name: $name}})
    CALL apoc.path.subgraphAll(
          c,
          {{
            maxLevel: {hops},
            limit: $limit,
            relationshipFilter: '>'
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
        }}]
    }} AS subgraph
    """

    with driver.session() as session:
        result = session.run(query, name=concept, limit=limit).single()
    return result["subgraph"] if result else None


def get_subgraph_with_manual_tree(concept, hops=2, limit=10):
    # Option 3: Build tree structure manually from subgraph
    query = f"""
    MATCH (c:Concept {{name: $name}})
    CALL apoc.path.subgraphAll(
          c,
          {{
            maxLevel: {hops},
            limit: $limit,
            relationshipFilter: '>'
          }}) YIELD nodes, relationships

    // Create a tree-like structure manually
    WITH c AS root, nodes, relationships
    UNWIND relationships AS rel
    WITH root, nodes, relationships,
         collect({{
           parent: startNode(rel),
           child: endNode(rel),
           relationship: rel
         }}) AS connections

    RETURN {{
        root: {{
            id: id(root),
            labels: labels(root),
            properties: properties(root)
        }},
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
        connections: connections
    }} AS tree_structure
    """

    with driver.session() as session:
        result = session.run(query, name=concept, limit=limit).single()
    return result["tree_structure"] if result else None