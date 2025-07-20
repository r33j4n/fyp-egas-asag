# kg_utils.py
from neo4j import GraphDatabase
import os
from typing import Dict, List, Optional

NEO_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO_USER = os.getenv("NEO4J_USER", "neo4j")
NEO_PW = os.getenv("NEO4J_PASSWORD", "password")

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PW))


class GraphQueryBuilder:
    """Build different types of graph queries based on user requirements"""

    @staticmethod
    def get_concept_neighborhood(concept: str, hops: int, limit: int) -> str:
        """Get immediate neighbors and their connections"""
        return f"""
        MATCH (c:Concept {{name: $name}})
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {hops},
            limit: {limit},
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
            }}],
            stats: {{
                node_count: size(nodes),
                relationship_count: size(relationships),
                max_hops: {hops}
            }}
        }} AS subgraph
        """

    @staticmethod
    def get_concept_hierarchy(concept: str, hops: int, limit: int) -> str:
        """Get hierarchical relationships (parent-child, is-a, part-of)"""
        return f"""
        MATCH (c:Concept {{name: $name}})
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {hops},
            limit: {limit},
            relationshipFilter: 'IS_A|PART_OF|HAS_COMPONENT|EXTENDS'
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
                max_hops: {hops}
            }}
        }} AS subgraph
        """

    @staticmethod
    def get_concept_usage(concept: str, hops: int, limit: int) -> str:
        """Get usage patterns and applications"""
        return f"""
        MATCH (c:Concept {{name: $name}})
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {hops},
            limit: {limit},
            relationshipFilter: 'USED_IN|APPLIED_TO|IMPLEMENTS|EXAMPLE_OF'
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
                max_hops: {hops}
            }}
        }} AS subgraph
        """

    @staticmethod
    def get_weighted_subgraph(concept: str, hops: int, limit: int) -> str:
        """Get subgraph with weighted relationships for more diverse results"""
        return f"""
        MATCH (c:Concept {{name: $name}})
        CALL apoc.path.subgraphAll(c, {{
            maxLevel: {hops},
            limit: {limit},
            relationshipFilter: '>',
            filterStartNode: false,
            optional: false
        }}) YIELD nodes, relationships

        WITH c, nodes, relationships
        // Add diversity by including nodes with different relationship types
        UNWIND relationships as rel
        WITH c, nodes, relationships, type(rel) as rel_type, count(*) as type_count
        ORDER BY type_count ASC  // Prioritize less common relationship types

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
                diversity_score: size(collect(DISTINCT rel_type))
            }}
        }} AS subgraph
        LIMIT 1
        """


def get_subgraph_dynamic(concept: str, hops: int, limit: int, query_type: str = "neighborhood") -> Optional[Dict]:
    """
    Get subgraph based on query type with proper parameter validation
    """
    # Validate parameters
    hops = max(1, min(hops, 10))  # Ensure hops is between 1-10
    limit = max(1, min(limit, 100))  # Ensure limit is between 1-100

    query_builders = {
        "neighborhood": GraphQueryBuilder.get_concept_neighborhood,
        "hierarchy": GraphQueryBuilder.get_concept_hierarchy,
        "usage": GraphQueryBuilder.get_concept_usage,
        "weighted": GraphQueryBuilder.get_weighted_subgraph
    }

    if query_type not in query_builders:
        query_type = "neighborhood"  # Default fallback

    query = query_builders[query_type](concept, hops, limit)

    with driver.session() as session:
        try:
            result = session.run(query, name=concept, limit=limit).single()
            if result and result["subgraph"]:
                subgraph = result["subgraph"]
                # Ensure we have meaningful data
                if subgraph.get("stats", {}).get("node_count", 0) > 1:
                    return subgraph
            return None
        except Exception as e:
            print(f"Query error for concept '{concept}': {e}")
            return None


# Legacy functions for backward compatibility
def get_subgraph_alternative(concept: str, hops: int, limit: int) -> Optional[Dict]:
    return get_subgraph_dynamic(concept, hops, limit, "neighborhood")