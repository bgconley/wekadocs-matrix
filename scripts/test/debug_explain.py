#!/usr/bin/env python3
"""Debug script to inspect Neo4j EXPLAIN plan structure."""

import os

from neo4j import GraphDatabase


def main():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd = os.getenv("NEO4J_PASSWORD", "weka_graphrag_password")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))

    try:
        with driver.session() as session:
            # First, ensure we have some data
            session.run(
                """
                MERGE (s:Section {id: 'debug-sec-1', text: 'test'})
            """
            ).consume()

            # Run EXPLAIN
            result = session.run("EXPLAIN MATCH (s:Section) RETURN s LIMIT 5")
            summary = result.consume()
            plan = summary.plan

            print("Plan object:", plan)
            print("\nPlan type:", type(plan))
            print("\nPlan attributes:")
            for attr in dir(plan):
                if not attr.startswith("_"):
                    try:
                        value = getattr(plan, attr)
                        if not callable(value):
                            print(f"  {attr}: {value}")
                    except Exception as e:
                        print(f"  {attr}: <error: {e}>")

            # Try to flatten
            print("\n\nFlattening plan:")
            operators = []
            stack = [plan]
            while stack:
                node = stack.pop()
                print(f"  Node: {node}")
                op_type = getattr(node, "operator_type", None) or getattr(
                    node, "operatorType", None
                )
                print(f"    operator_type: {op_type}")
                if op_type:
                    operators.append(op_type)
                children = getattr(node, "children", None)
                print(f"    children: {children}")
                if children:
                    for child in children:
                        stack.append(child)

            print(f"\n  Found operators: {operators}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
