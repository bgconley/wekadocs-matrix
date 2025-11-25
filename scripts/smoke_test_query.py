import asyncio
import os
import sys

sys.path.append(os.getcwd())

from src.mcp_server.query_service import get_query_service
from src.shared.connections import close_connections, initialize_connections


async def run_test():
    print("Initializing connections...")
    await initialize_connections()

    qs = get_query_service()
    query = "What are the prerequisites for storage expansion?"

    print(f"\nExecuting search for: '{query}'")
    try:
        response = qs.search(
            query=query,
            top_k=3,
            verbosity="graph",
        )

        print("\n=== Search Results ===")
        print(f"Confidence: {response.answer_json.confidence}")
        print(f"Evidence Count: {len(response.answer_json.evidence)}")

        print("\n--- Answer (markdown) ---")
        ans = response.answer_markdown or ""
        if len(ans) > 2000:
            print(ans[:2000] + "... [truncated]")
        else:
            print(ans)

        print("\n--- Answer JSON ---")
        try:
            # Pydantic v1 style
            print(response.answer_json.json(indent=2))
        except Exception:
            # Fallback: print the object directly
            print(response.answer_json)

        print("\n--- Top Evidence ---")
        for i, ev in enumerate(response.answer_json.evidence[:3]):
            print(
                f"{i+1}. title={getattr(ev, 'title', None)} "
                f"confidence={getattr(ev, 'confidence', None)} "
                f"source={getattr(getattr(ev, 'metadata', {}), 'get', lambda k: None)('source_uri')}"
            )
    except Exception as e:
        print(f"\nERROR: Search failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await close_connections()


if __name__ == "__main__":
    asyncio.run(run_test())
