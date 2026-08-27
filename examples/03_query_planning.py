"""Query-planning example placeholder.

This example will compare direct topic search, one-shot query rewriting, and
iterative multi-query planning after `ResearchAgent` is implemented.
"""

from hyscript.agent import ResearchAgent


def main() -> None:
    agent = ResearchAgent()
    print(f"Query-planning scaffold: {agent.__class__.__name__}")


if __name__ == "__main__":
    main()
