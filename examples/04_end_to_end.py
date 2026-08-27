"""Application pipeline placeholder from topic discovery to generated script."""

from hyscript.agent import ResearchAgent, ScriptAgent, TopicAgent


def main() -> None:
    stages = (TopicAgent(), ResearchAgent(), ScriptAgent())
    print("Pipeline scaffold:", " -> ".join(type(stage).__name__ for stage in stages))


if __name__ == "__main__":
    main()
