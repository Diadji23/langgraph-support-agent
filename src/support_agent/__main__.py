"""CLI : python -m support_agent "ma question" [--hitl]"""

import argparse
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_agent.graph import build_graph
from support_agent.providers import get_llm, get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent support LangGraph (RAG).")
    parser.add_argument("question", help="Question sur LangGraph")
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="En cas d'escalade, mettre le graphe en pause et demander la réponse à un humain",
    )
    args = parser.parse_args()

    checkpointer = InMemorySaver() if args.hitl else None
    app = build_graph(
        get_llm(), get_retriever(), human_in_the_loop=args.hitl, checkpointer=checkpointer
    )
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = app.invoke({"question": args.question}, config)
    if "__interrupt__" in result:
        print("⏸  Escalade : l'agent n'a pas trouvé la réponse dans la documentation.")
        human_answer = input("Réponse de l'opérateur humain > ")
        result = app.invoke(Command(resume=human_answer), config)

    print(result["answer"])
    if result.get("sources"):
        print("\nSources :", ", ".join(result["sources"]))


if __name__ == "__main__":
    main()
