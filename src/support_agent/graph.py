"""Graphe LangGraph : retrieve -> generate -> (END | escalate).

    START -> retrieve -> generate --(réponse justifiée)--> END
                            └------(sinon)-----------> escalate -> END

`generate` fait UN appel LLM en sortie structurée (`Decision`) : le modèle doit
citer l'extrait du contexte qui justifie sa réponse AVANT de décider. Le router
ne fait pas confiance à la seule décision du modèle : il vérifie, en code, que
la citation existe vraiment dans les documents récupérés.

En mode human-in-the-loop, `escalate` met le graphe en pause (`interrupt`) et
reprend avec la réponse d'un opérateur humain.
"""

import re
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

HANDOFF_MESSAGE = (
    "Je ne trouve pas cette information dans la documentation. "
    "Je transmets votre question à un humain."
)

SYSTEM_PROMPT = """Tu es l'assistant support de LangGraph.
Tu réponds UNIQUEMENT à partir du contexte fourni, jamais de tes connaissances générales.

1. `subject` : le sujet précis de la question, en quelques mots (ex. « déploiement
   sur Kubernetes », « persistance avec SQLite »).
2. `evidence` : recopie MOT POUR MOT la phrase du contexte qui répond à la question
   sur CE sujet. Une phrase générale qui ne traite pas du sujet ne compte pas.
   Si aucune phrase du contexte ne traite explicitement du sujet, laisse ce champ vide.
3. `can_answer` : vrai seulement si `evidence` répond réellement à la question posée.
   Faux si la question porte sur un outil, une plateforme ou une fonctionnalité que
   le contexte ne mentionne pas, même s'il aborde un sujet voisin.
4. `answer` : si `can_answer` est vrai, une à trois phrases en français, fidèles au
   contexte. Sinon, chaîne vide.
N'invente jamais et ne complète pas avec des connaissances extérieures."""


class Decision(BaseModel):
    """Sortie structurée du LLM. L'ordre des champs est voulu : le modèle génère dans
    cet ordre, il identifie donc le sujet et cite la preuve AVANT de décider."""

    subject: str = Field(description="Sujet précis de la question")
    evidence: str = Field(description="Extrait exact du contexte qui répond, ou vide")
    can_answer: bool = Field(description="Le contexte permet-il de répondre précisément ?")
    answer: str = Field(description="Réponse en français, ou vide si can_answer est faux")


class State(TypedDict, total=False):
    question: str
    documents: list[Document]
    decision: Decision
    answer: str
    sources: list[str]
    escalated: bool


def _normalize(text: str) -> str:
    """Minuscules, sans liens markdown, URL ni ponctuation, espaces fusionnés."""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [texte](url) -> texte
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(text.split())


def is_grounded(evidence: str, documents: list[Document]) -> bool:
    """Vrai si la citation du modèle figure réellement dans un des documents.

    Comparaison tolérante (casse, ponctuation, espaces) mais littérale :
    une citation paraphrasée ou inventée est rejetée.
    """
    needle = _normalize(evidence)
    if len(needle) < 15:  # citation vide ou trop courte pour prouver quoi que ce soit
        return False
    return any(needle in _normalize(doc.page_content) for doc in documents)


def should_answer(decision: Decision, documents: list[Document]) -> bool:
    return (
        decision.can_answer
        and bool(decision.answer.strip())
        and is_grounded(decision.evidence, documents)
    )


def format_context(documents: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[{_source_name(doc)}]\n{doc.page_content}" for doc in documents)


def _source_name(doc: Document) -> str:
    return Path(doc.metadata.get("source", "inconnu")).name


def build_graph(
    llm,
    retriever: Runnable,
    *,
    human_in_the_loop: bool = False,
    checkpointer=None,
):
    """Assemble le graphe. LLM et retriever sont injectés (testable sans réseau).

    `llm` doit exposer `with_structured_output` (tout chat model LangChain).
    """
    if human_in_the_loop and checkpointer is None:
        raise ValueError("Le mode human-in-the-loop nécessite un checkpointer.")

    decider = llm.with_structured_output(Decision)

    def retrieve(state: State):
        return {"documents": retriever.invoke(state["question"])}

    def generate(state: State):
        documents = state["documents"]
        if not documents:
            # Rien à citer : inutile de payer un appel LLM.
            return {"decision": Decision(subject="", evidence="", can_answer=False, answer="")}
        decision = decider.invoke(
            [
                SystemMessage(SYSTEM_PROMPT),
                HumanMessage(
                    f"Contexte :\n{format_context(documents)}\n\nQuestion : {state['question']}"
                ),
            ]
        )
        update = {"decision": decision}
        if should_answer(decision, documents):
            # dict.fromkeys : dédoublonne en gardant l'ordre de pertinence.
            sources = list(dict.fromkeys(_source_name(doc) for doc in documents))
            update |= {"answer": decision.answer.strip(), "sources": sources, "escalated": False}
        return update

    def route(state: State) -> str:
        return END if should_answer(state["decision"], state["documents"]) else "escalate"

    def escalate(state: State):
        if human_in_the_loop:
            # Pause : l'état est sauvegardé par le checkpointer ; reprise via
            # app.invoke(Command(resume="réponse humaine"), config).
            human_answer = interrupt({"question": state["question"]})
            return {"answer": human_answer, "sources": [], "escalated": True}
        return {"answer": HANDOFF_MESSAGE, "sources": [], "escalated": True}

    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", route, ["escalate", END])
    graph.add_edge("escalate", END)

    return graph.compile(checkpointer=checkpointer)
