"""Tests hors-ligne du graphe : LLM et retriever factices, aucun appel réseau."""

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from support_agent.graph import HANDOFF_MESSAGE, Decision, build_graph, is_grounded

DOCS = [
    Document(
        "Checkpoint is a snapshot of the graph state at a given point in time.",
        metadata={"source": "data/03_checkpoint.md"},
    ),
    Document(
        "`langgraph-prebuilt` provides an [implementation](https://example.com/ToolNode) "
        "of a node that executes tool calls - `ToolNode`:",
        metadata={"source": "data/02_prebuilt.md"},
    ),
    Document("Another chunk of the same file.", metadata={"source": "data/03_checkpoint.md"}),
]


class FakeLLM:
    """Imite `with_structured_output` d'un chat model en renvoyant une Decision fixe."""

    def __init__(self, decision: Decision):
        self.decision = decision
        self.calls = 0

    def with_structured_output(self, schema):
        assert schema is Decision

        def _invoke(_messages):
            self.calls += 1
            return self.decision

        return RunnableLambda(_invoke)


def fake_retriever(documents):
    return RunnableLambda(lambda _question: documents)


def decision(evidence="", can_answer=False, answer=""):
    return Decision(subject="test", evidence=evidence, can_answer=can_answer, answer=answer)


GROUNDED = decision(
    evidence="Checkpoint is a snapshot of the graph state at a given point in time.",
    can_answer=True,
    answer="Un checkpoint est un instantané de l'état du graphe.",
)


# --- is_grounded -----------------------------------------------------------


def test_grounded_exact_quote():
    assert is_grounded("Checkpoint is a snapshot of the graph state", DOCS)


def test_grounded_tolerates_case_punctuation_and_markdown_links():
    # Le modèle cite le texte du lien sans l'URL : doit être accepté.
    assert is_grounded(
        "langgraph-prebuilt provides an implementation of a node that executes tool calls",
        DOCS,
    )
    assert is_grounded("CHECKPOINT is a snapshot... of the graph state!", DOCS)


def test_invented_quote_is_rejected():
    assert not is_grounded("Deploy LangGraph on Kubernetes with Helm charts.", DOCS)


@pytest.mark.parametrize("evidence", ["", "   ", "graph"])
def test_empty_or_too_short_quote_is_rejected(evidence):
    assert not is_grounded(evidence, DOCS)


# --- graphe ------------------------------------------------------------------


def test_grounded_answer_ends_without_escalation():
    app = build_graph(FakeLLM(GROUNDED), fake_retriever(DOCS))
    result = app.invoke({"question": "Qu'est-ce qu'un checkpoint ?"})

    assert result["escalated"] is False
    assert result["answer"] == GROUNDED.answer
    # Sources dédoublonnées, dans l'ordre de pertinence.
    assert result["sources"] == ["03_checkpoint.md", "02_prebuilt.md"]


def test_model_refusal_escalates():
    app = build_graph(FakeLLM(decision()), fake_retriever(DOCS))
    result = app.invoke({"question": "Kubernetes ?"})

    assert result["escalated"] is True
    assert result["answer"] == HANDOFF_MESSAGE


def test_hallucinated_evidence_escalates_even_if_model_says_yes():
    """Le garde-fou en code l'emporte sur la décision du modèle."""
    liar = decision(
        evidence="LangGraph ships official Helm charts for Kubernetes.",
        can_answer=True,
        answer="Utilisez les charts Helm officiels.",
    )
    app = build_graph(FakeLLM(liar), fake_retriever(DOCS))
    result = app.invoke({"question": "Kubernetes ?"})

    assert result["escalated"] is True
    assert result["sources"] == []


def test_no_documents_escalates_without_calling_llm():
    llm = FakeLLM(GROUNDED)
    app = build_graph(llm, fake_retriever([]))
    result = app.invoke({"question": "n'importe quoi"})

    assert result["escalated"] is True
    assert llm.calls == 0


def test_human_in_the_loop_pauses_then_resumes_with_human_answer():
    app = build_graph(
        FakeLLM(decision()),
        fake_retriever(DOCS),
        human_in_the_loop=True,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}}

    paused = app.invoke({"question": "Prix de LangSmith ?"}, config)
    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == {"question": "Prix de LangSmith ?"}

    resumed = app.invoke(Command(resume="Voir la page tarifs."), config)
    assert resumed["answer"] == "Voir la page tarifs."
    assert resumed["escalated"] is True


def test_human_in_the_loop_requires_checkpointer():
    with pytest.raises(ValueError):
        build_graph(FakeLLM(GROUNDED), fake_retriever(DOCS), human_in_the_loop=True)
