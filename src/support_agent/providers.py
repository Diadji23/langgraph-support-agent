"""Construction des clients NVIDIA (LLM, embeddings) et du retriever Chroma.

Isolé du graphe pour que les tests puissent injecter des faux sans clé API.
"""

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

from support_agent.config import ROOT_DIR, settings

load_dotenv(ROOT_DIR / ".env")


def get_llm() -> ChatNVIDIA:
    # Nemotron 3.5 est un modèle de raisonnement : par défaut il génère une longue
    # chaîne de pensée avant de répondre (lent, et la réponse peut être tronquée).
    # Le flag officiel du chat template désactive ce mode ; un "/no_think" dans
    # le prompt est ignoré (mesuré : 14 s / 470 tokens vs 3,5 s / 43 tokens).
    return ChatNVIDIA(
        model=settings.llm_model,
        temperature=0,
        model_kwargs={"chat_template_kwargs": {"enable_thinking": False}},
    )


def get_embeddings() -> NVIDIAEmbeddings:
    return NVIDIAEmbeddings(model=settings.embedding_model)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection,
        persist_directory=str(settings.chroma_dir),
        embedding_function=get_embeddings(),
    )


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})
