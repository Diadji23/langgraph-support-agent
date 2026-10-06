"""Indexation du corpus : data/*.md -> chunks -> embeddings NVIDIA -> Chroma.

Usage : python -m support_agent.ingest
Idempotent : la collection est recréée à chaque exécution (pas de doublons).
"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownTextSplitter

from support_agent.config import settings
from support_agent.providers import get_vectorstore


def main() -> None:
    loader = DirectoryLoader(
        str(settings.data_dir),
        glob="[0-9]*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"{len(docs)} documents chargés depuis {settings.data_dir}")

    # Découpe en priorité sur les titres markdown pour garder les sections entières.
    splitter = MarkdownTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(docs)
        # Écarte les chunks réduits à un titre : ils n'apportent aucune réponse
        # mais prennent des places dans le top-k du retrieval.
        if len(chunk.page_content) >= settings.min_chunk_chars
    ]
    print(f"{len(chunks)} chunks créés")

    vectorstore = get_vectorstore()
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks)
    print(f"Index écrit dans {settings.chroma_dir} (collection '{settings.collection}')")


if __name__ == "__main__":
    main()
