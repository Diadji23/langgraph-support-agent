from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# 1. CHARGER : lire tous les .md du dossier data/
loader = DirectoryLoader("data/", glob="*.md", loader_cls=TextLoader)
docs = loader.load()
print(f"{len(docs)} documents chargés")

# 2. DÉCOUPER : casser les docs en morceaux (chunks)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks créés")



embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")

# 4. STOCKER : créer le vector store à partir des chunks + embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
print("Vector store créé dans ./chroma_db")


vs = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

results = vs.similarity_search("Comment persister l'état d'un graphe ?", k=2)
for r in results:
    print("---")
    print(r.page_content)