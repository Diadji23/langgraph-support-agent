from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", temperature=0)
embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

class State(TypedDict):
    question: str
    context: str      # les chunks récupérés
    verdict :str 
    answer: str



def retrieve(state: State):
    query = state["question"]
    sims = vectorstore.similarity_search(query, k=3)
    pages = []
    for doc in sims:
        pages.append(doc.page_content)
    context = " ".join(pages)
    print("=== Q:", query[:40], "| CONTEXTE:", context[:120])
    return {"context": context}


def generate(state: State):
    prompt = f"""Tu réponds à des questions sur LangGraph en te basant UNIQUEMENT sur le contexte ci-dessous.

Règle stricte : si le contexte ne permet pas de répondre PRÉCISÉMENT à la question, réponds uniquement par le mot ESCALADE. N'invente jamais.

Réponds par une phrase complète et explicative, jamais par un seul mot.

Contexte :
{state["context"]}

Question : {state["question"]}

Réponse (ou ESCALADE) :"""
    res = llm.invoke(prompt)
    return {"answer": res.content}


def escalate(state: State):
    return {"answer": "Je ne trouve pas cette information dans la documentation. Je transmets à un humain."}

def router(state: State):
    if "escalade" in state["answer"].lower():
        return "escalate"
    return "generate_ok"

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_node("escalate", escalate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve" ,"generate")

graph.add_conditional_edges("generate" , router ,{"escalate":"escalate" , "generate_ok" : END})
graph.add_edge("escalate", END)


app = graph.compile()
if __name__ == "__main__":
    result = app.invoke({"question": "Explique ce qu'est LangGraph"})
    print(result["answer"])