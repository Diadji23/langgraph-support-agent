from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

load_dotenv()

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")
embeddings = NVIDIAEmbeddings(model="nvidia/nemotron-3-embed-1b")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

class State(TypedDict):
    question: str
    context: str      # les chunks récupérés
    verdict :str 
    answer: str


def retrieve(state: State):
    #  cherche les chunks pertinents pour state["question"]
    
    query = state["question"]
    sims = vectorstore.similarity_search( query, k =3) 
    pages =[]
    for doc  in sims: 
        pages.append(doc.page_content)
    context = " ".join(pages)
    return { "context" :context}


def generate(state: State):
    prompt = f"""Tu es un assistant qui répond à des questions sur LangGraph.
Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
Si le contexte ne contient pas la réponse, dis-le clairement au lieu d'inventer.

Contexte :
{state["context"]}

Question : {state["question"]}

Réponse :"""

    res = llm.invoke(prompt)
    return {"answer": res.content}

def grade(state: State):
    prompt = f"""Tu vérifies si un contexte contient de quoi répondre à une question.
Sois généreux : si le contexte parle du sujet de la question, réponds "oui".
Ne réponds "non" QUE si le contexte est totalement hors-sujet.

Réponds par un seul mot : oui ou non.

Contexte :
{state["context"]}

Question : {state["question"]}

Un seul mot :"""
    res = llm.invoke(prompt)
    verdict = res.content.strip().lower()
    print("=== VERDICT ===", repr(verdict))
    return {"verdict": verdict}


def escalate(state: State):
    return {"answer": "Je ne trouve pas cette information dans la documentation. Je transmets à un humain."}


def router(state: State):
    if "oui" in state["verdict"]:
        return "generate"
    return "escalate"

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_node("grade", grade)
graph.add_node("escalate", escalate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve" ,"grade")

graph.add_conditional_edges("grade" , router ,{"generate":"generate" , "escalate" : "escalate"})
graph.add_edge("generate", END)
graph.add_edge("escalate", END)


app = graph.compile()
result = app.invoke({"question": "Explique ce qu'est LangGraph"})
print(result["answer"])