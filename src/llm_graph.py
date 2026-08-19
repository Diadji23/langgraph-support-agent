from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv 

load_dotenv()

llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

class State(TypedDict):
    question: str 
    answer : str

def call_llm(state: State):
    # À TOI : appelle le llm avec state["question"], renvoie la réponse dans "answer"
    # Indice : llm.invoke("du texte") renvoie un objet ; sa réponse est dans .content
    res = llm.invoke(state["question"]) 
    return {"answer":  res.content}



graph = StateGraph(State)
graph.add_node("llm", call_llm)
graph.add_edge(START, "llm")
graph.add_edge("llm", END)

app = graph.compile()
result = app.invoke({"question": "Explique ce qu'est LangGraph en une phrase." , "answer" : "test"})
print(result["answer"])