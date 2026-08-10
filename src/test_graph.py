from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def node_a(state: State):
    return {"message": state["message"] + " → passé par A"}

def node_b(state: State):
    return {"message": state["message"] + " → passé par B"}


def node_c(state: State):
    return {"message":state["message"]  + " -> passé par C"}


graph = StateGraph(State)
graph.add_node("A", node_a)
graph.add_node("B", node_b)

graph.add_node("C", node_c)
graph.add_edge(START, "A")
graph.add_edge("A", "C")

graph.add_edge("C", "B")
graph.add_edge("B", END)

app = graph.compile()
result = app.invoke({"message": "départ"})
print(result)