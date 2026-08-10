from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str

def node_a(state: State):
    return {"message": state["message"] + " → passé par A"}
3632
def node_b(state: State):
    return {"message": state["message"] + " → passé par B"}

graph = StateGraph(State)
graph.add_node("A", node_a)
graph.add_node("B", node_b)
graph.add_edge(START, "A")
graph.add_edge("A", "B")
graph.add_edge("B", END)

app = graph.compile()
result = app.invoke({"message": "départ"})
print(result)