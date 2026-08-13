from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    message: str
    count : int 



def node_increment(state  :State):
    state["count"] += 1 
    return {"count": state["count"]}

def router(state : State): # edge ?
    if state["count"] < 3 : 
        return "increment"
    return "end"




graph = StateGraph(State)

graph.add_node("increment" ,node_increment)

graph.add_edge(START, "increment")


graph.add_conditional_edges("increment", router, {"increment": "increment", "end": END})

app = graph.compile()
result = app.invoke({"message": "départ" 
            , "count" : 0})
print(result)