from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

def union_reducer(current:set[int], new:set[int]) -> set[int]:
    return (set.union(current, new))

class State(TypedDict):
    order: str
    status: str
    log: Annotated[list[str], operator.add] 
    chef: str
    temp_list: Annotated[set[int], union_reducer] # We could write custom reducer functions such as Union, according to the usecase of the application

def take_order(state: State) -> dict:
    return {"status": "ordered", "log": [f"Order received: {state['order']}"], "temp_list": [11, 2, 32]}

def cook(state: State) -> dict:
    return {"status": f"cooked {state['order']}", "chef": "Mr. Lloyd", "log": [f"Cooking {state['order']}"]}

def plate(state: State) -> dict:
    return {"status": f"{state['order']} is set on the plate", "log": [f"Plating {state['order']}"], "temp_list": [32, 5, 11, 21]}

builder = StateGraph(State)
builder.add_node("take_order", take_order)
builder.add_node("cook", cook)
builder.add_node("plate", plate)
builder.add_edge(START, "take_order")
builder.add_edge("take_order", "cook")
builder.add_edge("cook", "plate")
builder.add_edge("plate", END)
graph = builder.compile()

print(graph.invoke({"order": "Butter Naan"}))
graph_img = graph.get_graph().draw_mermaid_png()

with open("graph_img.png", "wb") as f:
    f.write(graph_img)
