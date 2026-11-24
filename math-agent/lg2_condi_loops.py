from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    ticket: str
    team: str
    reply: str

def classify(state: State) -> dict:
    text = state["ticket"].lower()
    if ("refund" in text or "charge" in text):
        team = "billing"
    elif("price" in text):
        team = "sales"
    else:
        team = "tech"
    return {"team": team}

def route(state: State) -> Literal["billing", "tech", "sales"]: # return one of these exact strings, no other string
    return state["team"] # --> "billing" or --> "tech", etc

def billing(state: State) -> dict:
    return {"reply": "Billing team: your refund is on its way."}

def tech(state: State) -> dict:
    return {"reply": "Tech team: try restarting the app."}

def sales(state: State) -> dict:
    return {"reply": "Sales: price details are mailed"}

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", billing)
builder.add_node("tech", tech)
builder.add_node("sales", sales)
builder.add_edge(START, "classify")
# builder.add_conditional_edges(
#     "classify", 
#     route,
#     {
#         "billing": "billing",
#         "tech": "tech"
#     }
# )
builder.add_conditional_edges("classify", route)
builder.add_edge("billing", END)
builder.add_edge("tech", END)
builder.add_edge("sales", END)
graph = builder.compile()

print(graph.invoke({"ticket": "I was charged twice!"})["reply"])
print(graph.invoke({"ticket": "The app crashes on login"})["reply"])
print(graph.invoke({"ticket": "What is the price of the TV"})["reply"])

graph_img = graph.get_graph().draw_mermaid_png()
with open("graph_img2.png", "wb") as f:
    f.write(graph_img)


# Conditional Loops

class State2(TypedDict):
    draft: str
    tries: int

def write(state: State2) -> dict:
    tries = state["tries"] + 1
    return {"draft": "word " * (tries * 3), "tries": tries}

def good_enough(state: State2) -> str:
    if len(state["draft"].split()) >= 100 or state["tries"] >= 5:
        return END
    return "write"

builder2 = StateGraph(State2)
builder2.add_node("write", write)
builder2.add_edge(START, "write")
builder2.add_conditional_edges("write", good_enough, ["write", END])
graph2 = builder2.compile()

for step in graph2.stream({"draft": "", "tries": 0}):
    print(step)

graph2_img = graph2.get_graph().draw_mermaid_png()
with open("graph_img3.png", "wb") as f:
    f.write(graph2_img)