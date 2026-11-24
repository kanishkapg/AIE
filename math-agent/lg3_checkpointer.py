from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from agent.config import GROQ_API_KEY


model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

def chatbot(state: MessagesState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
graph = builder.compile(checkpointer=InMemorySaver())

config1 = {"configurable": {"thread_id": "chat-1"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Kanishka"}]}, config1)
graph.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config1)

snap = graph.get_state(config1)
print("values :", [m.content for m in snap.values["messages"]])
print("next   :", snap.next)
print("step   :", snap.metadata["step"])
print()
for s in graph.get_state_history(config1):
    print(s.metadata["step"], s.metadata["source"], s.next, len(s.values.get("messages", [])), "messages")

print("\n==========================================================\n")

config2 = {"configurable": {"thread_id": "chat-2"}}
graph.invoke({"messages": [{"role": "user", "content": "Hi, my name is Alice"}]}, config2)
graph.invoke({"messages": [{"role": "user", "content": "I am 25 years old"}]}, config2)
result2 = graph.invoke({"messages": [{"role": "user", "content": "What is my name and age?"}]}, config2)
print("response: ", result2["messages"][-1].content)

# Edit the state by hand, as if a node had written it
graph.update_state(config2, {"messages": [{"role": "user", "content": "I am 30 years old"}]})
result3 = graph.invoke({"messages": [{"role": "user", "content": "What is my name and age?"}]}, config2)
print("After update: ", result3["messages"][-1].content)

# History of chat-2
history2 = list(graph.get_state_history(config2))
print("\nHistory of chat-2:")
for s in history2:
    print(f"step {s.metadata['step']} | {s.metadata['source']} | next={s.next}")
    for m in s.values.get("messages", []):
        print(f"   {m.type:>5}: {m.content}")
    print()

print("----------------------------------------------------------\n")

# Go back in time: pick an older checkpoint and continue from it
first_turn = next(s for s in history2 if len(s.values.get("messages", [])) == 2)
fork = graph.invoke({"messages": [{"role": "user", "content": "What is my name and age?"}]}, first_turn.config)
print("forked      :", fork["messages"][-1].content)
print("latest now  :", [m.content for m in graph.get_state(config2).values["messages"]])
print("checkpoints :", len(list(graph.get_state_history(config2))))

new_config = graph.update_state(
    first_turn.config,
    {"messages": [{"role": "user", "content": "I am 35 years old"}]}
)
graph.invoke({"messages": [{"role": "user", "content": "I live in New York"}]}, new_config) # forks here
new_result = graph.invoke({"messages": [{"role": "user", "content": "What is my name and age and address?"}]}, config2) # continues on the fork
print("new fork :", new_result["messages"][-1].content)