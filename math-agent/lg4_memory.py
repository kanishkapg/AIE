from dataclasses import dataclass
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

def remember(state: MessagesState, runtime: Runtime[Context]) -> dict:
    text = state["messages"][-1].content
    namespace = ("memories", runtime.context.user_id)    # a folder per user
    if text.lower().startswith("remember:"):
        fact = text.split(":", 1)[1].strip()
        runtime.store.put(namespace, f"fact-{len(runtime.store.search(namespace))}", {"fact": fact})
        return {"messages": [{"role": "ai", "content": f"Saved: {fact}"}]}
    facts = [item.value["fact"] for item in runtime.store.search(namespace)]
    return {"messages": [{"role": "ai", "content": f"What I know about you: {facts}"}]}

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node("remember", remember)
builder.add_edge(START, "remember")
builder.add_edge("remember", END)

store = InMemoryStore()
graph = builder.compile(checkpointer=InMemorySaver(), store=store)

def say(text, thread, user):
    r = graph.invoke({"messages": [{"role": "user", "content": text}]},
                     {"configurable": {"thread_id": thread}},
                     context=Context(user_id=user))
    print(f"{user}/{thread}: {r['messages'][-1].content}")

say("remember: I prefer Python over JS", thread="monday", user="kanishka")
say("remember: I am learning LangGraph", thread="monday", user="kanishka")
say("what do you know?", thread="friday", user="kanishka")   # new thread, same user
say("what do you know?", thread="friday-2", user="ravi")     # different user

print(store.get(("memories", "kanishka"), "fact-0").value)