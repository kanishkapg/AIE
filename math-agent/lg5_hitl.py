from typing import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


# 1. Define the graph state
class State(TypedDict):
    action: str
    status: str


# 2. Define a node that pauses using interrupt()
def approval_node(state: State):
    # IMPORTANT: Anything above interrupt() re-runs when the graph resumes.
    # Keep code before interrupt() idempotent (no duplicate DB writes or API calls).

    # Pause execution and pass a JSON-serializable payload to the caller.
    # Optional parameter: `response_schema` (e.g., a Pydantic model or TypedDict)
    # can be passed to validate and structure the resume input.
    is_approved = interrupt(
        {
            "question": "Approve this action?",
            "action": state["action"],
        }
    )

    # When resumed via Command(resume=...), `is_approved` receives that value.
    if is_approved:
        return {"status": f"Executed: {state['action']}"}
    return {"status": "Cancelled by user"}


# 3. Build and compile the graph
builder = StateGraph(State)
builder.add_node("approval", approval_node)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

# REQUIRED: A checkpointer must be attached to persist state across pauses.
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# REQUIRED: A unique thread_id links the paused run to the resumed run.
config = {"configurable": {"thread_id": "approval-thread-1"}}

# --- Step 1: Run until the interrupt ---
result = graph.invoke({"action": "Delete old logs", "status": "pending"}, config)

# Inspect the payload passed to interrupt() via the special "__interrupt__" key
print("Paused with payload:", result["__interrupt__"][0].value)

# --- Step 2: Resume execution ---
# Pass the human's response into Command(resume=...).
# Other useful Command parameters:
# - update={"status": "..."} -> modify graph state directly while resuming
# - goto="another_node"       -> override the next node to execute
final_result = graph.invoke(Command(resume=True), config)

print("Final state:", final_result)
