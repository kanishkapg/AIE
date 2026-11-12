from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from agent.math_agent import MODEL

@tool
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    print(f"*** EMAIL SENT to {to}: {subject} ***")
    return f"Email sent to {to}."

agent = create_agent(
    MODEL,
    tools=[send_email],
    checkpointer=InMemorySaver(),      # required: the paused run must be saved
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": {"allowed_decisions": ["approve", "reject"]}},
            interrupt_mode="per_call",
        )
    ],
)

config = {"configurable": {"thread_id": "email-1"}}
question = {"messages": [{"role": "user", "content": "Email sam@example.com with subject 'Meeting moved'."}]}

# 1. The agent runs until it wants the tool, then pauses.
result = agent.invoke(question, config, version="v2")

for interrupt in result.interrupts:
    print("Waiting for approval:", interrupt.value["name"], interrupt.value["args"])

# 2. You decide. Change "approve" to "reject" and run it again.
decisions = {interrupt.id: {"type": "approve"} for interrupt in result.interrupts}

# 3. The agent continues from where it paused.
result = agent.invoke(Command(resume=decisions), config, version="v2")
print(result.value["messages"][-1].content)