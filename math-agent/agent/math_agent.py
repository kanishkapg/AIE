from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from agent.config import GROQ_API_KEY, MODEL
from agent.tools import add, multiply, devide, square_root
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

TOOLS = [add, multiply, devide, square_root]

SYSTEM_PROMPT = """
You are a helpful assistant that can perform mathematical operations.
1. Think carefully what is user asking, before acting.
2. If the user asks you to perform a calculation, use the appropriate tool to do so.
"""

MODEL = init_chat_model(
    "groq:openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
    temperature=0.7
)

def build_agent():
    agent = create_agent(
        tools=TOOLS,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )
    return agent


# Build the agent once when this module is imported, and reuse it for every call.
# This also keeps the same InMemorySaver alive, so conversation memory persists
# between calls that share a thread_id.
AGENT = build_agent()


def run_agent(question: str, thread_id: str = "user-1"):
    config = {"configurable": {"thread_id": thread_id}}
    response = AGENT.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return response["messages"]

if __name__ == "__main__":
    result1 = run_agent("Multiply 5 by 3.72 and the add 12 to it. After that take the square root of the result. Finally divide it by -2.3")
    result2 = run_agent("Multiply the result by 150")
    for message in result1:
        message.pretty_print()
    for message in result2:
        message.pretty_print()