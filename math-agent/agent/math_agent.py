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


def run_agent(question: str) -> str:
    agent = build_agent()
    config = {"configurable": {"thread_id": "user-1"}}
    response = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
    return response["messages"]

if __name__ == "__main__":
    result = run_agent("Multiply 6 by 3.7 and the add 12 to it. After that take the square root of the result. Finally divide it by -2.3")
    for message in result:
        message.pretty_print()