from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from agent.config import GROQ_API_KEY, MODEL_NAME, MAX_ITERATIONS
from agent.tools import web_search, read_url

TOOLS = [web_search, read_url]

SYSTEM_PROMPT = """You are a research assistant with access to web search and URL reading tools.

Your approach:
1. Think carefully about what the user is asking before acting.
2. Search for information when you don't already know something reliably.
3. Read URLs when a search result looks highly relevant but needs more detail.
4. Synthesize what you find into a clear, accurate answer.
5. Always cite where key facts came from.

Be concise. Do not over-search — if you have a confident answer, give it."""

def build_agent():
    """Construct and return the compiled agent graph."""

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=0,  # Deterministic — critical for tool calling reliability
        max_tokens=1024
    )

    # Replaces AgentExecutor's max_iterations. 'end' stops cleanly at the cap
    # instead of raising, so the user still gets whatever answer exists.
    step_limit = ModelCallLimitMiddleware(
        run_limit=MAX_ITERATIONS,
        exit_behavior="end",
    )

    # No prompt template needed — create_agent builds the message list itself,
    # so there is no agent_scratchpad or chat_history placeholder to wire up.
    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[step_limit],
    )

def run_agent(question: str, chat_history: list, verbose: bool = True) -> dict:
    """
    Run the agent on a single question.
    Returns the full result including intermediate steps for inspection.
    """

    agent = build_agent()

    # History and the new question go in as one message list — the graph
    # appends tool calls and results to this same list as it runs.
    messages = list(chat_history or [])
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})

    if verbose:  # Set False in production — shows the full tool-calling loop in dev
        for message in result["messages"][len(messages):]:
            message.pretty_print()

    # Keep the old AgentExecutor key so callers reading "output" still work.
    final_message = result["messages"][-1]
    return {
        "output": final_message.content,
        "messages": result["messages"],
    }
