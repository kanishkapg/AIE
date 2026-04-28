from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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

def build_agent() -> AgentExecutor:
    """Construct and return agent executor"""

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL_NAME,
        temperature=0,  # Deterministic — critical for tool calling reliability
        max_tokens=1024
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])

    agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        max_iterations=MAX_ITERATIONS,
        verbose=True,    # Set False in production — shows the full ReAct loop in dev
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

def run_agent(question: str, chat_history: list) -> str:
    """
    Run the agent on a single question.
    Returns the full result including intermediate steps for inspection.
    """

    executor = build_agent()
    result = executor.invoke({
        "input": question,
        "chat_history": chat_history or []
    })

    return result

