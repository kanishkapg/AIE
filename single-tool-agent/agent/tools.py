import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient
from agent.config import TAVILY_API_KEY

_tavily = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information about a topic.
    Use this when you need facts, recent events, or any information
    you don't already know. Returns a list of relevant results with
    titles, URLs, and summaries.

    Args:
        query: A clear, specific search query string.
    """

    try:
        results = _tavily.search(query=query, max_results=3)
        formatted = []
        for r in results.get("results", []):
            formatted.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Summary: {r['summary']}\n"
            )
        return "\n\n".join(formatted) if formatted else "No results found."
    except Exception as e:
        return f"Error during web search: {str(e)}"
    

@tool
def read_url(url: str) -> str:
    """
    Fetch and read the full text content of a webpage.
    Use this when a search result looks relevant but you need
    more detail than the summary provides — for example to
    get exact figures, quotes, or step-by-step instructions.

    Args:
        url: The full URL of the page to read (must include https://).
    """

    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise - scripts, styles, nav, footer, etc.
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Truncate to avoid blowing the context window
        return text[:4000] if len(text) > 4000 else text
    
    except requests.RequestException as e:
        return f"HTTP error fetching URL: {str(e)}"

    except Exception as e:
        return f"Error reading page: {str(e)}"
