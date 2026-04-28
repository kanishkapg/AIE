from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Get API keys from environment variables
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
MODEL_NAME = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 10