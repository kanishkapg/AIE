from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(override=True)

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

prompt = ChatPromptTemplate.from_messages([
    {"role": "system", "content": "You are a helpful assistant that can explain the concepts clearly and simply."},
    {"role": "human", "content": "{concept}"}
])

chain = prompt | model | StrOutputParser()

if __name__ == "__main__":
    response = chain.invoke({"concept": "What are the pillars of 'quality data' in machine learning?"})
    print(response)