from agent.agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

def main():
    print("Research Agent - Phase 01")
    print("Type 'quit' to exit.\n")

    chat_history = []

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            print("Exiting. Goodbye!")
            break

        result = run_agent(question, chat_history)
        answer = result.get("output", "Sorry, I couldn't generate an answer.")

        print(f"\nAgent: {answer}\n")
        print("-" * 50 + "\n")

        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    main()

