from agent.agent import run_agent

if __name__ == "__main__":

    q1 = "What is the sum of 5 and 3?"

    response1 = run_agent(q1)
    print(f"Question: {q1}\nResponse: {response1}\n")