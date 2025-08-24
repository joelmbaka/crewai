from crewai import Agent, Task, Crew, Process
from llms import llama_70b

chatbot = Agent(
    role="chat with a user",
    goal="answer the questions you are asked ina  friendly manner",
    backstory="You are a friendly chatbot.",
    llm=llama_70b
)

chat = Task(
    description="chat task",
    expected_output="A clear response to the user's question.",
    agent=chatbot,
)

crew = Crew(
    agents=[chatbot],
    tasks=[chat],
)

def main():
    """Entry point for running this script as a standalone module."""
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    main()
