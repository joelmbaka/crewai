from crewai import Agent, Task, Crew, Process
from llms import llama_70b
from tool import MyCustomTool
from models import ChatResponse

chatbot = Agent(
    role="chat with a user",
    goal="answer the questions you are asked ina  friendly manner",
    backstory="You are a friendly chatbot.",
    llm=llama_70b,
    tools=[MyCustomTool()]
)

chat = Task(
    description="chat task",
    expected_output="A clear response to the user's question.",
    agent=chatbot,
    tools=[MyCustomTool()],
    output_pydantic=ChatResponse
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
