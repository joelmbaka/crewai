from crewai import Agent, Task, Crew, Process
from llms import llama_70b, llama_scout

fitness_coach = Agent(
    role="fitness coach",
    goal="Generate personalized workout plans based on the user's goals, schedule, and equipment availability.",
    backstory="You are a certified personal trainer with expertise in designing effective exercise programs.",
    llm=llama_70b,
    verbose=True,
)

plan_workout_task = Task(
    description="Create a detailed workout plan for the next {period} tailored to the user's preferences: {preferences}. The plan should specify exercises, sets, reps, rest, and include rest days.",
    expected_output="A structured, day-by-day workout schedule including exercises, sets, reps, rest periods, and progression guidance.",
    agent=fitness_coach,
)

exercise_planner_crew = Crew(
    agents=[fitness_coach],
    tasks=[plan_workout_task],
    verbose=True,
)

def main():
    """Entry point for running this script as a standalone module."""
    result = exercise_planner_crew.kickoff(inputs={
        "period": "5 days",
        "preferences": "Goal: muscle gain; Equipment: dumbbells and pull-up bar; Sessions per day: 1"
    })
    print(result)


if __name__ == "__main__":
    main()
