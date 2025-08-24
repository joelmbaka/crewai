
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from crews.exercise_planner import exercise_planner_crew

app = FastAPI()


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint used as a basic health-check."""
    return {"message": "Hello from ai-agents-server!"}


# Endpoint for generating exercise plans
class ExercisePlanRequest(BaseModel):
    period: str
    preferences: str


class ExercisePlanResponse(BaseModel):
    plan: str


@app.post("/exercise-plan", response_model=ExercisePlanResponse, tags=["Exercise Planner"])
async def generate_exercise_plan(request: ExercisePlanRequest):
    """Generate a personalized workout plan using the Exercise Planner crew."""
    result = exercise_planner_crew.kickoff(inputs=request.dict())
    return {"plan": result}


def main() -> None:
    """Run the FastAPI application using Uvicorn."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# cli: uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    main()
