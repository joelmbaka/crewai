
# FastAPI + CrewAI Template

from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel, Field
from typing import Dict, Any
# from crews.your_crew_name import your_crew_instance

app = FastAPI(title="CrewAI API Server", version="1.0.0")

@app.get("/", tags=["Health"])
async def root():
    return {"message": "Hello from CrewAI-powered API server!"}

# Pydantic Models
class CrewRequest(BaseModel):
    task_description: str = Field(..., description="Task to execute")
    context: str = Field(default="", description="Additional context")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extra parameters")

class CrewResponse(BaseModel):
    result: str = Field(..., description="Crew execution result")
    status: str = Field(default="completed", description="Execution status")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# API Endpoints
@app.post("/execute-crew", response_model=CrewResponse, tags=["Crew Execution"])
async def execute_crew(request: CrewRequest):
    """Execute CrewAI crew with inputs."""
    try:
        crew_inputs = request.model_dump()  # Convert to dict for CrewAI
        # result = your_crew_instance.kickoff(inputs=crew_inputs)
        result = f"Crew executed: {request.task_description}"
        
        return CrewResponse(result=result, status="completed", metadata={"inputs": crew_inputs})
    except Exception as e:
        return CrewResponse(result=f"Error: {str(e)}", status="error", metadata={"error": str(e)})


def main() -> None:
    """Run FastAPI server with Uvicorn."""
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

"""
Terminal Commands:
- python main.py
- uvicorn main:app --reload --port 8000
- uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

Docs: http://localhost:8000/docs
"""

if __name__ == "__main__":
    main()
