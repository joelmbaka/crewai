# CrewAI Implementation Guide

This guide provides comprehensive instructions for creating and implementing CrewAI agents with Python code examples.

## Prerequisites

```bash
pip install crewai fastapi uvicorn pydantic
```

## 1. LLM Configuration (`llms.py`)

```python
from crewai import LLM

# Configure your LLM - replace with your actual LLM setup
llama_70b = LLM(
    model="meta/llama-4-scout-17b-16e-instruct",
    api_key=os.getenv("NVIDIA_NIM_API_KEY"),
    base_url="https://integrate.api.nvidia.com/v1"
)

```

## 2. Custom Tools (`tool.py`)

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class MyToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = "What this tool does. It's vital for effective utilization."
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, argument: str) -> str:
        # Your tool's logic here
        return "Tool's result"
```

## 3. Pydantic Models (`models.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ChatResponse(BaseModel):
    """Example output model for chat task."""
    response: str = Field(..., description="The chatbot's response to the user")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    topics: List[str] = Field(default_factory=list, description="Topics identified in the conversation")
    follow_up_questions: Optional[List[str]] = Field(default=None, description="Suggested follow-up questions")

class TaskRequest(BaseModel):
    period: str
    preferences: str

class TaskResponse(BaseModel):
    plan: str
```

## 4. Agent, Task, and Crew Setup (`crew.py`)

```python
from crewai import Agent, Task, Crew, Process
from llms import llama_70b
from tool import MyCustomTool
from models import ChatResponse

# Create Agent
chatbot = Agent(
    role="chat with a user",
    goal="answer the questions you are asked in a friendly manner",
    backstory="You are a friendly chatbot.",
    llm=llama_70b,
    tools=[MyCustomTool()]
)

# Create Task with Pydantic output
chat = Task(
    description="chat task",
    expected_output="A clear response to the user's question.",
    agent=chatbot,
    tools=[MyCustomTool()],
    output_pydantic=ChatResponse  # Structured output
)

# Create Crew
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
```

## 5. FastAPI Integration (`main.py`)

```python
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

if __name__ == "__main__":
    main()
```

## 6. Environment Configuration (`.env`)

```bash
# API Keys
NVIDIA_NIM_API_KEY=your_nvidia_nim_api_key

# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

## 7. Complete Implementation Steps

### Step 1: Setup LLM
```python
# llms.py
from crewai import LLM
llama_70b = LLM(model="ollama/llama3.1:70b", base_url="http://localhost:11434")
```

### Step 2: Define Tools
```python
# tool.py
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    argument: str = Field(..., description="Input argument")

class MyCustomTool(BaseTool):
    name: str = "Custom Tool"
    description: str = "Tool description"
    args_schema: Type[BaseModel] = MyToolInput
    
    def _run(self, argument: str) -> str:
        return f"Processed: {argument}"
```

### Step 3: Create Models
```python
# models.py
from pydantic import BaseModel, Field

class OutputModel(BaseModel):
    result: str = Field(..., description="Task result")
    confidence: float = Field(..., description="Confidence score")
```

### Step 4: Build Crew
```python
# crew.py
from crewai import Agent, Task, Crew
from llms import llama_70b
from tool import MyCustomTool
from models import OutputModel

agent = Agent(
    role="Task executor",
    goal="Execute tasks efficiently",
    backstory="You are an expert task executor.",
    llm=llama_70b,
    tools=[MyCustomTool()]
)

task = Task(
    description="Execute the given task",
    expected_output="Task completion result",
    agent=agent,
    output_pydantic=OutputModel
)

crew = Crew(agents=[agent], tasks=[task])
```

### Step 5: Execute Crew
```python
# Execute with inputs
result = crew.kickoff(inputs={
    "task": "Your task description",
    "context": "Additional context"
})
```

## 8. Terminal Commands

```bash
# Run crew directly
python crew.py

# Run FastAPI server
python main.py

# Run with uvicorn
uvicorn main:app --reload --port 8000

# Production (adjust workers based on CPU cores)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 5
```

## 9. Key Integration Points

1. **crew.kickoff(inputs=dict)** - Execute crew with dictionary inputs
2. **request.model_dump()** - Convert Pydantic to dict for CrewAI
3. **output_pydantic=Model** - Structured task outputs
4. **tools=[Tool()]** - Add tools to agents and tasks
5. **llm=your_llm** - Assign LLM to agents

## 10. File Structure

```
project/
├── .env                 # Environment variables
├── llms.py             # LLM configurations
├── tool.py             # Custom tools
├── models.py           # Pydantic models
├── crew.py             # Crew implementation
├── main.py             # FastAPI server
└── requirements.txt    # Dependencies
```

## 11. Error Handling Pattern

```python
try:
    crew_inputs = request.model_dump()
    result = crew.kickoff(inputs=crew_inputs)
    return {"result": result, "status": "success"}
except Exception as e:
    return {"error": str(e), "status": "error"}
```

This guide provides all necessary components to implement a One Agent, One Task CrewAI system with FastAPI integration.
