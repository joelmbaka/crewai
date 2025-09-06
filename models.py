from pydantic import BaseModel, Field
from typing import List, Optional


class ChatResponse(BaseModel):
    """Example output model for chat task."""
    response: str = Field(..., description="The chatbot's response to the user")
    confidence: float = Field(..., description="Confidence score between 0 and 1")
    topics: List[str] = Field(default_factory=list, description="Topics identified in the conversation")
    follow_up_questions: Optional[List[str]] = Field(default=None, description="Suggested follow-up questions")


class ExampleRequest(BaseModel):
    period: str
    preferences: str


class ExampleResponse(BaseModel):
    plan: str
