from typing import Literal, Protocol, Any
from pydantic import BaseModel

class ToolCall(BaseModel):
    id: str
    name: str
    input: dict
    thought_signature: str | None = None  # Gemini 3-specific; unused by Anthropic/OpenAI
    
class AssistantTurn(BaseModel):
    role: Literal["assistant"] = "assistant"
    text: str| None = None
    tool_calls: list[ToolCall] = []
    
class UserTurn(BaseModel):
    role: Literal["user"] = "user"
    content: str
    
class ToolResultTurn(BaseModel):
    role: Literal["tool_result"] = "tool_result"
    results: list[dict]

Turn = UserTurn | AssistantTurn | ToolResultTurn

class NormalizedResponse(BaseModel):
    text: str | None
    tool_calls: list[ToolCall]
    stop_reason: Literal["tool_use", "end"]


class Provider(Protocol):
    model: str

    async def call(self, history: list[Turn], tool_schemas: list[dict]) -> NormalizedResponse: ...