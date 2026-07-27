from typing import Literal , Protocol , runtime_checkable
from pydantic import BaseModel

class ToolResult(BaseModel):
    ok:bool
    output: str
    error: str | None = None
    
class ToolSpec(BaseModel):
    name: str
    description:str
    permission: Literal["read","write","exec"]
    
@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec
    
    async def run(self,input:BaseModel, ctx: "AgentContext") -> ToolResult: ...

class AgentContext(BaseModel):
    """Minimal context passed into every tool call."""
    agent_id: str
    session_id: str
    cwd: str
    