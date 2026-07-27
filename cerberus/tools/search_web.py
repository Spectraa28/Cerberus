import asyncio
import os
from pydantic import BaseModel
from tavily import TavilyClient
from cerberus.tools.base import ToolSpec, ToolResult, AgentContext

class SearchWebInput(BaseModel):
    query: str
    max_results: int = 5


class SearchWebTool:
    spec = ToolSpec(
        name="search_web",
        description="Search the web for current information via Tavily.",
        permission="read",
    )

    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self._client = TavilyClient(api_key=api_key)

    async def run(self, input: SearchWebInput, ctx: AgentContext) -> ToolResult:
        try:
            # tavily-python is sync, so run it off the event loop thread
            response = await asyncio.to_thread(
                self._client.search, input.query, max_results=input.max_results
            )
            results = response.get("results", [])
            if not results:
                return ToolResult(ok=True, output="(no results)", error=None)

            lines = [
                f"- {r['title']} ({r['url']})\n  {r['content'][:200]}"
                for r in results
            ]
            return ToolResult(ok=True, output="\n".join(lines), error=None)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))