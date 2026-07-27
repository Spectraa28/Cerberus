import  asyncio
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool  , ShellExecInput
from cerberus.tools.base import AgentContext
from cerberus.tools.search import SearchFilesTool, SearchFilesInput
from dotenv import load_dotenv

load_dotenv()

from cerberus.tools.search_web import SearchWebTool, SearchWebInput


async def main():
    registry = ToolRegistry()
    registry.register(ShellExecTool(),category="shell")
    registry.register(SearchFilesTool(), category="search")
    registry.register(SearchWebTool(), category="search")
    tool = registry.get("shell_exec")
    ctx = AgentContext(agent_id="test-agent", session_id="test-session", cwd=".")
    result = await tool.run(ShellExecInput(command="echo hello from cerberus"), ctx)
    print(result)
    tool2 = registry.get("search_files")
    result2 = await tool2.run(SearchFilesInput(pattern="ToolResult", file_glob="*.py"), ctx)
    print(result2)
    tool3 = registry.get("search_web")
    result3 = await tool3.run(SearchWebInput(query="ripgrep vs grep performance"), ctx)
    print(result3)

if __name__ == "__main__":
    asyncio.run(main())