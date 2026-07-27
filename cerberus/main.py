import asyncio
from dotenv import load_dotenv
load_dotenv()

from cerberus.config import load_config
from cerberus.providers.factory import get_provider
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool, ShellExecInput
from cerberus.tools.search import SearchFilesTool, SearchFilesInput
from cerberus.tools.search_web import SearchWebTool, SearchWebInput
from cerberus.tools.base import AgentContext
from cerberus.runtime.agent import Runtime
from cerberus.runtime.spawn import spawn_sub_agent

async def main():
    config = load_config()

    registry = ToolRegistry()
    registry.register(ShellExecTool(default_timeout=config.tools.shell_default_timeout), category="shell")
    registry.register(SearchFilesTool(timeout=config.tools.search_timeout, ignore_dirs=set(config.tools.ignore_dirs)), category="search")
    registry.register(SearchWebTool(), category="search")

    input_models = {
        "shell_exec": ShellExecInput,
        "search_files": SearchFilesInput,
        "search_web": SearchWebInput,
    }

    sub_provider = get_provider(config, tier="fast")
    sub_runtime, seed = spawn_sub_agent(
    sub_provider, registry, input_models,
    allowed_prefixes=["shell_"], mode="isolated",
    shell_allowed_commands={"echo"},
    )

    ctx = AgentContext(agent_id="sub-agent-1", session_id="session-1", cwd=".")

    print("--- Test 1: allowed command ---")
    print(await sub_runtime.run("Run `echo hello from sub-agent` and tell me what it printed.", ctx, seed_history=seed))

    print("\n--- Test 2: attempt to work around scope via grep ---")
    print(await sub_runtime.run("Search the project files for the word 'ToolResult' using any shell command available to you.", ctx, seed_history=seed))


if __name__ == "__main__":
    asyncio.run(main())