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
from cerberus.providers.base import UserTurn, AssistantTurn, ToolResultTurn
from cerberus.tools.adapter import tools_to_api_schema

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

    print("--- Test 3: context_seeded with a tool the sub-agent doesn't have ---")

    parent_provider = get_provider(config, tier="pro")
    parent_runtime = Runtime(parent_provider, registry, input_models, max_turns=config.runtime.max_turns)

    # Run the parent so it actually calls search_files and builds real history
    parent_ctx = AgentContext(agent_id="parent-1", session_id="session-1", cwd=".")
    parent_history: list = [UserTurn(content="Use search_files to list Python files in cerberus/, then summarize what you found.")]

    response = await parent_provider.call(parent_history, tools_to_api_schema(registry, input_models))
    parent_history.append(AssistantTurn(text=response.text, tool_calls=response.tool_calls))

    if response.tool_calls:
        tc = response.tool_calls[0]
        tool = registry.get(tc.name)
        model = input_models[tc.name]
        result = await tool.run(model(**tc.input), parent_ctx)
        parent_history.append(ToolResultTurn(results=[{
            "tool_call_id": tc.id, "name": tc.name,
            "output": result.output, "is_error": not result.ok,
        }]))

    # Now spawn a SHELL-ONLY sub-agent, seeded with history that references search_files
    sub_runtime_seeded, seed = spawn_sub_agent(
        sub_provider, registry, input_models,
        allowed_prefixes=["shell_"], mode="context_seeded",
        parent_history=parent_history,
        shell_allowed_commands={"echo"},
    )

    sub_ctx = AgentContext(agent_id="sub-agent-2", session_id="session-1", cwd=".")
    print(await sub_runtime_seeded.run("Given what you found, echo a one-line summary.", sub_ctx, seed_history=seed))

if __name__ == "__main__":
    asyncio.run(main())