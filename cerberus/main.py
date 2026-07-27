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

    # Parent agent: pro tier, full reasoning
    parent_provider = get_provider(config, tier="pro")
    parent_runtime = Runtime(parent_provider, registry, input_models, max_turns=config.runtime.max_turns)

    # Sub-agent: fast tier, scoped to shell tools only (registry.filtered from earlier)
    sub_provider = get_provider(config, tier="fast")
    sub_registry_view = registry.filtered("shell_")
    # Runtime currently takes a full ToolRegistry — for a filtered sub-agent you'd
    # pass a lightweight wrapper exposing only .get()/.all() over sub_registry_view,
    # or extend ToolRegistry with a `from_dict` constructor. Worth doing when we
    # actually build spawning — flagging it now so it's not a surprise later.

    ctx = AgentContext(agent_id="agent-1", session_id="session-1", cwd=".")
    answer = await parent_runtime.run(
        "List the Python files in the current project using search_files, then tell me how many there are.",
        ctx,
    )
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())