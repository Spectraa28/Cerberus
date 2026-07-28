import asyncio
from dotenv import load_dotenv
load_dotenv()

from cerberus.config import load_config
from cerberus.providers.factory import get_provider
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool, ShellExecInput
from cerberus.tools.base import AgentContext
from cerberus.runtime.agent import Runtime
from cerberus.runtime.session import EventLog


async def main():
    session_id = input("paste session_id: ")
    config = load_config()

    event_log = EventLog()  # same db_path as gateway — same underlying file
    await event_log.connect()

    registry = ToolRegistry()
    registry.register(ShellExecTool(default_timeout=config.tools.shell_default_timeout), category="shell")
    input_models = {"shell_exec": ShellExecInput}

    provider = get_provider(config, tier="fast")
    runtime = Runtime(provider, registry, input_models, max_turns=config.runtime.max_turns, event_log=event_log)

    ctx = AgentContext(agent_id="live-agent", session_id=session_id, cwd=".")
    answer = await runtime.run("Run `echo live broadcast works` and tell me what it printed.", ctx)
    print("answer:", answer)

    await event_log.close()

asyncio.run(main())