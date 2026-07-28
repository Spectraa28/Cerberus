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
from cerberus.runtime.session import EventLog
import asyncio

async def main():
    config = load_config()
    event_log = EventLog()
    await event_log.connect()
    session_id = await event_log.create_session(label="concurrent test")

    registry = ToolRegistry()
    registry.register(ShellExecTool(default_timeout=config.tools.shell_default_timeout), category="shell")
    input_models = {"shell_exec": ShellExecInput}

    provider = get_provider(config, tier="fast")

    async def run_sub_agent(agent_id: str, message: str):
        runtime = Runtime(provider, registry, input_models, max_turns=3, event_log=event_log)
        ctx = AgentContext(agent_id=agent_id, session_id=session_id, cwd=".")
        return await runtime.run(f"Run `echo {message}` and tell me what it printed.", ctx)

    # THREE sub-agents writing to the SAME session concurrently
    results = await asyncio.gather(
        run_sub_agent("sub-agent-a", "hello from A"),
        run_sub_agent("sub-agent-b", "hello from B"),
        run_sub_agent("sub-agent-c", "hello from C"),
    )
    for r in results:
        print(r)

    print("\n--- Full interleaved replay ---")
    events = await event_log.replay_from(session_id, from_seq=0)
    for e in events:
        print(f"[{e['seq']}] {e['type']} ({e['agent_id']})")

    await event_log.close()
    
if __name__ == "__main__":
    asyncio.run(main())