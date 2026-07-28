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
    session_id = await event_log.create_session(label="resume test")

    registry = ToolRegistry()
    registry.register(ShellExecTool(default_timeout=config.tools.shell_default_timeout), category="shell")
    input_models = {"shell_exec": ShellExecInput}

    provider = get_provider(config, tier="fast")
    ctx = AgentContext(agent_id="agent-1", session_id=session_id, cwd=".")

    # --- Simulate a crash: manually write a user turn + an assistant turn
    # that made a tool call, but NEVER write the tool_result. This is exactly
    # what the log would look like if the process died right after the model
    # decided to call shell_exec, before the tool actually ran.
    await event_log.append_event(session_id, ctx.agent_id, "user", {
        "content": "Run `echo resumed successfully` and tell me what it printed."
    })
    await event_log.append_event(session_id, ctx.agent_id, "assistant", {
        "text": None,
        "tool_calls": [{"id": "fake_call_1", "name": "shell_exec", "input": {"command": "echo resumed successfully"}}],
    })
    print("--- Simulated crash: log ends mid tool-call, no tool_result was ever written ---")

    # --- Now bring up a BRAND NEW Runtime, as if this were a fresh process,
    # and resume purely from the log ---
    fresh_runtime = Runtime(provider, registry, input_models, max_turns=config.runtime.max_turns, event_log=event_log)
    answer = await fresh_runtime.resume(ctx)
    print("--- Resumed answer ---")
    print(answer)

    print("\n--- Full replay after resume ---")
    events = await event_log.replay_from(session_id)
    for e in events:
        print(f"[{e['seq']}] {e['type']} ({e['agent_id']}): {e['payload']}")

    await event_log.close()
    
if __name__ == "__main__":
    asyncio.run(main())