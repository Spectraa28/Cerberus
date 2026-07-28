import asyncio
from cerberus.tools.base import AgentContext
from cerberus.runtime.agent import Runtime
from cerberus.runtime.spawn import spawn_sub_agent
from cerberus.providers.base import UserTurn, AssistantTurn, ToolResultTurn
from cerberus.tools.adapter import tools_to_api_schema


async def test_basic_tool_call(provider, registry, input_models, event_log):
    runtime = Runtime(provider, registry, input_models, max_turns=10, event_log=event_log)
    session_id = await event_log.create_session(label="basic")
    ctx = AgentContext(agent_id="parent", session_id=session_id, cwd=".")

    answer = await runtime.run("Run `echo integration test running` and tell me exactly what it printed.", ctx)
    assert "integration test running" in answer


async def test_subagent_permission_boundary(provider, registry, input_models, event_log):
    session_id = await event_log.create_session(label="permissions")
    sub_runtime, seed = spawn_sub_agent(
        provider, registry, input_models,
        allowed_prefixes=["shell_"], mode="isolated",
        shell_allowed_commands={"echo"},
    )
    ctx = AgentContext(agent_id="sub-restricted", session_id=session_id, cwd=".")

    ok_answer = await sub_runtime.run("Run `echo allowed` and report the output.", ctx, seed_history=seed)
    assert "allowed" in ok_answer

    blocked_answer = await sub_runtime.run("Use grep to search for 'ToolResult' in this project.", ctx, seed_history=seed)
    assert any(w in blocked_answer.lower() for w in ("not permitted", "cannot", "unable"))


async def test_context_seeded_provider_safe(provider, registry, input_models):
    parent_history = [UserTurn(content="Use search_files to list Python files in cerberus/.")]
    resp = await provider.call(parent_history, tools_to_api_schema(registry, input_models))
    parent_history.append(AssistantTurn(text=resp.text, tool_calls=resp.tool_calls))

    if resp.tool_calls:
        tc = resp.tool_calls[0]
        tool = registry.get(tc.name)
        model = input_models[tc.name]
        result = await tool.run(model(**tc.input), AgentContext(agent_id="parent", session_id="s", cwd="."))
        parent_history.append(ToolResultTurn(results=[{
            "tool_call_id": tc.id, "name": tc.name,
            "output": result.output, "is_error": not result.ok,
        }]))

    seeded_runtime, seed = spawn_sub_agent(
        provider, registry, input_models,
        allowed_prefixes=["shell_"], mode="context_seeded",
        parent_history=parent_history, shell_allowed_commands={"echo"},
    )
    ctx = AgentContext(agent_id="sub-seeded", session_id="s", cwd=".")
    answer = await seeded_runtime.run("Based on what you found, echo a one-word confirmation.", ctx, seed_history=seed)
    assert len(answer) > 0


async def test_concurrent_event_log_writes(provider, registry, input_models, event_log):
    session_id = await event_log.create_session(label="concurrent")

    async def run_one(agent_id: str, msg: str):
        r = Runtime(provider, registry, input_models, max_turns=3, event_log=event_log)
        c = AgentContext(agent_id=agent_id, session_id=session_id, cwd=".")
        return await r.run(f"Run `echo {msg}` and report the output.", c)

    results = await asyncio.gather(
        run_one("concurrent-a", "conc-A"),
        run_one("concurrent-b", "conc-B"),
        run_one("concurrent-c", "conc-C"),
    )
    assert all(len(r) > 0 for r in results)

    events = await event_log.replay_from(session_id)
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(len(seqs))), f"seq gaps/dupes: {seqs}"


async def test_crash_and_resume(provider, registry, input_models, event_log):
    session_id = await event_log.create_session(label="resume")
    ctx = AgentContext(agent_id="resume-agent", session_id=session_id, cwd=".")

    await event_log.append_event(session_id, ctx.agent_id, "user", {
        "content": "Run `echo resumed ok` and tell me what it printed."
    })
    await event_log.append_event(session_id, ctx.agent_id, "assistant", {
        "text": None,
        "tool_calls": [{"id": "sim_crash_1", "name": "shell_exec", "input": {"command": "echo resumed ok"}}],
    })

    fresh_runtime = Runtime(provider, registry, input_models, max_turns=10, event_log=event_log)
    answer = await fresh_runtime.resume(ctx)
    assert "resumed ok" in answer