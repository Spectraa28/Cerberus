from cerberus.tools.registry import ToolRegistry
from cerberus.tools.base import AgentContext
from cerberus.tools.adapter import tools_to_api_schema
from cerberus.providers.base import Provider, UserTurn, AssistantTurn, ToolResultTurn, ToolCall, Turn
from cerberus.runtime.session import EventLog
from cerberus.runtime.compaction import compact_if_needed


async def reconstruct_history(event_log: EventLog, session_id: str, agent_id: str) -> list[Turn]:
    """
    Rebuild a Turn history for one agent from the event log, starting from
    its latest summary (if any) rather than from the beginning — shared by
    Runtime.resume() and the gateway's spawn endpoint so both stay in sync.
    """
    all_events = await event_log.replay_from(session_id)
    agent_events = [e for e in all_events if e["agent_id"] == agent_id]

    last_summary_seq = -1
    summary_text = None
    for e in agent_events:
        if e["type"] == "summary":
            last_summary_seq = e["seq"]
            summary_text = e["payload"]["summary"]

    history: list[Turn] = []
    if summary_text:
        history.append(UserTurn(content=f"[Summary of earlier conversation]\n{summary_text}"))

    for e in agent_events:
        if e["seq"] <= last_summary_seq:
            continue
        if e["type"] == "user":
            history.append(UserTurn(content=e["payload"]["content"]))
        elif e["type"] == "assistant":
            tool_calls = [ToolCall(**tc) for tc in e["payload"].get("tool_calls", [])]
            history.append(AssistantTurn(text=e["payload"].get("text"), tool_calls=tool_calls))
        elif e["type"] == "tool_result":
            history.append(ToolResultTurn(results=e["payload"]["results"]))

    return history


class Runtime:
    def __init__(
        self,
        provider: Provider,
        registry: ToolRegistry,
        input_models: dict[str, type],
        max_turns: int = 10,
        event_log: EventLog | None = None,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.input_models = input_models
        self.max_turns = max_turns
        self.event_log = event_log
        self.tool_schemas = tools_to_api_schema(registry, input_models)

    async def _log(self, ctx: AgentContext, event_type: str, payload: dict) -> None:
        if self.event_log:
            await self.event_log.append_event(ctx.session_id, ctx.agent_id, event_type, payload)

    async def _dispatch_tools(self, tool_calls: list[ToolCall], ctx: AgentContext) -> list[dict]:
        results = []
        for tc in tool_calls:
            tool = self.registry.get(tc.name)
            model = self.input_models[tc.name]
            tool_input = model(**tc.input)
            result = await tool.run(tool_input, ctx)
            results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "output": result.output if result.ok else f"ERROR: {result.error}",
                "is_error": not result.ok,
            })
        return results

    async def _loop(self, history: list[Turn], ctx: AgentContext) -> str:
        for _ in range(self.max_turns):
            history, summary_text = await compact_if_needed(history, self.provider)
            if summary_text:
                await self._log(ctx, "summary", {"summary": summary_text})

            response = await self.provider.call(history, self.tool_schemas)

            if response.stop_reason == "end":
                await self._log(ctx, "assistant", {"text": response.text})
                return response.text or ""

            history.append(AssistantTurn(text=response.text, tool_calls=response.tool_calls))
            await self._log(ctx, "assistant", {
                "text": response.text,
                "tool_calls": [tc.model_dump() for tc in response.tool_calls],
            })

            results = await self._dispatch_tools(response.tool_calls, ctx)
            history.append(ToolResultTurn(results=results))
            await self._log(ctx, "tool_result", {"results": results})

        return "(max turns reached without a final answer)"

    async def run(self, task: str, ctx: AgentContext, seed_history: list[Turn] | None = None) -> str:
        history: list[Turn] = list(seed_history or []) + [UserTurn(content=task)]
        await self._log(ctx, "user", {"content": task})
        return await self._loop(history, ctx)

    async def resume(self, ctx: AgentContext) -> str:
        if not self.event_log:
            raise RuntimeError("resume() requires an event_log")

        history = await reconstruct_history(self.event_log, ctx.session_id, ctx.agent_id)

        if history and isinstance(history[-1], AssistantTurn) and history[-1].tool_calls:
            results = await self._dispatch_tools(history[-1].tool_calls, ctx)
            history.append(ToolResultTurn(results=results))
            await self._log(ctx, "tool_result", {"results": results})

        return await self._loop(history, ctx)