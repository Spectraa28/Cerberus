from cerberus.tools.registry import ToolRegistry
from cerberus.tools.base import AgentContext
from cerberus.tools.adapter import tools_to_api_schema
from cerberus.providers.base import Provider, UserTurn, AssistantTurn, ToolResultTurn, Turn
from cerberus.runtime.session import EventLog


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

    async def run(
        self,
        task: str,
        ctx: AgentContext,
        seed_history: list[Turn] | None = None,
    ) -> str:
        history: list[Turn] = list(seed_history or []) + [UserTurn(content=task)]

        if self.event_log:
            await self.event_log.append_event(ctx.session_id, ctx.agent_id, "user", {"content": task})

        for _ in range(self.max_turns):
            response = await self.provider.call(history, self.tool_schemas)

            if response.stop_reason == "end":
                if self.event_log:
                    await self.event_log.append_event(
                        ctx.session_id, ctx.agent_id, "assistant", {"text": response.text}
                    )
                return response.text or ""

            history.append(AssistantTurn(text=response.text, tool_calls=response.tool_calls))
            if self.event_log:
                await self.event_log.append_event(
                    ctx.session_id, ctx.agent_id, "assistant",
                    {"text": response.text, "tool_calls": [tc.model_dump() for tc in response.tool_calls]},
                )

            results = []
            for tc in response.tool_calls:
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

            history.append(ToolResultTurn(results=results))
            if self.event_log:
                await self.event_log.append_event(ctx.session_id, ctx.agent_id, "tool_result", {"results": results})

        return "(max turns reached without a final answer)"