from cerberus.tools.registry import ToolRegistry
from cerberus.tools.base import AgentContext
from cerberus.tools.adapter import tools_to_api_schema
from cerberus.providers.base import Provider, UserTurn, AssistantTurn, ToolResultTurn, Turn


class Runtime:
    def __init__(self, provider, registry, input_models, max_turns: int = 10) -> None:
        self.provider = provider
        self.registry = registry
        self.input_models = input_models
        self.max_turns = max_turns
        self.tool_schemas = tools_to_api_schema(registry, input_models)

    async def run(self, task: str, ctx: AgentContext, max_turns: int = 10) -> str:
        history: list[Turn] = [UserTurn(content=task)]

        for _ in range(max_turns):
            response = await self.provider.call(history, self.tool_schemas)

            if response.stop_reason == "end":
                return response.text or ""

            history.append(AssistantTurn(text=response.text, tool_calls=response.tool_calls))

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

        return "(max turns reached without a final answer)"