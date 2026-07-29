import os
from anthropic import Anthropic
from cerberus.providers.base import Turn,Usage, UserTurn, AssistantTurn, ToolResultTurn, ToolCall, NormalizedResponse


class AnthropicProvider:
    def __init__(self, model: str, max_tokens: int = 1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _to_native_tools(self, tool_schemas: list[dict]) -> list[dict]:
        return tool_schemas  # already in Anthropic's shape

    def _to_native_messages(self, history: list[Turn]) -> list[dict]:
        messages = []
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                content = []
                if turn.text:
                    content.append({"type": "text", "text": turn.text})
                for tc in turn.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
                messages.append({"role": "assistant", "content": content})
            elif isinstance(turn, ToolResultTurn):
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["tool_call_id"],
                        "content": r["output"],
                        "is_error": r["is_error"],
                    }
                    for r in turn.results
                ]
                messages.append({"role": "user", "content": content})
        return messages

    async def call(self, history: list[Turn], tool_schemas: list[dict]) -> NormalizedResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=self._to_native_tools(tool_schemas),
            messages=self._to_native_messages(history),
        )
        usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

        text = "".join(b.text for b in response.content if b.type == "text") or None
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content if b.type == "tool_use"
        ]
        stop_reason = "tool_use" if response.stop_reason == "tool_use" else "end"
        return NormalizedResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason,usage=usage)