import os
import json
from openai import OpenAI
from cerberus.providers.base import Turn, UserTurn, AssistantTurn, ToolResultTurn, ToolCall, NormalizedResponse


class OpenAIProvider:
    def __init__(self, model: str, max_tokens:int=1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _to_native_tools(self, tool_schemas: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": s["name"],
                    "description": s["description"],
                    "parameters": s["input_schema"],
                },
            }
            for s in tool_schemas
        ]

    def _to_native_messages(self, history: list[Turn]) -> list[dict]:
        messages = []
        for turn in history:
            if isinstance(turn, UserTurn):
                messages.append({"role": "user", "content": turn.content})
            elif isinstance(turn, AssistantTurn):
                msg = {"role": "assistant", "content": turn.text}
                if turn.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.input)},
                        }
                        for tc in turn.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(turn, ToolResultTurn):
                for r in turn.results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["tool_call_id"],
                        "content": r["output"],
                    })
        return messages

    async def call(self, history: list[Turn], tool_schemas: list[dict]) -> NormalizedResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            tools=self._to_native_tools(tool_schemas),
            messages=self._to_native_messages(history),
        )
        choice = response.choices[0]
        msg = choice.message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        stop_reason = "tool_use" if tool_calls else "end"
        return NormalizedResponse(text=msg.content, tool_calls=tool_calls, stop_reason=stop_reason)