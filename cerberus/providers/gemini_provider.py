import os
import json
from google import genai
from google.genai import types
from cerberus.providers.base import Turn,Usage, UserTurn, AssistantTurn, ToolResultTurn, ToolCall, NormalizedResponse
import base64


class GeminiProvider:
    def __init__(self, model: str,max_tokens:int =1024) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def _to_native_tools(self, tool_schemas: list[dict]) -> list[types.Tool]:
        declarations = [
            types.FunctionDeclaration(
                name=s["name"], description=s["description"], parameters=s["input_schema"]
            )
            for s in tool_schemas
        ]
        return [types.Tool(function_declarations=declarations)]

    def _to_native_contents(self, history: list[Turn]) -> list[types.Content]:
        contents = []
        for turn in history:
            if isinstance(turn, UserTurn):
                contents.append(types.Content(role="user", parts=[types.Part(text=turn.content)]))
            elif isinstance(turn, AssistantTurn):
                parts = []
                if turn.text:
                    parts.append(types.Part(text=turn.text))
                for tc in turn.tool_calls:
                    part = types.Part(function_call=types.FunctionCall(name=tc.name, args=tc.input))
                    if tc.thought_signature:
                        # signature was captured as base64 text for safe JSON storage; decode back to bytes for the SDK
                        part.thought_signature = base64.b64decode(tc.thought_signature)
                    parts.append(part)
                contents.append(types.Content(role="model", parts=parts))
            elif isinstance(turn, ToolResultTurn):
                parts = [
                    types.Part(function_response=types.FunctionResponse(name=r["name"], response={"output": r["output"]}))
                    for r in turn.results
                ]
                contents.append(types.Content(role="user", parts=parts))
        return contents

    async def call(self, history: list[Turn], tool_schemas: list[dict]) -> NormalizedResponse:
        response = self.client.models.generate_content(
            model=self.model,
            contents=self._to_native_contents(history),
            config=types.GenerateContentConfig(tools=self._to_native_tools(tool_schemas)),
        )
        candidate = response.candidates[0]
        text = None
        tool_calls = []
        for part in candidate.content.parts:
            if part.text:
                text = (text or "") + part.text
            if part.function_call:
                sig = getattr(part, "thought_signature", None)
                sig_b64 = base64.b64encode(sig).decode() if sig else None
                tool_calls.append(ToolCall(
                    id=part.function_call.name,
                    name=part.function_call.name,
                    input=dict(part.function_call.args),
                    thought_signature=sig_b64,
                ))
        stop_reason = "tool_use" if tool_calls else "end"
        usage = Usage(
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
        )
        return NormalizedResponse(text=text, tool_calls=tool_calls, stop_reason=stop_reason, usage=usage)