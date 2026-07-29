from typing import Literal
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool
from cerberus.providers.base import Provider, Turn, AssistantTurn, ToolResultTurn
from cerberus.runtime.agent import Runtime
from cerberus.providers.base import UserTurn
from cerberus.runtime.session import EventLog


def _filter_history_to_scope(history: list[Turn], allowed_tool_names: set[str]) -> list[Turn]:
    """
    Collapse any tool_calls/tool_results referencing tools outside the
    sub-agent's scope into plain text turns, so the sub-agent keeps the
    *information* from the parent's earlier tool use without the raw
    tool_use/tool_result schema reference some providers reject.
    """
    filtered: list[Turn] = []
    pending_summaries: dict[str, str] = {}  # call_id -> "tool_name(args) -> "

    for turn in history:
        if isinstance(turn, AssistantTurn):
            kept_calls = [tc for tc in turn.tool_calls if tc.name in allowed_tool_names]
            out_of_scope = [tc for tc in turn.tool_calls if tc.name not in allowed_tool_names]

            for tc in out_of_scope:
                pending_summaries[tc.id] = f"[earlier tool call: {tc.name}({tc.input})]"

            if turn.text or kept_calls:
                filtered.append(AssistantTurn(text=turn.text, tool_calls=kept_calls))

        elif isinstance(turn, ToolResultTurn):
            kept_results = []
            text_lines = []
            for r in turn.results:
                if r["tool_call_id"] in pending_summaries:
                    text_lines.append(f"{pending_summaries[r['tool_call_id']]} -> {r['output']}")
                else:
                    kept_results.append(r)

            if text_lines:
                filtered.append(UserTurn(content="Context from earlier in this task:\n" + "\n".join(text_lines)))
            if kept_results:
                filtered.append(ToolResultTurn(results=kept_results))

        else:
            filtered.append(turn)

    return filtered

def spawn_sub_agent(
    provider: Provider,
    parent_registry: ToolRegistry,
    input_models: dict[str, type],
    allowed_prefixes: list[str],
    mode: Literal["isolated", "context_seeded"] = "isolated",
    parent_history: list[Turn] | None = None,
    max_turns: int = 5,
    shell_allowed_commands: set[str] | None = None,
    event_log: EventLog | None = None,   # NEW
) -> tuple[Runtime, list[Turn] | None]:
    scoped_registry = parent_registry.scoped(*allowed_prefixes)

    if "shell_exec" in scoped_registry.all() and shell_allowed_commands is not None:
        restricted_shell = ShellExecTool(allowed_commands=shell_allowed_commands)
        scoped_registry.override("shell_exec", restricted_shell)

    scoped_input_models = {
        name: model for name, model in input_models.items() if name in scoped_registry.all()
    }
    runtime = Runtime(
        provider, scoped_registry, scoped_input_models,
        max_turns=max_turns, event_log=event_log,   # NEW — this was the missing piece
    )

    seed = None
    if mode == "context_seeded" and parent_history:
        seed = _filter_history_to_scope(parent_history, allowed_tool_names=set(scoped_registry.all()))

    return runtime, seed