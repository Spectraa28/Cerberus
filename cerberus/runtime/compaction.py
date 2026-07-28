import tiktoken
from cerberus.providers.base import Provider, Turn, UserTurn, AssistantTurn, ToolResultTurn

_encoding = tiktoken.get_encoding("cl100k_base")

DEFAULT_TOKEN_BUDGET = 6000
KEEP_RECENT_TURNS = 6


def _estimate_tokens(text: str) -> int:
    return len(_encoding.encode(text))


def _estimate_turn_tokens(turn: Turn) -> int:
    if isinstance(turn, UserTurn):
        return _estimate_tokens(turn.content)
    if isinstance(turn, AssistantTurn):
        total = _estimate_tokens(turn.text or "")
        for tc in turn.tool_calls:
            total += _estimate_tokens(f"{tc.name}({tc.input})")
        return total
    if isinstance(turn, ToolResultTurn):
        return sum(_estimate_tokens(r["output"]) for r in turn.results)
    return 0


def estimate_history_tokens(history: list[Turn]) -> int:
    return sum(_estimate_turn_tokens(t) for t in history)


def _history_to_text(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        if isinstance(t, UserTurn):
            lines.append(f"User: {t.content}")
        elif isinstance(t, AssistantTurn):
            if t.text:
                lines.append(f"Assistant: {t.text}")
            for tc in t.tool_calls:
                lines.append(f"Assistant called {tc.name}({tc.input})")
        elif isinstance(t, ToolResultTurn):
            for r in t.results:
                lines.append(f"Tool {r['name']} result: {r['output'][:500]}")
    return "\n".join(lines)


async def _summarize(provider: Provider, turns: list[Turn]) -> str:
    prompt = (
        "Summarize this agent conversation for the purpose of continuing the task later. "
        "Do NOT write a general narrative recap. Extract and organize ONLY into these sections, "
        "omitting any section that has nothing relevant:\n\n"
        "## Decisions\n"
        "Concrete choices made (design choices, names picked, approaches selected over alternatives).\n\n"
        "## Constraints & Requests\n"
        "Explicit behavioral instructions, requirements, or preferences stated during this conversation "
        "that must still be honored going forward.\n\n"
        "## Completed\n"
        "What has actually been built, verified, or confirmed working — be specific (file names, "
        "test results, what was proven, not just attempted).\n\n"
        "## Open / Unresolved\n"
        "Anything left incomplete, still broken, or explicitly deferred.\n\n"
        "Be concise within each section — bullet points, not paragraphs. Omit anything not clearly "
        "supported by the conversation below.\n\n"
        "--- Conversation to summarize ---\n"
        + _history_to_text(turns)
    )
    response = await provider.call([UserTurn(content=prompt)], tool_schemas=[])
    return response.text or "(summary generation failed)"


async def compact_if_needed(
    history: list[Turn],
    provider: Provider,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    keep_recent: int = KEEP_RECENT_TURNS,
) -> tuple[list[Turn], str | None]:
    """
    Returns (possibly-compacted history, summary_text or None).
    Only compacts when over budget AND there's enough history to compact
    without eating into the recent turns being preserved.
    """
    if estimate_history_tokens(history) <= token_budget or len(history) <= keep_recent:
        return history, None

    to_summarize = history[:-keep_recent]
    recent = history[-keep_recent:]
    summary_text = await _summarize(provider, to_summarize)
    summary_turn = UserTurn(content=f"[Summary of earlier conversation]\n{summary_text}")
    return [summary_turn] + recent, summary_text