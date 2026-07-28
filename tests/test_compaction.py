from cerberus.providers.base import UserTurn, AssistantTurn
from cerberus.runtime.compaction import compact_if_needed, estimate_history_tokens


async def test_compaction_triggers_and_produces_structured_summary(provider):
    # Build a long synthetic history with clear decisions/constraints/completions,
    # padded to force it well over a small token budget.
    history = [
        UserTurn(content="Let's build Cerberus, a multi-agent harness. Use Python and uv, not TypeScript."),
        AssistantTurn(text="Decision made: Python with uv for dependency management, confirmed."),
    ]
    for i in range(8):
        history.append(UserTurn(
            content=f"Please implement feature {i}, and remember: always use the shell_ prefix for shell tools. "
            + ("padding to increase token count " * 15)
        ))
        history.append(AssistantTurn(
            text=f"Implemented and verified feature {i} — tests passed, output confirmed correct. "
            + ("more detail padding here " * 15)
        ))

    original_tokens = estimate_history_tokens(history)
    assert original_tokens > 600, f"synthetic history too small to force compaction: {original_tokens} tokens"

    compacted, summary = await compact_if_needed(history, provider, token_budget=500, keep_recent=4)

    assert summary is not None, "compaction should have triggered given the low budget"
    assert len(compacted) < len(history), "compacted history should be shorter than original"

    # Structural check: at least the sections with real content should appear
    assert "Decisions" in summary or "Constraints" in summary
    assert "Completed" in summary

    print("\n--- Generated summary ---\n" + summary)