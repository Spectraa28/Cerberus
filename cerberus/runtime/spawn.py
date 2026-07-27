from typing import Literal
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool
from cerberus.providers.base import Provider, Turn
from cerberus.runtime.agent import Runtime


def spawn_sub_agent(
    provider: Provider,
    parent_registry: ToolRegistry,
    input_models: dict[str, type],
    allowed_prefixes: list[str],
    mode: Literal["isolated", "context_seeded"] = "isolated",
    parent_history: list[Turn] | None = None,
    max_turns: int = 5,
    shell_allowed_commands: set[str] | None = None,
) -> tuple[Runtime, list[Turn] | None]:
    scoped_registry = parent_registry.scoped(*allowed_prefixes)

    # if shell is in scope and a command allowlist was given, swap in a restricted instance
    if "shell_exec" in scoped_registry.all() and shell_allowed_commands is not None:
        restricted_shell = ShellExecTool(allowed_commands=shell_allowed_commands)
        scoped_registry.override("shell_exec", restricted_shell)

    scoped_input_models = {
        name: model for name, model in input_models.items() if name in scoped_registry.all()
    }
    runtime = Runtime(provider, scoped_registry, scoped_input_models, max_turns=max_turns)

    seed = parent_history if mode == "context_seeded" else None
    return runtime, seed