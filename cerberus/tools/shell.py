import asyncio
import re
import shlex
from pydantic import BaseModel
from cerberus.tools.base import ToolSpec, ToolResult, AgentContext

# splits on shell control operators without breaking quoted strings apart
_SHELL_OPERATORS = re.compile(r'(\|\||&&|;|\||&)')


def _extract_binaries(command: str) -> list[str]:
    """Best-effort extraction of every binary a compound shell command would invoke."""
    segments = _SHELL_OPERATORS.split(command)
    binaries = []
    for seg in segments:
        seg = seg.strip()
        if not seg or seg in ("||", "&&", ";", "|", "&"):
            continue
        try:
            tokens = shlex.split(seg)
        except ValueError:
            tokens = seg.split()
        if tokens:
            binaries.append(tokens[0])
    return binaries


class ShellExecInput(BaseModel):
    command: str
    timeout: float | None = None


class ShellExecTool:
    spec = ToolSpec(
        name="shell_exec",
        description="Run a shell command and capture stdout, stderr, and exit code.",
        permission="exec",
    )

    def __init__(self, default_timeout: float = 15.0, allowed_commands: set[str] | None = None) -> None:
        self.default_timeout = default_timeout
        self.allowed_commands = allowed_commands  # None = unrestricted

    def _check_allowed(self, command: str) -> str | None:
        """Returns an error string if any binary in the command isn't allowed, else None."""
        if self.allowed_commands is None:
            return None
        binaries = _extract_binaries(command)
        for b in binaries:
            if b not in self.allowed_commands:
                return f"command '{b}' is not permitted for this agent (allowed: {sorted(self.allowed_commands)})"
        return None

    async def run(self, input: ShellExecInput, ctx: AgentContext) -> ToolResult:
        denial = self._check_allowed(input.command)
        if denial:
            return ToolResult(ok=False, output="", error=denial)

        timeout = input.timeout if input.timeout is not None else self.default_timeout
        try:
            proc = await asyncio.create_subprocess_shell(
                input.command,
                cwd=ctx.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(ok=False, output="", error="timed out")

            output = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            return ToolResult(ok=proc.returncode == 0, output=output, error=err or None)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))