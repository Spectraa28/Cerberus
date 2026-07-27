import asyncio
import re
from pathlib import Path
from pydantic import BaseModel
from cerberus.tools.base import ToolSpec, ToolResult, AgentContext


class SearchFilesInput(BaseModel):
    pattern: str            # regex pattern to search for
    path: str = "."         # directory to search in (relative to ctx.cwd)
    file_glob: str = "*"    # e.g. "*.py"
    max_results: int = 50


class SearchFilesTool:
    spec = ToolSpec(
        name="search_files",
        description="Search files for a regex pattern, using ripgrep if available.",
        permission="read",
    )

    async def run(self, input: SearchFilesInput, ctx: AgentContext) -> ToolResult:
        if await self._has_ripgrep():
            return await self._run_ripgrep(input, ctx)
        return self._run_python_fallback(input, ctx)

    async def _has_ripgrep(self) -> bool:
        proc = await asyncio.create_subprocess_shell(
            "command -v rg",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        return await proc.wait() == 0

    async def _run_ripgrep(self, input: SearchFilesInput, ctx: AgentContext) -> ToolResult:
        cmd = (
            f"rg --line-number --glob {input.file_glob!r} "
            f"-e {input.pattern!r} {input.path!r} "
            f"| head -n {input.max_results}"
        )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=ctx.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            output = stdout.decode(errors="replace")
            # rg returns exit code 1 when no matches — that's not an error
            if proc.returncode not in (0, 1):
                return ToolResult(ok=False, output="", error=stderr.decode(errors="replace"))
            return ToolResult(ok=True, output=output or "(no matches)", error=None)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

    def _run_python_fallback(self, input: SearchFilesInput, ctx: AgentContext) -> ToolResult:
        try:
            base = Path(ctx.cwd) / input.path
            regex = re.compile(input.pattern)
            matches: list[str] = []
            for file in base.rglob(input.file_glob):
                if not file.is_file() or len(matches) >= input.max_results:
                    continue
                try:
                    for lineno, line in enumerate(file.read_text(errors="ignore").splitlines(), 1):
                        if regex.search(line):
                            matches.append(f"{file}:{lineno}:{line.strip()}")
                            if len(matches) >= input.max_results:
                                break
                except (UnicodeDecodeError, PermissionError):
                    continue
            return ToolResult(ok=True, output="\n".join(matches) or "(no matches)", error=None)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))