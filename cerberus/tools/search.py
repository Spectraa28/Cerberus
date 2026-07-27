import asyncio
import re
from pathlib import Path
from pydantic import BaseModel
from cerberus.tools.base import ToolSpec, ToolResult, AgentContext


class SearchFilesInput(BaseModel):
    pattern: str | None = None  # regex to search file CONTENTS; omit to just list matching files
    path: str = "."
    file_glob: str = "*"
    max_results: int = 50


class SearchFilesTool:
    spec = ToolSpec(
        name="search_files",
        description=(
            "Find files by name/glob, optionally filtering by content regex. "
            "Omit 'pattern' to just list files matching file_glob; provide it to search their contents."
        ),
        permission="read",
    )

    async def run(self, input: SearchFilesInput, ctx: AgentContext) -> ToolResult:
        if input.pattern is None:
            return await self._list_files(input, ctx)
        if await self._has_ripgrep():
            return await self._run_ripgrep(input, ctx)
        return self._run_python_fallback(input, ctx)

    async def _list_files(self, input: SearchFilesInput, ctx: AgentContext) -> ToolResult:
        try:
            base = Path(ctx.cwd) / input.path
            ignore_dirs = {".venv", "venv", "__pycache__", ".git", "node_modules"}

            all_files = [
                f for f in base.rglob(input.file_glob)
                if f.is_file() and not any(part in ignore_dirs for part in f.parts)
            ]
            truncated = len(all_files) > input.max_results
            files = [str(f) for f in all_files[: input.max_results]]

            output = "\n".join(files) or "(no files found)"
            if truncated:
                output += f"\n... truncated, showing {input.max_results} of {len(all_files)} total"

            return ToolResult(ok=True, output=output, error=None)
        except Exception as e:
            return ToolResult(ok=False, output="", error=str(e))

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