import asyncio
from pydantic import BaseModel
from cerberus.tools.base import ToolSpec, ToolResult, AgentContext

class ShellExecInput(BaseModel):
    command:str
    
class ShellExecTool:
    spec= ToolSpec(
        name="shell_exec",
        description="Run a shell command and capture stdout, stderr, and exit code.",
        permission="exec",
    )
    
    def __init__(self, default_timeout: float = 15.0) -> None:
        self.default_timeout = default_timeout
    
    async def run(self,input:ShellExecInput,ctx:AgentContext) -> ToolResult:
        timeout = input.timeout or self.default_timeout

        try:
            proc = await asyncio.create_subprocess_shell(
                input.command,
                cwd=ctx.cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout,stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=input.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(ok=False,output="",error="timed out")
        
            output = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")
            return ToolResult(ok=proc.returncode == 0 , output=output,error=err or None)
        except Exception as e:
            return ToolResult(ok=False,output="",error=str(e))