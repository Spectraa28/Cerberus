import asyncio
import json
import sys
import httpx
import websockets
from rich.console import Console
from rich.panel import Panel

console = Console()

GATEWAY_URL = "http://localhost:8000"
AGENT_ID = "cerberus"  # renamed from "you" — this identifies the agent, not the human


def _truncate(text: str, limit: int = 150) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


class TerminalSession:
    def __init__(self, session_id: str, agent_id: str = AGENT_ID) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.is_paused = False
        self._last_sent_task: str | None = None  # dedupes our own echoed input

    def status_panel(self) -> Panel:
        total = self.total_input_tokens + self.total_output_tokens
        return Panel(
            f"[cyan]session:[/cyan] {self.session_id}   "
            f"[{'red' if self.is_paused else 'green'}]{'⏸ paused' if self.is_paused else '● live'}[/]   "
            f"[yellow]tokens[/yellow] in:{self.total_input_tokens} out:{self.total_output_tokens} total:{total}",
            border_style="dim",
            padding=(0, 1),
        )

    def print_status(self) -> None:
        console.print(self.status_panel())

    def render_event(self, event: dict) -> None:
        etype = event["type"]
        payload = event["payload"]
        agent = event["agent_id"]

        if etype == "user":
            content = payload["content"]
            if content == self._last_sent_task:
                # we just typed this ourselves — the terminal already echoed it, skip the duplicate
                self._last_sent_task = None
                return
            console.print(f"[bold cyan]you[/bold cyan] [dim]›[/dim] {content}")

        elif etype == "assistant":
            if payload.get("text"):
                console.print(f"[bold green]{agent}[/bold green]: {payload['text']}")
            for tc in payload.get("tool_calls", []):
                console.print(f"[yellow]{agent} → {tc['name']}({_truncate(str(tc['input']))})[/yellow]")
            usage = payload.get("usage")
            if usage:
                self.total_input_tokens += usage["input_tokens"]
                self.total_output_tokens += usage["output_tokens"]

        elif etype == "tool_result":
            for r in payload["results"]:
                style = "red" if r["is_error"] else "dim"
                console.print(f"[{style}]  ↳ {_truncate(r['output'], 300)}[/{style}]")

        elif etype == "status" and payload.get("status") == "paused":
            self.is_paused = True
            console.print(f"[red]— {agent} paused —[/red]")
            self.print_status()

        elif etype == "summary":
            console.print("[magenta]— conversation compacted —[/magenta]")


async def listen(session: TerminalSession, ws_url: str) -> None:
    async with websockets.connect(ws_url) as ws:
        async for raw in ws:
            event = json.loads(raw)
            session.render_event(event)


async def run_task(session: TerminalSession, task: str) -> None:
    session._last_sent_task = task  # so render_event knows to skip the echo-back of this exact message
    async with httpx.AsyncClient(timeout=120.0) as client:
        await client.post(
            f"{GATEWAY_URL}/sessions/{session.session_id}/run",
            json={"task": task, "agent_id": session.agent_id},
        )


async def pause(session: TerminalSession) -> None:
    async with httpx.AsyncClient() as client:
        await client.post(f"{GATEWAY_URL}/sessions/{session.session_id}/pause", json={"agent_id": session.agent_id})
    session.is_paused = True
    session.print_status()


async def resume(session: TerminalSession) -> None:
    session.is_paused = False
    async with httpx.AsyncClient(timeout=120.0) as client:
        await client.post(f"{GATEWAY_URL}/sessions/{session.session_id}/resume", json={"agent_id": session.agent_id})
    session.print_status()


async def create_session() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{GATEWAY_URL}/sessions", json={"label": "terminal session"})
        return resp.json()["session_id"]


async def input_loop(session: TerminalSession) -> None:
    console.print("[dim]Type a task, or /pause /resume /status /quit[/dim]\n")
    while True:
        line = await asyncio.to_thread(console.input, "[bold cyan]>[/bold cyan] ")
        line = line.strip()
        if not line:
            continue
        if line == "/quit":
            console.print("[dim]bye[/dim]")
            return
        elif line == "/pause":
            await pause(session)
        elif line == "/resume":
            await resume(session)
        elif line == "/status":
            session.print_status()
        else:
            asyncio.create_task(run_task(session, line))


async def main() -> None:
    session_id = sys.argv[1] if len(sys.argv) > 1 else await create_session()
    session = TerminalSession(session_id)

    console.print(f"[bold]Cerberus[/bold] — connected to session [cyan]{session_id}[/cyan]")
    session.print_status()

    ws_url = f"{GATEWAY_URL.replace('http://', 'ws://')}/ws/{session_id}"
    asyncio.create_task(listen(session, ws_url))

    await input_loop(session)


if __name__ == "__main__":
    asyncio.run(main())