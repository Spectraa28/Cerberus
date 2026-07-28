from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from cerberus.config import load_config
from cerberus.providers.factory import get_provider
from cerberus.tools.registry import ToolRegistry
from cerberus.tools.shell import ShellExecTool, ShellExecInput
from cerberus.tools.base import AgentContext
from cerberus.runtime.agent import Runtime
from cerberus.runtime.session import EventLog


# --- Shared state, built once at module load ---
event_log = EventLog()
_connections: dict[str, set[WebSocket]] = {}

_config = load_config()
_registry = ToolRegistry()
_registry.register(ShellExecTool(default_timeout=_config.tools.shell_default_timeout), category="shell")
_input_models = {"shell_exec": ShellExecInput}
_provider = get_provider(_config, tier="fast")


async def broadcast_event(session_id: str, event: dict):
    for ws in _connections.get(session_id, set()):
        await ws.send_json(event)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await event_log.connect()
    event_log.set_broadcaster(broadcast_event)
    yield
    await event_log.close()


app = FastAPI(lifespan=lifespan)


# --- WebSocket: attach to a session, replay history, then receive live events ---
@app.websocket("/ws/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _connections.setdefault(session_id, set()).add(websocket)

    try:
        events = await event_log.replay_from(session_id, from_seq=0)
        for e in events:
            await websocket.send_json(e)

        while True:
            await websocket.receive_text()  # keep-alive; client->server commands go here later
    except WebSocketDisconnect:
        _connections[session_id].discard(websocket)


# --- HTTP: create a session ---
class CreateSessionRequest(BaseModel):
    label: str | None = None


@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    session_id = await event_log.create_session(label=req.label)
    return {"session_id": session_id}


# --- HTTP: run an agent against a session, using the SAME EventLog instance
# that has the broadcaster registered, so every append_event() call
# genuinely pushes live to any connected WebSocket clients ---
class RunRequest(BaseModel):
    task: str
    agent_id: str = "gateway-agent"


@app.post("/sessions/{session_id}/run")
async def run_agent(session_id: str, req: RunRequest):
    runtime = Runtime(
        _provider, _registry, _input_models,
        max_turns=_config.runtime.max_turns, event_log=event_log,
    )
    ctx = AgentContext(agent_id=req.agent_id, session_id=session_id, cwd=".")
    answer = await runtime.run(req.task, ctx)
    return {"answer": answer}