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
from cerberus.runtime.agent import Runtime, reconstruct_history
from cerberus.runtime.spawn import spawn_sub_agent
from cerberus.runtime.session import EventLog
from cerberus.tools.search import SearchFilesTool, SearchFilesInput
from cerberus.tools.search_web import SearchWebTool, SearchWebInput

# --- Shared state, built once at module load ---
event_log = EventLog()
_connections: dict[str, set[WebSocket]] = {}
_pause_flags: dict[str, bool] = {}  # key: f"{session_id}:{agent_id}"

_config = load_config()
_registry = ToolRegistry()
_registry.register(ShellExecTool(default_timeout=_config.tools.shell_default_timeout), category="shell")
_registry.register(SearchFilesTool(timeout=_config.tools.search_timeout, ignore_dirs=set(_config.tools.ignore_dirs)), category="search")
_registry.register(SearchWebTool(), category="search")
_input_models = {
    "shell_exec": ShellExecInput,
    "search_files": SearchFilesInput,
    "search_web": SearchWebInput,
}

def _provider_for(tier: str):
    return get_provider(_config, tier=tier)


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


@app.websocket("/ws/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    _connections.setdefault(session_id, set()).add(websocket)
    try:
        events = await event_log.replay_from(session_id, from_seq=0)
        for e in events:
            await websocket.send_json(e)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _connections[session_id].discard(websocket)


class CreateSessionRequest(BaseModel):
    label: str | None = None


@app.post("/sessions")
async def create_session(req: CreateSessionRequest):
    session_id = await event_log.create_session(label=req.label)
    return {"session_id": session_id}


class RunRequest(BaseModel):
    task: str
    agent_id: str = "gateway-agent"
    tier: str = "fast"


@app.post("/sessions/{session_id}/run")
async def run_agent(session_id: str, req: RunRequest):
    provider = _provider_for(req.tier)
    runtime = Runtime(provider, _registry, _input_models, max_turns=_config.runtime.max_turns,
                       event_log=event_log, pause_checker=_check_pause)
    ctx = AgentContext(agent_id=req.agent_id, session_id=session_id, cwd=".")
    answer = await runtime.run(req.task, ctx)
    return {"agent_id": req.agent_id, "answer": answer}


class SpawnRequest(BaseModel):
    task: str
    parent_agent_id: str
    sub_agent_id: str
    allowed_prefixes: list[str]
    mode: str = "isolated"  # "isolated" | "context_seeded"
    shell_allowed_commands: list[str] | None = None
    tier: str = "fast"


@app.post("/sessions/{session_id}/spawn")
async def spawn_agent(session_id: str, req: SpawnRequest):
    provider = _provider_for(req.tier)

    parent_history = None
    if req.mode == "context_seeded":
        parent_history = await reconstruct_history(event_log, session_id, req.parent_agent_id)

    sub_runtime, seed = spawn_sub_agent(
        provider, _registry, _input_models,
        allowed_prefixes=req.allowed_prefixes,
        mode=req.mode,
        parent_history=parent_history,
        max_turns=_config.runtime.max_turns,
        shell_allowed_commands=set(req.shell_allowed_commands) if req.shell_allowed_commands else None,
        event_log=event_log,   # NEW — the fix
    )

    ctx = AgentContext(agent_id=req.sub_agent_id, session_id=session_id, cwd=".")
    answer = await sub_runtime.run(req.task, ctx, seed_history=seed)
    return {"agent_id": req.sub_agent_id, "answer": answer}


def _pause_key(session_id: str, agent_id: str) -> str:
    return f"{session_id}:{agent_id}"

async def _check_pause(ctx: AgentContext) -> bool:
    return _pause_flags.get(_pause_key(ctx.session_id, ctx.agent_id), False)


class PauseRequest(BaseModel):
    agent_id: str

@app.post("/sessions/{session_id}/pause")
async def pause_agent(session_id: str, req: PauseRequest):
    _pause_flags[_pause_key(session_id, req.agent_id)] = True
    return {"paused": True}

@app.post("/sessions/{session_id}/unpause")
async def unpause_agent(session_id: str, req: PauseRequest):
    _pause_flags[_pause_key(session_id, req.agent_id)] = False
    return {"paused": False}

class ResumeRequest(BaseModel):
    agent_id: str
    tier: str = "fast"

@app.post("/sessions/{session_id}/resume")
async def resume_agent(session_id: str, req: ResumeRequest):
    _pause_flags[_pause_key(session_id, req.agent_id)] = False
    provider = _provider_for(req.tier)
    runtime = Runtime(provider, _registry, _input_models, max_turns=_config.runtime.max_turns,
                       event_log=event_log, pause_checker=_check_pause)
    ctx = AgentContext(agent_id=req.agent_id, session_id=session_id, cwd=".")
    answer = await runtime.resume(ctx)
    return {"agent_id": req.agent_id, "answer": answer}