from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from cerberus.runtime.session import EventLog

event_log = EventLog()
_connections: dict[str, set[WebSocket]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    await event_log.connect()
    yield
    # shutdown
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


async def broadcast_event(session_id: str, event: dict):
    for ws in _connections.get(session_id, set()):
        await ws.send_json(event)