import json
import time
import uuid
import asyncio
import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    label TEXT
);

CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    type TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_events_session_seq ON events(session_id, seq);
"""


class EventLog:
    def __init__(self, db_path: str = "cerberus_sessions.db") -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()  # serializes appends across ALL concurrent agents

    async def connect(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    async def create_session(self, label: str | None = None) -> str:
        session_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO sessions (session_id, created_at, label) VALUES (?, ?, ?)",
            (session_id, time.time(), label),
        )
        await self._db.commit()
        return session_id

    async def append_event(self, session_id: str, agent_id: str, event_type: str, payload: dict) -> int:
        async with self._write_lock:  # read-next-seq + insert is now atomic w.r.t. other coroutines
            async with self._db.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM events WHERE session_id = ?", (session_id,)
            ) as cur:
                row = await cur.fetchone()
                next_seq = row[0]

            await self._db.execute(
                "INSERT INTO events (session_id, seq, ts, type, agent_id, payload) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, next_seq, time.time(), event_type, agent_id, json.dumps(payload)),
            )
            await self._db.commit()
            return next_seq

    async def replay_from(self, session_id: str, from_seq: int = 0) -> list[dict]:
        events = []
        async with self._db.execute(
            "SELECT seq, ts, type, agent_id, payload FROM events "
            "WHERE session_id = ? AND seq >= ? ORDER BY seq ASC",
            (session_id, from_seq),
        ) as cur:
            async for seq, ts, event_type, agent_id, payload in cur:
                events.append({
                    "seq": seq, "ts": ts, "type": event_type,
                    "agent_id": agent_id, "payload": json.loads(payload),
                })
        return events