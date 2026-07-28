# seed_session.py — temporary, just to create test data
import asyncio
from cerberus.runtime.session import EventLog

async def main():
    log = EventLog()  # same default db_path as the gateway uses
    await log.connect()
    session_id = await log.create_session(label="gateway test")
    await log.append_event(session_id, "agent-1", "user", {"content": "test message"})
    await log.append_event(session_id, "agent-1", "assistant", {"text": "test response"})
    print("session_id:", session_id)
    await log.close()

asyncio.run(main())